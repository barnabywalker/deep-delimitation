from turtle import forward
import torch

import torch.nn.functional as F
import pytorch_lightning as pl
from collections import defaultdict

from torch import nn
from torchmetrics.functional import accuracy as tm_accuracy
from torchmetrics.functional.classification.f_beta import f1_score

from .projection import ProjectionHead

TARGETS = ["name", "species", "genus", "family", "order"]

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes

        self.model = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Dropout(p=0.25),
            nn.Linear(self.input_dim, self.input_dim, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(self.input_dim),
            nn.Dropout(p=0.5),
            nn.Linear(self.input_dim, self.n_classes, bias=False)
        )

    def forward(self, x):
        return self.model(x)

class Classifier(pl.LightningModule):
    """A lightning module for a classifier network set up for identification of plant taxa from
    herbarium specimens.

    The model allows the user to specify a pre-defined encoder that is then attached to a classification
    head. Passing an index and mapping of taxonomic classes allows prediction at all levels of the taxonomic
    hierarchy, which can be incorporated into the loss function.

    args:
        encoder: a pre-defined network architecture to use as the backbone of the model e.g., `torchvision.models.resnet18`.
        target: the taxonomic level to predict at, one of [name, species, genus, family, order].
        class_index: an index of classes for items in the dataset.
        class_map: a dictionary mapping betweem the taxonomic levels for each name in the dataset.
    
    kwargs:
        hierarchical: whether to use hierarchical loss or not.
        lr: the learning rate.
        scheduler: the name of a learning rate scheduler e.g., 'one_cycle'.
        max_lr: the maximum learning rate, only used if a scheduler is specified.
        nsteps: the number of steps to run the scheduler for, can be calculated as the number of training batches times the number of epochs.
    """
    def __init__(self, encoder, target, class_index, class_map, hierarchical=False,
                 hidden_dim=None, feat_dim=128, proj_layers=1, lr=1e-3, scheduler=None, max_lr=None, nsteps=None):
        super().__init__()
        
        # build tensor mapping for hierarchical class probabilities
        self.hierarchical = hierarchical
        self.target_idx = TARGETS.index(target)
        self.class_sizes = dict()
        for class_name in TARGETS[self.target_idx+1:]:
            target_map, map_idx, class_sizes = build_class_maps_(class_index, class_map, target, class_name)
            self.register_buffer(f"map_{class_name}", target_map)
            self.register_buffer(f"idx_{class_name}", map_idx)
            self.class_sizes[class_name] = class_sizes
        
        self.num_classes = len(class_index[target])
        self.encoder = encoder
        input_dim = [params for params in encoder.parameters()][-1].shape[0]
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        self.projection = ProjectionHead(input_dim, hidden_dim, feat_dim, n_layers=proj_layers)
        
        self.fc = ClassificationHead(feat_dim, self.num_classes)

        self.lr = lr
        self.scheduler = scheduler
        self.max_lr = lr if max_lr is None else max_lr
        self.nsteps = nsteps

    def loss_fcn(self, x, y):
        loss = F.nll_loss(x, y)
        if self.hierarchical:
            for class_name in TARGETS[self.target_idx+1:]:
                target_map = getattr(self, f"map_{class_name}")
                map_idx = getattr(self, f"idx_{class_name}")
                sizes = self.class_sizes[class_name]
                x_new = marginalise_logits_(x, map_idx, sizes)
                y_new = target_map[y]
                loss += F.nll_loss(x_new, y_new)

            loss = loss / (len(TARGETS[self.target_idx+1:]) + 1)

        return loss

    def forward(self, x):
        feats = self.encoder(x)
        feats = self.projection(feats)
        out = self.fc(feats)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = self.loss_fcn(logits, y)

        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None, dataloader_idx=0):
        x, y = batch
        logits = self(x)
        loss = self.loss_fcn(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = tm_accuracy(preds, y)
        top5 = tm_accuracy(logits, y, top_k=5)
        f1_macro = f1_score(preds, y, average="macro", num_classes=self.num_classes)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_top5", top5, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_f1-macro", f1_macro, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx, dataloader_idx):
        self.evaluate(batch, "test", dataloader_idx)

    def configure_optimizers(self):
        if self.scheduler is None:
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        
        optimiser = torch.optim.SGD(self.parameters(), lr=self.lr)
        if self.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceOnPlateau(optimiser, "min")
        elif self.scheduler == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=self.max_lr, total_steps=self.nsteps)

        return [optimiser], [{"scheduler": scheduler, "interval": "step", "name": f"{self.scheduler}_lr"}]
            

def build_class_maps_(class_index, class_mapping, class_from, class_to):
    """Build a vector to map between two taxonomic levels.

    Args:
        class_index: A list or dict specifying the class for each item in a dataset.
        class_mapping: A list or dict mapping between the taxonomic levels for each class.
        class_from: The taxonomic level to map from.
        class_to: The taxonomic level to map to.
    """
    target_map = torch.zeros((len(class_index[class_from])), dtype=torch.int64)
    pred_map = defaultdict(set)
    for taxon in class_mapping:
        lower = taxon[class_from]
        higher = taxon[class_to]

        lower_idx = class_index[class_from][lower]
        higher_idx = class_index[class_to][higher]

        target_map[lower_idx] = higher_idx
        pred_map[higher_idx].update([lower_idx])

    map_idx = []
    class_sizes = []
    for i in range(len(pred_map)):
        map_idx.extend(pred_map[i])
        class_sizes.append(len(pred_map[i]))
    map_idx = torch.tensor(map_idx, dtype=torch.int64)
    return target_map, map_idx, class_sizes


def marginalise_logits_(x, map_idx, sizes):
    """Combine the logits to go from a lower taxonomic level to a higher one, so probabilities sum to 1.

    Args:
        x: a vector of logits.
        map_idx: a vector of indices for the classes in the higher taxonomic level.
        sizes: a vector of number of examples in each class at the higher taxonomic level.
    """
    return torch.log(torch.stack([cls.sum(axis=1) for cls in torch.exp(x[:, map_idx]).split(sizes, dim=1)], dim=1))
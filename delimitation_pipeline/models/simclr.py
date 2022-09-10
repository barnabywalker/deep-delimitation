import torch

import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay

from ..losses import ntxent

from .projection import ProjectionHead

class SimCLR(pl.LightningModule):
    def __init__(self, encoder, feat_dim=128, hidden_dim=None, proj_layers=1, temperature=0.1, max_epochs=10,
                 warmup_epochs=1, lr=1e-3, train_iters_per_epoch=100, scheduler=None):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.temperature = temperature
        self.lr = lr
        self.train_iters_per_epoch = train_iters_per_epoch
        self.scheduler = scheduler

        self.encoder = encoder

        input_dim = [params for params in encoder.parameters()][-1].shape[0]
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        self.projection = ProjectionHead(input_dim, hidden_dim, feat_dim, n_layers=proj_layers)

    def forward(self, x):
        feats = self.encoder(x)
        return self.projection(feats)

    def evaluate(self, batch, stage=None, dataloader_idx=0):
        (img1, img2), y = batch

        z1 = self(img1)
        z2 = self(img2)

        return ntxent(z1, z2, self.temperature)
        
    def training_step(self, batch, batch_idx):
        loss = self.evaluate(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.evaluate(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx):
        loss = self.evaluate(batch)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimiser = torch.optim.SGD(self.parameters(), lr=self.lr)

        total_steps = self.train_iters_per_epoch * self.max_epochs
        if self.scheduler == "lambda_lr":
            warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, linear_warmup_decay(warmup_steps, total_steps, cosine=True))
        elif self.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceOnPlateau(optimiser, "min")
        elif self.scheduler == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=self.lr, total_steps=total_steps)
            
        return [optimiser], [{"scheduler": scheduler, "interval": "step", "name": f"{self.scheduler}_lr"}]        

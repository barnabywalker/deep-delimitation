import os

import pytorch_lightning as pl
import numpy as np

from typing import Optional, Callable
from collections import Counter

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
from torchvision import transforms

from ..datasets import HalfEarthDataset
from ..transforms import CropAspect

class HalfEarthModule(pl.LightningDataModule):
    def __init__(self, 
        data_dir: str = "./", 
        target_type: str = "name",
        balanced: bool = False,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        val_split: float = 0.2,
        pin_memory: bool = True,
        drop_last: bool = False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.target = target_type
        self.balanced = balanced
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle = shuffle

        base_tfms = [
            CropAspect(aspect=1),
            transforms.ToTensor()
        ]
        
        self.train_transform = transforms.Compose([
            *base_tfms,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 10)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(size=256, scale=(0.75, 1.0), ratio=(1.0,1.0))
        ])

        self.val_transform = transforms.Compose([
            *base_tfms,
            transforms.Resize(256)
        ])

    def prepare_data(self):
        root = os.path.join(self.data_dir, "herbarium-2021-fgvc8/train")
        if not os.path.exists(root):
            HalfEarthDataset(self.data_dir, download=True)

    def setup(self, stage: Optional[str] = None):
        half_earth = HalfEarthDataset(root=self.data_dir, train=True, transform=self.train_transform, target_type=self.target)
        
        if self.target != "id":
            self.num_classes = len(half_earth.categories_index[self.target])
            self.vocab = [""] * self.num_classes
            for name, idx in half_earth.categories_index[self.target].items():
                self.vocab[idx] = name

        # to speed up indexing of test sets
        half_earth_idx = np.array(half_earth.index)
        rng = np.random.default_rng()
        
        # make one test set of all herbaria other than NYBG    
        herb_test_idx = [idx for idx, (_, _, inst, _) in enumerate(half_earth.index) if inst > 0]
        remaining_idx = [idx for idx, (_, _, inst, _) in enumerate(half_earth.index) if inst == 0]
        
        # hold out 10 % of taxa as an independent test set
        remaining_taxa = np.unique(half_earth_idx[remaining_idx, -1].astype(int))
        n_taxa_test = int(len(remaining_taxa) * 0.1)
        hold_out_taxa = rng.choice(remaining_taxa, size=n_taxa_test)

        taxon_test_idx = [idx for idx in remaining_idx if int(half_earth_idx[idx, -1]) in hold_out_taxa]
        remaining_idx = list(set(remaining_idx) - set(taxon_test_idx))
        
        # hold out 10 % of specimens as independent test set
        n_spec_test = int(len(remaining_idx) * 0.1)
        spec_test_idx = list(rng.choice(remaining_idx, size=n_spec_test))

        # use the rest for training and validation
        train_idx = list(set(remaining_idx) - set(spec_test_idx))
        
        if stage == "fit" or stage is None:
            total_length = len(train_idx)
            val_length = int(total_length * self.val_split)
            val_idx = rng.choice(train_idx, size=val_length, replace=False)

            train_idx = list(set(train_idx) - set(val_idx))
            train_set = Subset(half_earth, train_idx)
            val_set = Subset(half_earth, val_idx)

            # don't want to do augmentation transforms on validation set
            val_set.dataset.transform = self.val_transform

            # calculate sample weighting for balanced sampling
            class_counts = Counter([taxon[self.target] for taxon in half_earth.categories_map])
            total_count = sum(class_counts.values())
            class_weights = {k: total_count/v for k, v in class_counts.items()}
            sample_weights = []
            for idx in train_idx:
                _, _, _, class_id = half_earth.index[idx]
                sample_weights.append(class_weights[half_earth.categories_map[class_id][self.target]])
            
            self.sample_weights = sample_weights

            self.half_earth_train = train_set
            self.half_earth_val = val_set

        if stage == "test":
            self.half_earth_tests = [
                Subset(half_earth, herb_test_idx), 
                Subset(half_earth, taxon_test_idx),
                Subset(half_earth, spec_test_idx)
            ]

    def train_dataloader(self):
        sampler = None
        if self.balanced:
            sampler = WeightedRandomSampler(self.sample_weights, len(self.sample_weights))

        shuffle = self.shuffle if sampler is None else None

        return self._data_loader(self.half_earth_train, shuffle=shuffle, sampler=sampler)

    def val_dataloader(self):
        return self._data_loader(self.half_earth_val)

    def test_dataloader(self):
        return [self._data_loader(dl) for dl in self.half_earth_tests]

    def predict_dataloader(self):
        return [self._data_loader(dl) for dl in self.half_earth_tests]

    def _data_loader(self, dataset: Dataset, shuffle: bool = False, sampler: Optional[Callable] = None) -> DataLoader:
        return DataLoader(
            dataset,
            sampler=sampler,
            shuffle=shuffle,
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )

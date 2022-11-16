
import os
import numpy as np
import pytorch_lightning as pl

from typing import Optional
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from delimitation_pipeline.datasets.fern_mask_dataset import SmithsonianFernMaskDataset

from delimitation_pipeline.transforms.transforms import CropAspect, PadAspect

class SmithsonianFernMaskModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        pad: Optional[bool] = False,
        batch_size: Optional[int] = 32,
        shuffle: Optional[bool] = True,
        num_workers: Optional[int] = 0,
        val_split: Optional[float] = 0.2,
        pin_memory: Optional[bool] = True,
        drop_last: Optional[bool] = False
    ) -> None:
        super().__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.val_split = val_split
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        base_tfms = []
        if pad:
            base_tfms.append(PadAspect(aspect=1))
        else:
            base_tfms.append(CropAspect(aspect=1))

        base_tfms.append(transforms.ToTensor())

        self.transforms = transforms.Compose([
            *base_tfms,
            transforms.Resize(256)
        ])

    def prepare_data(self) -> None:
        root = os.path.join(self.data_dir, "images/original_hires_images")
        if not os.path.exists(root):
            SmithsonianFernMaskDataset(self.data_dir, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        ds = SmithsonianFernMaskDataset(self.data_dir, download=False, transform=self.transforms)
        rng = np.random.default_rng()
        
        if stage == "fit":
            total_length = len(ds)
            total_idx = [i for i in range(total_length)]

            val_length = int(total_length * self.val_split)
            val_idx = rng.choice(total_idx, size=val_length, replace=False)

            train_idx = [i for i in total_idx if i not in val_idx]
            train_set = Subset(ds, train_idx)
            val_set = Subset(ds, val_idx)

            self.train = train_set
            self.val = val_set

    def train_dataloader(self):
        return self._data_loader(self.train, shuffle=self.shuffle)

    def val_dataloader(self):
        return self._data_loader(self.val)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )

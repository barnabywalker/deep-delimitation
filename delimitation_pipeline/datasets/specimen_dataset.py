import os
from PIL import Image
from torchvision.datasets.vision import VisionDataset

class SpecimenDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.index = []
        for dirpath, _, fnames in os.walk(root):
            self.index.extend([os.path.join(dirpath, fname) for fname in fnames])

    def __getitem__(self, index):
        fpath = self.index[index]
        img = Image.open(fpath).convert("RGB")

        target = fpath.split("/")[-1].split(".")[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.index)
import json
import kaggle
import zipfile
import os

from PIL import Image
from torch.utils.data import Subset
from torchvision.datasets.vision import VisionDataset
from typing import Union, Callable, List, Optional, Tuple, Any

from delimitation_pipeline.datasets.utils import _download_kaggle, _extract_archive

CATEGORIES = ["order", "family", "genus", "species", "name"]

def download_from_kaggle(
    dataset_name: str, 
    download_root: str, 
    extract_root: Optional[str] = None, 
    remove_finished: bool = False
) -> None:
    if extract_root is None:
        extract_root = download_root
    archive = _download_kaggle(dataset_name, download_root)
    _extract_archive(archive, extract_root, remove_finished=remove_finished)

def _verify_type(value: str, valid_values: Optional[List[str]]=None) -> str:
    if valid_values is None:
        return value
    
    if value not in valid_values:
        raise ValueError(f"Unknown target type '{value}', should be one of {{{valid_values}}}")
        
    return value

class HalfEarthDataset(VisionDataset):
    def __init__(
        self, 
        root: str, 
        target_type: Union[List[str], str] = "name", 
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train: bool = True,
        download: bool = False
    ) -> None:
        version = "herbarium-2021-fgvc8"
        super().__init__(os.path.join(root, version), transform=transform, target_transform=target_transform)
        
        os.makedirs(root, exist_ok=True)
        if download:
            self.download()
        
        if train:
            self.root = os.path.join(self.root, "train")
        else:
            self.root = os.path.join(self.root, "test")
            
        if not self._check_exists():
            raise RuntimeError("Dataset not found. Download it using `download=True`.")
            
        if not isinstance(target_type, list):
            target_type = [target_type]
            
        self.target_type = [_verify_type(t) for t in target_type]
        
        self._load_meta()
    
    def _load_meta(self) -> None:
        with open(os.path.join(self.root, "metadata.json"), "r") as mfile:
            metadata = json.load(mfile)

        # list indexed by cat id with mapping from category type -> image id
        self.categories_map = [{"genus": line["name"].split()[0], "species": " ".join(line["name"].split()[:2]), **line} for line in metadata["categories"]]
        self.institution_map = metadata["institutions"]

        # folder number, image id, category id, institution_id
        self.index: List[Tuple[int, str, str]] = []
        for image_info, annotation in zip(metadata["images"], metadata["annotations"]):
            if os.path.exists(os.path.join(self.root, image_info["file_name"])):
                self.index.append((
                    image_info["file_name"], 
                    annotation["image_id"],
                    annotation["institution_id"],
                    annotation["category_id"]
                ))

        self.categories_index = {k: {} for k in CATEGORIES}
        for category_type in CATEGORIES:
            for item in self.categories_map:
                if item[category_type] not in self.categories_index[category_type]:
                    self.categories_index[category_type][item[category_type]] = len(self.categories_index[category_type])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        fpath, image_id, institution_id, cat_id = self.index[index]
        img = Image.open(os.path.join(self.root, fpath))

        target: Any = []
        for t in self.target_type:
            if t == "id":
                target.append(image_id)
            elif t == "name":
                target.append(cat_id)
            else:
                cat_name = self.categories_map[cat_id][t]
                target.append(self.categories_index[t][cat_name])

        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.index)

    def category_name(self, category_type: str, category_id: int) -> str:
        category_type = _verify_type(category_type)

        for name, idx in self.categories_index[category_type].items():
            if idx == category_id:
                return name
        
        raise ValueError(f"Invalid category ID {category_id} for {category_type}")

    def institution_name(self, institution_id: int) -> str:
        return self.institution_map[institution_id]["name"]

    def _check_exists(self) -> bool:
        return os.path.exists(self.root) and len(os.listdir(self.root)) > 0

    def download(self) -> None:
        if self._check_exists():
            raise RuntimeError(
                f"The directory {self.root} already exists.",
                f"To re-download or re-extract the images, please delete the directory."
            )

        base_root = os.path.dirname(self.root)
        download_from_kaggle("herbarium-2021-fgvc8", base_root)
        
import os
from typing import Any, Callable, Optional, Tuple, List

from PIL import Image
from torchvision.datasets import VisionDataset

from delimitation_pipeline.datasets.utils import _download_file, _extract_archive

IMG_URL = "https://smithsonian.figshare.com/ndownloader/files/17851235"
MASK_URL = "https://smithsonian.figshare.com/ndownloader/files/17851277"

def download_tarfile(
    url: str,
    download_root: str,
    download_name: Optional[str] = None,
    extract_root: Optional[str] = None,
    chunk_size: Optional[int] = 128,
    remove_finished: bool = False
) -> None:
    if extract_root is None:
        extract_root = download_root

    archive = _download_file(url, download_root, download_name=download_name, chunk_size=chunk_size)
    _extract_archive(archive, extract_root, remove_finished=remove_finished)
    

class SmithsonianFernMaskDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True
    ) -> None:
        super().__init__(os.path.join(root), transform=transform, target_transform=target_transform)
        if self.target_transform is None:
            self.target_transform = transform

        os.makedirs(root, exist_ok=True)
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Download it using `download=True`.")

        self._load_paths()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path, mask_path = self.index[index]

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def _load_paths(self) -> None:
        base_dir = os.path.dirname(self.root)
        img_dir = os.path.join(base_dir, "images/original_hires_images")
        mask_dir = os.path.join(base_dir, "masks/hires_masks")

        img_list = [f for f in os.listdir(img_dir) if not f.startswith("._")]

        self.index: List[Tuple[str, str]] = []
        for img in img_list:
            img_id = img.split(".")[0]
            mask = f"{img_id}_mask.jpg"
            self.index.append((os.path.join(img_dir, img), os.path.join(mask_dir, mask)))

    def __len__(self) -> int:
        return len(self.index)

    def _check_exists(self) -> bool:
        return os.path.exists(self.root) and len(os.listdir(self.root)) > 0

    def download(self) -> None:
        if self._check_exists():
            raise RuntimeError(
                f"The directory {self.root} already exists.",
                "To re-download or re-extract the images, please delete the directory."
            )
        
        base_root = os.path.dirname(self.root)

        img_root = os.path.join(base_root, "images")
        if not os.path.exists(img_root):
            os.mkdir(img_root)

        mask_root = os.path.join(base_root, "masks")
        if not os.path.exists(mask_root):
            os.mkdir(mask_root)

        download_tarfile(IMG_URL, img_root, download_name="images.tar.gz", chunk_size=2048)
        download_tarfile(MASK_URL, mask_root, download_name="masks.tar.gz", chunk_size=2048)


import kaggle
import os
import numpy as np
import requests
import zipfile
import tarfile

from typing import Optional
from tqdm import tqdm

def _extract_archive(
    archive: str,
    extract_root: str,
    remove_finished: Optional[bool] = True
):
    print(f"Extracting file from {archive} to {extract_root}")

    if archive.endswith("zip"):
        with zipfile.ZipFile(archive, "r", compression=zipfile.ZIP_STORED) as zip:
            zip.extractall(extract_root)
    elif archive.endswith("tar.gz"):
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extract_root)
    else:
        raise ValueError(f"{archive} not a recognised archive file.")

    if remove_finished:
        os.remove(archive)
    

def _download_file (
    url: str,
    download_root: str,
    download_name: Optional[str] = None,
    chunk_size: Optional[int] = 128
) -> str:
    if download_name is None:
        download_name = url.split("/")[-1]

    archive = os.path.join(download_root, download_name)
    response = requests.get(url, stream=True)

    nbytes = int(response.headers["Content-Length"])
    nchunks = int(np.ceil(nbytes / chunk_size))
    with open(archive, 'wb') as urlfile:
        for chunk in tqdm(response.iter_content(chunk_size=chunk_size), total=nchunks, 
                          desc=f"downloading file from {url} to {download_root}"):
            urlfile.write(chunk)

    return archive


def _download_kaggle(
    dataset_name: str,
    download_root: str
) -> None:
    kaggle.api.competition_download_files(dataset_name, path=download_root)

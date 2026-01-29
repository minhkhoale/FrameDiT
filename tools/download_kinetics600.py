from typing import Any, Dict, List, Optional, Literal
from fractions import Fraction
import csv
import os
import random
from os import path
from pathlib import Path
import urllib
import shutil
from multiprocessing import Pool
from functools import partial
from omegaconf import DictConfig
import torch
from torchvision.io import write_video
from torchvision.datasets.utils import (
    download_and_extract_archive,
    check_integrity,
    download_url,
)
from tqdm import tqdm
import numpy as np
import argparse
# from .utils import read_video, rescale_and_crop

_TAR_URL = "https://s3.amazonaws.com/kinetics/600/{split}/k600_{split}_path.txt"
_ANNOTATION_URLS = "https://s3.amazonaws.com/kinetics/600/annotations/{split}.csv"

def _dl_wrap(tarpath: str, videopath: str, line: str) -> None:
    download_and_extract_archive(line, tarpath, videopath, remove_finished=True)

def download_dataset(save_dir: Path) -> None:
    os.makedirs(save_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        _download_videos(save_dir, split)

def _download_videos(save_dir, split) -> None:
    print(f"Downloading {split} videos...")
    split_folder = save_dir / split
    tar_path = save_dir / "tars"
    file_list_path = save_dir / "files"

    split_url = _TAR_URL.format(split=split)
    split_url_filepath = file_list_path / path.basename(split_url)
    if not check_integrity(split_url_filepath):
        download_url(split_url, file_list_path)

    with open(split_url_filepath) as file:
        list_video_urls = [
            urllib.parse.quote(line, safe="/,:")
            for line in file.read().splitlines()
        ]

    part = partial(_dl_wrap, tar_path, split_folder)
    with Pool(32) as pool:
        list(
            tqdm(
                pool.imap(part, list_video_urls),
                total=len(list_video_urls),
                desc=f"Downloading {split} videos",
            )
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kinetics-600 dataset")
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save the downloaded dataset",
    )

    args = parser.parse_args()
    download_dataset(Path(args.save_dir))
"""
Converts a dataset of images into a dataset of processed images.
Supports resizing, cropping, etc. using torchvision and PIL.
"""

import os
import argparse
from pathlib import Path
from multiprocessing import Pool
from collections import Counter
from typing import List
from PIL import Image
import torchvision.transforms.functional as TVF
from tqdm import tqdm


def process_images(
    source_dir: os.PathLike,
    target_dir: os.PathLike,
    num_workers: int = 8,
    image_exts=('jpg', 'jpeg', 'png'),
    target_size: int = None,
    crop_type: str = 'center',  # center, random, none
    **kwargs
):
    os.makedirs(target_dir, exist_ok=True)

    image_paths = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(image_exts):
                image_paths.append(os.path.join(root, file))

    tasks = [dict(
        image_path=path,
        target_path=os.path.join(target_dir, os.path.basename(os.path.dirname(path)), os.path.basename(path)),
        target_size=target_size,
        crop_type=crop_type
    ) for path in image_paths]

    pool = Pool(processes=num_workers)
    for _ in tqdm(pool.imap_unordered(task_proxy, tasks), total=len(tasks)):
        pass
    pool.close()
    pool.join()

    print(f"Processed {len(tasks)} images.")


def task_proxy(kwargs):
    return process_image(**kwargs)


def process_image(image_path: os.PathLike, target_path: os.PathLike, target_size=None, crop_type='center'):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open image: {image_path} ({e})")
        return

    # # Cropping
    if crop_type == 'center':
        min_size = min(img.size)
        img = TVF.center_crop(img, output_size=min_size)
    elif crop_type == 'random':
        min_size = min(img.size)
        img = TVF.resized_crop(
            img,
            top=0,
            left=0,
            height=min_size,
            width=min_size,
            size=min_size,
            interpolation=Image.LANCZOS
        )

    # Resizing
    if target_size is not None:
        img = TVF.resize(img, size=target_size, interpolation=Image.LANCZOS)

    # Save output
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(target_path, quality=95)


def listdir_full_paths(d) -> List[os.PathLike]:
    return sorted([os.path.join(d, x) for x in os.listdir(d)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert and process images with resizing/cropping')
    parser.add_argument('-s', '--source_dir', type=str, required=True, help='Path to source image folder')
    parser.add_argument('-t', '--target_dir', type=str, required=True, help='Path to save processed images')
    parser.add_argument('--target_size', type=int, help='Resize images to this size')
    parser.add_argument('--crop_type', type=str, default='center', choices=['center', 'random', 'none'], help='Crop type')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker processes')
    args = parser.parse_args()

    process_images(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        target_size=args.target_size,
        crop_type=args.crop_type,
        num_workers=args.num_workers
    )

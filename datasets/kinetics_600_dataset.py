import os
import torch
import random
from pathlib import Path
import torch.utils.data as data
from torchvision.datasets.video_utils import _VideoTimestampsDataset, _collate_fn
from torchvision.io import read_video_timestamps
from typing import Literal, List, Dict, Any, Callable, Tuple, Optional
from decord import VideoReader, cpu
import warnings
import pandas as pd

import numpy as np
import io
import json
from PIL import Image
from decord import VideoReader
from decord import cpu, gpu
from tqdm import tqdm

class SafeVideoTimestampsDataset(torch.utils.data.Dataset):
    """A safer version of _VideoTimestampsDataset that skips broken videos."""
    def __init__(self, video_paths):
        self.video_paths = list(video_paths)

    def __getitem__(self, idx):
        path = str(self.video_paths[idx])
        try:
            vr = VideoReader(path, ctx=cpu(0))
            total_frames = len(vr)
            return total_frames
        except Exception as e:
            warnings.warn(f"[WARN] Skipping corrupted video: {path} ({e})", RuntimeWarning)
            # Return empty to signal failure
            return None

    def __len__(self):
        return len(self.video_paths)

class Kinetics600(data.Dataset):
    def __init__(self, configs, transform, temporal_sample=None, train=True):
        self.configs = configs
        self.data_path = Path(configs.data_path)
        self.metadata_dir = Path(configs.metadata_dir)
        if hasattr(configs, 'annotation_path'):
            self.annotation_path = Path(configs.annotation_path)
            self.annotation, self.class_to_idx = self.load_annotation()
        else:
            self.annotation_path = None
            self.annotation = None
            self.class_to_idx = None
        self.transform = transform

        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.frame_interval = self.configs.frame_interval
        # Preprocessed latents

        self.data_all = self.load_metadata("train" if train else "val")
        print(f"Loaded {len(self.data_all)} videos for {'train' if train else 'val'} split.")
        self.video_num = len(self.data_all)

    def __getitem__(self, index):
        video_path, total_frames = self.data_all[index]
        video_path = str(video_path)
        # Sampling video frames
        if self.temporal_sample is not None:
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert end_frame_ind - start_frame_ind >= self.target_video_len
            frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int).tolist()
        # Load whole video frames
        else:
            frame_indice = np.linspace(0, total_frames-1, total_frames, dtype=int).tolist()

        vr = VideoReader(video_path, ctx=cpu(0))
        frames = vr.get_batch(frame_indice).asnumpy()
        video_clip = torch.from_numpy(frames).permute(0, 3, 1, 2)
        video_clip = self.transform(video_clip)

        # video_name = video_path.split('/')[-1]
        # youtube_id = video_name.split('_')[0]
        # class_name = self.annotation[self.annotation['youtube_id'] == youtube_id]['label'].values[0] if self.annotation is not None else 'unknown'
        # class_idx = 1

        return {'video': video_clip, 'video_name': 1, 'video_path': video_path}

    def __len__(self):
        return self.video_num
    
    def load_annotation(self):
        if self.annotation_path is None:
            return None
        with open(self.annotation_path, 'r') as f:
            annotations = pd.read_csv(f)

        classes = sorted(annotations['label'].unique().tolist())
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        annotations['label_idx'] = annotations['label'].map(class_to_idx)

        return annotations, class_to_idx
    
    def load_metadata(self, split: Literal["train", "val", "test"]):
        metadata_path = self.metadata_dir / f"{split}.pt"
        if not metadata_path.exists():
            print(f"Metadata file {metadata_path} not found. Building metadata...")
            self.build_metadata(split)

        metadata = torch.load(metadata_path)

        data = []
        for video_path, video_len in list(zip(metadata['video_paths'], metadata['video_lens'])):
            if video_len >= self.target_video_len:
                data.append((video_path, video_len))

        return data
    
    def load_video_list(self, split):
        corrupted_file = self.metadata_dir / f"{split}_corrupted.txt"
        corrupted = set()
        if corrupted_file.exists():
            corrupted = set(p.strip() for p in open(corrupted_file))
        all_videos = sorted(list((self.data_path).glob("**/*.mp4")), key=str)
        return [p for p in all_videos if str(p) not in corrupted]
    
    def build_metadata(self, split: Literal["train", "val", "test"], save_every: int = 100):
        if self.metadata_dir.exists() is False:
            self.metadata_dir.mkdir(parents=True, exist_ok=True)

        video_paths = self.load_video_list(split)
        dl = torch.utils.data.DataLoader(
            SafeVideoTimestampsDataset(video_paths),
            batch_size=64,
            num_workers=64,
            collate_fn=_collate_fn,
        )
        print(f"[INFO] Building metadata for {split} ({len(video_paths)} videos)...")

        metadata_path = self.metadata_dir / f"{split}.pt"
        corrupted_log = self.metadata_dir / f"{split}_corrupted.txt"

        video_lens = []
        valid_paths = []
        corrupted_videos = []

        for i, batch in enumerate(tqdm(dl, desc="Reading video timestamps")):
            try:
                for j, vlen in enumerate(batch):
                    if isinstance(vlen, int):
                        video_lens.append(vlen)
                        valid_paths.append(video_paths[i * 64 + j])
                    else:
                        corrupted_videos.append(video_paths[i * 64 + j])
            except Exception as e:
                # identify which videos failed in this batch
                batch_start = i * 16
                batch_end = min(batch_start + 16, len(video_paths))
                failed_batch = video_paths[batch_start:batch_end]
                corrupted_videos.extend(failed_batch)
                print(f"[WARN] Batch {i} failed ({len(failed_batch)} videos): {e}")
                continue

            # periodic checkpoint saving to avoid full reruns
            if (i + 1) % save_every == 0:
                torch.save({
                    "video_paths": valid_paths,
                    "video_lens": video_lens,
                }, metadata_path)
                print(f"[INFO] Partial metadata saved at batch {i+1}")

        # final save
        torch.save({
            "video_paths": valid_paths,
            "video_lens": video_lens,
        }, metadata_path)

        # log corrupted videos
        if corrupted_videos:
            with open(corrupted_log, "w") as f:
                for path in corrupted_videos:
                    f.write(str(path) + "\n")
            print(f"[INFO] Logged {len(corrupted_videos)} corrupted videos to {corrupted_log}")

        print(f"[DONE] Metadata built: {len(valid_paths)} valid videos, {len(corrupted_videos)} corrupted.")


if __name__ == '__main__':

    import argparse
    import torchvision
    import video_transforms
    import torch.utils.data as data

    from torchvision import transforms
    from torchvision.utils import save_image
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=6)
    parser.add_argument("--load_fron_ceph", type=bool, default=True)
    parser.add_argument("--data-path", type=str, default="/scratch/s224075134/temporal_diffusion/datasets/video/taichi/train")
    config = parser.parse_args()


    target_video_len = config.num_frames

    temporal_sample = video_transforms.TemporalRandomCrop(target_video_len * config.frame_interval)
    trans = transforms.Compose([
        video_transforms.ToTensorVideo(),
        video_transforms.RandomHorizontalFlipVideo(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    taichi_dataset = Taichi(config, transform=trans, temporal_sample=temporal_sample)
    taichi_dataloader = data.DataLoader(dataset=taichi_dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, video_data in enumerate(taichi_dataloader):
        video_data = video_data['video']
        print(video_data.shape)
        print(video_data.dtype)
        for i in range(target_video_len):
            save_image(video_data[0][i], os.path.join('./test_data', '%04d.png' % i), normalize=True, value_range=(-1, 1))

        video_ = ((video_data[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
        torchvision.io.write_video('./test_data/test.mp4', video_, fps=8)
        exit()
import os
import torch
import random
import torch.utils.data as data

import numpy as np
import io
import json
from PIL import Image
from decord import VideoReader
from decord import cpu, gpu


class Taichi(data.Dataset):
    def __init__(self, configs, transform, temporal_sample=None, train=True):

        self.configs = configs
        self.data_path = configs.data_path
        self.transform = transform

        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.frame_interval = self.configs.frame_interval
        # Preprocessed latents

        self.data_all = self.load_video_frames(self.data_path)
        self.video_num = len(self.data_all)

    def __getitem__(self, index):
        video_path, total_frames = self.data_all[index]
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
        return {'video': video_clip, 'video_name': 1, 'video_path': video_path}

    def __len__(self):
        return self.video_num
    
    def load_video_frames(self, dataroot):
        data_all = []
        # Find all mp4 files in dataroot (recursively)
        video_files = []
        for root, _, files in os.walk(dataroot):
            for file in files:
                if file.lower().endswith('.mp4'):
                    video_files.append(os.path.join(root, file))

        for video_path in video_files:
            vr = VideoReader(video_path, ctx=cpu(0))
            n_frames = len(vr)
            if n_frames > 0:
                data_all.append((video_path, n_frames))
            del vr
        return data_all    


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
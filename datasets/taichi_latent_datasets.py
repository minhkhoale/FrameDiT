import os
import torch
import random
import torch.utils.data as data
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models import AutoencoderKL
import numpy as np
import io
import json
from PIL import Image
from decord import VideoReader
from decord import cpu, gpu
from tqdm import tqdm


class TaichiLatent(data.Dataset):
    def __init__(self, configs, transform, temporal_sample=None, train=True):

        self.configs = configs
        self.latent_path = configs.latent_path
        self.transform = transform

        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.frame_interval = self.configs.frame_interval
        # Preprocessed latents
        self.data_all = self.load_latents(self.latent_path)
        self.video_num = len(self.data_all)

    def __getitem__(self, index):
        video_path, total_frames = self.data_all[index]
        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.target_video_len
        frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int).tolist()
        
        # load latent
        params = torch.load(video_path, weights_only=False)
        gaussian_dist = DiagonalGaussianDistribution(parameters=params)
        latent = gaussian_dist.sample()
        video_clip = latent[frame_indice]
        # load video
        video_clip = self.transform(video_clip)
        return {'video': video_clip, 'video_name': 1}

    def __len__(self):
        return self.video_num

    def load_latents(self, dataroot):
        data_all = []
        # Find all mp4 files in dataroot (recursively)
        latent_files = []
        for root, _, files in os.walk(dataroot):
            for file in files:
                if file.lower().endswith('.pt'):
                    latent_files.append(os.path.join(root, file))

        if len(latent_files) >= 1_000_000:
            for latent_path in tqdm(latent_files):
                n_frames = self.target_video_len
                data_all.append((latent_path, n_frames))
        else:
            for latent_path in tqdm(latent_files):
                latent = torch.load(latent_path, weights_only=False)
                n_frames = latent.shape[0]
                if n_frames >= self.temporal_sample.size:
                    data_all.append((latent_path, n_frames))
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
    parser.add_argument("--latent-path", type=str, default="/scratch/s224075134/temporal_diffusion/datasets/video/taichi_latent_16_kl_f8_autoencoder/train")
    config = parser.parse_args()


    target_video_len = config.num_frames

    temporal_sample = video_transforms.TemporalRandomCrop(target_video_len * config.frame_interval)
    trans = transforms.Compose([
        #video_transforms.ToTensorVideo(),
        #video_transforms.RandomHorizontalFlipVideo(p=1.0),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')

    taichi_dataset = TaichiLatent(config, transform=trans, temporal_sample=temporal_sample)
    taichi_dataloader = data.DataLoader(dataset=taichi_dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, video_data in enumerate(taichi_dataloader):
        if i < 20:
            continue
        latent = video_data['video']
        print(latent.shape)
        print(latent.dtype)
        latent = latent.flatten(0,1)
        video_data = vae.decode(latent).sample
        print(video_data.shape)
        video_data = video_data.reshape(1, target_video_len, 3, 128, 128)
        # for i in range(target_video_len):
        #     save_image(video_data[0][i], os.path.join('./test_data', '%04d.png' % i), normalize=True, value_range=(-1, 1))

        video_ = ((video_data[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
        torchvision.io.write_video('./test_data/test.mp4', video_, fps=8)
        exit()
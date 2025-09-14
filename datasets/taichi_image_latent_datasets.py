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

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class TaichiImagesLatent(data.Dataset):
    def __init__(self, configs, transform, temporal_sample=None, train=True):

        self.configs = configs
        self.latent_path = configs.latent_path
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.frame_interval = self.configs.frame_interval
        self.data_all, self.video_frame_all = self.load_video_frames_latent(self.latent_path)
        self.video_num = len(self.data_all)
        self.video_frame_num = len(self.video_frame_all)

        # sky video frames
        random.shuffle(self.video_frame_all)
        self.use_image_num = configs.use_image_num

    def __getitem__(self, index):

        video_index = index % self.video_num
        # vframes = self.data_all[video_index]
        video_path, total_frames = self.data_all[video_index]
        # total_frames = len(vframes)

        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.target_video_len
        frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
        
        params = torch.load(video_path, weights_only=False)
        gaussian_dist = DiagonalGaussianDistribution(parameters=params)
        latent = gaussian_dist.sample()
        video_clip = latent[frame_indice]

        # get video frames
        images = []
        for i in range(self.use_image_num):
            while True:
                try:
                    video_frame_path, frame_id = self.video_frame_all[index+i]
                    params = torch.load(video_frame_path, weights_only=False)[frame_id].unsqueeze(0)
                    gaussian_dist = DiagonalGaussianDistribution(parameters=params)
                    image = gaussian_dist.sample()
                    images.append(image)
                    break
                except Exception as e:
                    index = random.randint(0, self.video_frame_num - self.use_image_num)

        images =  torch.cat(images, dim=0)
        images = self.transform(images)
        assert len(images) == self.use_image_num

        video_cat = torch.cat([video_clip, images], dim=0)

        return {'video': video_cat, 'video_name': 1}

    def __len__(self):
        return self.video_frame_num
    
    def load_video_frames(self, dataroot):
        data_all = []
        frames_all = []
        frame_list = os.walk(dataroot)
        for _, meta in enumerate(frame_list):
            root = meta[0]
            try:
                frames = sorted(meta[2], key=lambda item: int(item.split('.')[0].split('_')[-1]))
            except:
                print(meta[0], meta[2])
            frames = [os.path.join(root, item) for item in frames if is_image_file(item)]
            # if len(frames) > max(0, self.sequence_length * self.sample_every_n_frames):
            if len(frames) != 0:
                data_all.append(frames)
                for frame in frames:
                    frames_all.append(frame)
        # self.video_num = len(data_all)
        return data_all, frames_all
    
    def load_video_frames_latent(self, dataroot):
        data_all = []
        frames_all = []
        # Find all mp4 files in dataroot (recursively)
        latent_files = []
        for root, _, files in os.walk(dataroot):
            for file in files:
                if file.lower().endswith('.pt'):
                    latent_files.append(os.path.join(root, file))

        for latent_path in latent_files:
            latent = torch.load(latent_path, weights_only=False)
            n_frames = latent.shape[0]
            if n_frames > 0:
                data_all.append((latent_path, n_frames))
                for i in range(n_frames):
                    frames_all.append((latent_path, i))

        return data_all, frames_all

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
    parser.add_argument("--use_image_num", type=int, default=8)
    parser.add_argument("--latent-path", type=str, default="/scratch/s224075134/temporal_diffusion/datasets/video/taichi_latent_32_kl_f8_autoencoder/train")
    config = parser.parse_args()


    target_video_len = config.num_frames

    temporal_sample = video_transforms.TemporalRandomCrop(target_video_len * config.frame_interval)
    trans = transforms.Compose([])
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')

    taichi_dataset = TaichiImagesLatent(config, transform=trans, temporal_sample=temporal_sample)
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
        video_data = video_data.reshape(1, 24, 3, 256, 256)
        # for i in range(target_video_len):
        #     save_image(video_data[0][i], os.path.join('./test_data', '%04d.png' % i), normalize=True, value_range=(-1, 1))

        video_ = ((video_data[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
        torchvision.io.write_video('./test_data/test.mp4', video_, fps=8)
        exit()
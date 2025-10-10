import sys
import os
sys.path.append("..")
sys.path.append("/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2")
current_directory = os.getcwd()
print(f"The current working directory is: {current_directory}")

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from datasets.taichi_datasets import Taichi
from datasets.taichi_latent_datasets import TaichiLatent
from datasets import video_transforms
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from vae import get_vae, encode_video
from einops import rearrange
import yaml
import cv2
import imageio


import cv2
latent_path = '/scratch/s224075134/temporal_diffusion/datasets/video/taichi_latent_16_kl_f8_autoencoder/train'
video_path = '/scratch/s224075134/temporal_diffusion/datasets/video/taichi/train'

config_path = '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/configs/taichi128/matlatte/taichi128_matlatte-8-256_train.yaml'
configs = OmegaConf.load(config_path)

transform_taichi = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.CenterCropResizeVideo(128),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])
transform_taichi1 = transforms.Compose([
                    video_transforms.ToTensorVideo(), # TCHW
                    transforms.Resize((128,128), interpolation=InterpolationMode.BILINEAR, antialias=False),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])

taichi_config = configs
taichi_config.load_latent = False
taichi_config.data_path = video_path
taichi_config.load_from_ceph = False
taichi_dataset = Taichi(taichi_config, transform=transform_taichi)
taichi_dataset1 = Taichi(taichi_config, transform=transform_taichi1)

taichi_loader = DataLoader(dataset=taichi_dataset, batch_size=1, shuffle=False, num_workers=0)
taichi_loader1 = DataLoader(dataset=taichi_dataset1, batch_size=1, shuffle=False, num_workers=0)


# load first video and latent
video_data = next(iter(taichi_loader))
video = video_data['video'] # B T C H W

video_data1 = next(iter(taichi_loader1))
video1 = video_data1['video'] # B T C H W

# encode video to latent
vae_config = OmegaConf.load("./configs/vae/autoencoder_kl.yaml")
vae = get_vae(vae_config).to('cuda:0')
vae.eval()
x = rearrange(video, 'b f c h w -> (b f) c h w').to('cuda:0')
x1 = rearrange(video1, 'b f c h w -> (b f) c h w').to('cuda:0')
x = torch.load('/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/test_data/preprocess_video.pt').to('cuda:0')
# torch.save(x.cpu(), './test_data/original_video.pt')
print('start encoding...')
with torch.no_grad():
    encode_latents_params = vae.encode(x).latent_dist.parameters
    encode_latents_params1 = vae.encode(x1).latent_dist.parameters

latent = torch.load('/scratch/s224075134/temporal_diffusion/datasets/video/taichi_latent_16_kl_f8_autoencoder_bilinear_flip/train/JTMn6S9cS_A#005235#005378.pt', weights_only=False).to('cuda:0')

print('encode_latents_params shape:', encode_latents_params.shape)
print('encode_latents_params1 shape:', encode_latents_params1.shape)

print('video', x[136, :, 8, 15])
print('encoded', encode_latents_params[136, :, 8, 15])
print('encoded1', encode_latents_params1[136, :, 8, 15])

print('mse:', torch.mean((encode_latents_params - encode_latents_params1)**2).item())
print('max abs diff:', torch.max(torch.abs(encode_latents_params - encode_latents_params1)).item())

# print('loaded latent shape:', latent.shape)
# print('encoded latent shape:', encode_latents_params.shape)


# print('encoded', encode_latents_params[0, :, 0, 0])
print('loaded', latent[136, :, 8, 15])

print('mse:', torch.mean((latent - encode_latents_params)**2).item())
print('where max abs diff:', torch.where(torch.abs(latent - encode_latents_params) == torch.max(torch.abs(latent - encode_latents_params))))

# sample = ((video[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
# imageio.mimwrite('/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/test_data/nearest_exact_128.mp4', sample, fps=8, quality=9)

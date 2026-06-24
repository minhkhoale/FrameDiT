import torch
from datasets import get_dataset
from vae import get_vae
from einops import rearrange
import os
import argparse
from torchvision import transforms
from tqdm import tqdm
from omegaconf import OmegaConf

def encode_video(vae, video, max_size=128):
    B = video.shape[0]
    video = rearrange(video, 'b f c h w -> (b f) c h w').contiguous()
    chunks = video.chunk((len(video) + max_size - 1) // max_size, dim=0)
    latents = []
    for chunk in chunks:
        with torch.no_grad():
            latent = vae.encode(chunk.contiguous()).latent_dist.parameters
            latents.append(latent)
    latents = torch.cat(latents, dim=0)
    latents = rearrange(latents, '(b f) c h w -> b f c h w', b=B).contiguous()
    return latents

def preprocess_latent(args):
    # mkdir save dir
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    dataset = get_dataset(args)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)

    vae = get_vae(OmegaConf.load(args.vae))
    vae.eval()
    vae.to('cuda:0')
    print("total number of videos: ", len(dataloader))

    for i, video_data in enumerate(tqdm(dataloader)):
        video = video_data['video'] # B T C H W
        flip_video = transforms.functional.hflip(video)

        video_path = video_data['video_path']
        B,T,C,H,W = video.shape
        with torch.no_grad():
            video = video.to('cuda:0')
            flip_video = flip_video.to('cuda:0')
            latents = encode_video(vae, video, max_size=64)
            flip_latents = encode_video(vae, flip_video, max_size=64)

        for b in range(B):
            video_name = os.path.basename(video_path[b])
            video_name = video_name.replace('.mp4', '')
            print(f'video {video_name} shape: {video[b].shape}, latent shape: {latents[b].shape}, flip latent shape: {flip_latents[b].shape}')
            save_file = os.path.join(args.save_path, f'{video_name}.pt')
            torch.save(latents[b].cpu(), save_file)

            flip_save_file = os.path.join(args.save_path, f'{video_name}_flip.pt')
            torch.save(flip_latents[b].cpu(), flip_save_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train.yaml")
    parser.add_argument("--save-path", type=str)
    args = parser.parse_args()

    configs = OmegaConf.load(args.config)
    configs.save_path = args.save_path
    configs.dataset += '_whole'
    assert configs.load_latent == False, "Should not load latent when preprocessing latent"
    assert configs.data_path is not None, "data_path should be specified in the config file"

    preprocess_latent(configs)


"""
python preprocess_latent.py --save-path /scratch/s224075134/temporal_diffusion/datasets/video/ucf101_latent_32_kl_f8_autoencoder_bilinear_flip/ --config configs/ucf101/fusedmatlatteimg/ucf101_fusedmatlatteimg-xl-pixel_train.yaml
python preprocess_latent.py --save-path /scratch/s224075134/temporal_diffusion/datasets/video/ucf101_latent_16_kl_f8_autoencoder_bilinear_flip/ --config configs/ucf101/fusedmatlatteimg/ucf101_fusedmatlatteimg-xl-pixel_train_128.yaml
"""
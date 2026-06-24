import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import get_dataset
from decord import VideoReader, cpu
from einops import rearrange
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm import tqdm

from utils import save_batch_videos


def get_video_fps(path, fallback_fps):
    if fallback_fps > 0:
        return fallback_fps

    try:
        vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps()
        del vr
        return fps if fps and fps > 0 else 8
    except Exception:
        return 8


def encode_decode_video(vae, video, max_size=64, use_mean=True):
    B = video.shape[0]
    flat_video = rearrange(video, "b f c h w -> (b f) c h w").contiguous()
    chunks = flat_video.chunk((len(flat_video) + max_size - 1) // max_size, dim=0)
    reconstructions = []

    for chunk in chunks:
        with torch.no_grad():
            latent_dist = vae.encode(chunk.contiguous()).latent_dist
            if use_mean and hasattr(latent_dist, "mode"):
                latents = latent_dist.mode()
            elif use_mean and hasattr(latent_dist, "mean"):
                latents = latent_dist.mean
            else:
                latents = latent_dist.sample()
            reconstruction = vae.decode(latents).sample
            reconstructions.append(reconstruction)

    reconstruction = torch.cat(reconstructions, dim=0)
    return rearrange(reconstruction, "(b f) c h w -> b f c h w", b=B).contiguous()


def save_video_pair(video, video_path, save_path, reconstruction_path, vae, args, output_name=None):
    video_name = output_name or os.path.splitext(os.path.basename(video_path))[0]
    fps = get_video_fps(video_path, args.fps)

    resized_file = os.path.join(save_path, f"{video_name}.mp4")
    # save_batch_videos(video.cpu(), [resized_file], fps=fps, quality=args.quality)

    if vae is not None:
        with torch.no_grad():
            reconstruction = encode_decode_video(
                vae,
                video.to(args.device),
                max_size=args.vae_batch_size,
                use_mean=args.reconstruction_use_mean,
            )
        reconstruction_file = os.path.join(reconstruction_path, f"{video_name}.mp4")
        save_batch_videos(reconstruction.cpu(), [reconstruction_file], fps=fps, quality=args.quality)


def preprocess_resized_reconstruction(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.reconstruction_path, exist_ok=True)

    dataset = get_dataset(args)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    vae = None
    if not args.no_reconstruction:
        from vae import get_vae

        assert args.vae is not None, "vae config should be specified when saving reconstructions"
        vae = get_vae(OmegaConf.load(args.vae))
        vae.eval()
        vae.to(args.device)

    print("total number of videos: ", len(dataloader))

    for i, video_data in enumerate(tqdm(dataloader)):
        if args.limit is not None and i >= args.limit:
            break

        video = video_data["video"]  # B T C H W, range [-1, 1]
        video_path = video_data["video_path"][0]

        save_video_pair(video, video_path, args.save_path, args.reconstruction_path, vae, args)

        if args.flip:
            flip_video = transforms.functional.hflip(video)
            save_video_pair(
                flip_video,
                video_path,
                args.save_path,
                args.reconstruction_path,
                vae,
                args,
                output_name=f"{os.path.splitext(os.path.basename(video_path))[0]}_flip",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ucf101/fusedmatlatteimg/ucf101_fusedmatlatteimg-xl-pixel_train_128.yaml")
    parser.add_argument("--data-path", type=str, default="/scratch/s224075134/temporal_diffusion/datasets/video/ucf101/video")
    parser.add_argument("--save-path", type=str, default="/scratch/s224075134/temporal_diffusion/datasets/video/ucf101/resized_128")
    parser.add_argument("--reconstruction-path", type=str, default="/scratch/s224075134/temporal_diffusion/datasets/video/ucf101/reconstruction_128")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--fps", type=float, default=0, help="Use source FPS when <= 0.")
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--vae-batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-flip", dest="flip", action="store_false")
    parser.add_argument("--no-reconstruction", action="store_true")
    parser.add_argument("--sample-reconstruction", dest="reconstruction_use_mean", action="store_false")
    parser.set_defaults(flip=True, reconstruction_use_mean=True)
    cli_args = parser.parse_args()

    configs = OmegaConf.load(cli_args.config)
    configs.data_path = cli_args.data_path
    configs.save_path = cli_args.save_path
    configs.reconstruction_path = cli_args.reconstruction_path
    configs.dataset = "ucf101_whole"
    configs.image_size = cli_args.image_size
    configs.load_latent = False
    configs.fps = cli_args.fps
    configs.quality = cli_args.quality
    configs.num_workers = cli_args.num_workers
    configs.vae_batch_size = cli_args.vae_batch_size
    configs.device = cli_args.device
    configs.limit = cli_args.limit
    configs.flip = cli_args.flip
    configs.no_reconstruction = cli_args.no_reconstruction
    configs.reconstruction_use_mean = cli_args.reconstruction_use_mean

    assert configs.data_path is not None, "data_path should be specified"

    preprocess_resized_reconstruction(configs)

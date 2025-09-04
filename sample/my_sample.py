# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained Latte.
"""
import os
import sys
try:
    import utils

    from diffusion import create_diffusion
    from utils import find_model, save_batch_videos
    from vae import get_vae, decode_video
except:
    sys.path.append(os.path.split(sys.path[0])[0])

    import utils
    from diffusion import create_diffusion
    from utils import find_model, save_batch_videos
    from vae import get_vae, decode_video

import torch
import argparse
import torchvision

from einops import rearrange
from models import get_models
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models.clip import TextEmbedder
import imageio
from omegaconf import OmegaConf

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(args):
    # Setup PyTorch:
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "Latte-XL/2", "Only Latte-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    using_cfg = args.cfg_scale > 1.0

    # Load model:
    latent_size = args.image_size // 8
    args.latent_size = latent_size
    model = get_models(args).to(device)

    if args.use_compile:
        model = torch.compile(model)

    # a pre-trained model or load a custom Latte checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = get_vae(OmegaConf.load(args.vae)).to(device)

    if args.use_fp16:
        print('WARNING: using half percision for inferencing!')
        vae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)
        # text_encoder.to(dtype=torch.float16)

    if not os.path.exists(args.save_video_path):
        os.makedirs(args.save_video_path)

    # Labels to condition the model with (feel free to change):

    local_batch_size = args.local_batch_size
    laten_shape = (local_batch_size, args.num_frames, args.in_channels, latent_size, latent_size) # b f c h w
    total_samples = args.total_samples
    num_batches = total_samples // local_batch_size

    for step in range(num_batches):
        # Create sampling noise:
        if args.use_fp16:
            z = torch.randn(laten_shape, dtype=torch.float16, device=device) # b c f h w
        else:
            z = torch.randn(laten_shape, device=device)

        # Setup classifier-free guidance:
        # z = torch.cat([z, z], 0)
        if using_cfg:
            z = torch.cat([z, z], 0)
            y = torch.randint(0, args.num_classes, (1,), device=device)
            y_null = torch.tensor([101] * 1, device=device)
            y = torch.cat([y, y_null], dim=0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, use_fp16=args.use_fp16)
            sample_fn = model.forward_with_cfg
        else:
            sample_fn = model.forward
            model_kwargs = dict(y=None, use_fp16=args.use_fp16)

        # Sample images:
        if args.sample_method == 'ddim':
            sample_loop = diffusion.ddim_sample_loop
        elif args.sample_method == 'ddpm':
            sample_loop = diffusion.p_sample_loop

        samples = sample_loop(sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device)
        
        if args.use_fp16:
            samples = samples.to(dtype=torch.float16)
        
        samples = decode_video(vae, samples/vae.scaler)
        video_paths = [os.path.join(args.save_video_path, f'video_{step*local_batch_size + i}' + '.mp4') for i in range(local_batch_size)]
        save_batch_videos(samples, video_paths, fps=args.fps, quality=args.video_quality)

        if step % 5 == 0:
            print(f'sample {step}/{num_batches} done!')
    
    print('save path {}'.format(args.save_video_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/ucf101/ucf101_sample.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--save_video_path", type=str, default="./sample_videos/")
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    omega_conf.ckpt = args.ckpt
    omega_conf.save_video_path = args.save_video_path
    main(omega_conf)

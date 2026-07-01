# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained Latte model using DDP.
Subsequently saves a .npz file that can be used to compute FVD and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import io
import os
import sys
import torch
sys.path.append(os.path.split(sys.path[0])[0])
import torch.distributed as dist
from utils import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import imageio
import json
from omegaconf import OmegaConf
from models import get_models
from einops import rearrange
from vae import get_vae, decode_video
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def setup_runtime(args):
    use_dist = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if use_dist:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    device = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    if args.seed is not None:
        torch.manual_seed(args.seed * world_size + rank)
    return use_dist, rank, world_size, device


def maybe_barrier(use_dist):
    if use_dist:
        dist.barrier()


def parse_action_sequence(action_sequence, num_frames):
    if not action_sequence:
        return None
    actions = [int(action.strip()) for action in action_sequence.split(",") if action.strip()]
    if len(actions) != num_frames:
        raise ValueError(
            f"--action-sequence must contain exactly {num_frames} comma-separated labels; got {len(actions)}"
        )
    return actions


def load_action_files(action_path):
    if not action_path:
        return []
    action_files = []
    for root, _, files in os.walk(action_path):
        for file_name in files:
            if file_name.lower().endswith(".npz"):
                action_files.append(os.path.join(root, file_name))
    return sorted(action_files)


def sample_action_sequences(action_files, batch_size, num_frames, rng):
    if not action_files:
        raise ValueError("No DMLab action .npz files found. Set --action-path or action_path in the config.")

    sequences = []
    metadata = []
    for _ in range(batch_size):
        action_file = action_files[int(rng.integers(0, len(action_files)))]
        actions = np.load(action_file)["actions"]
        if actions.ndim != 1:
            actions = actions.reshape(-1)

        if len(actions) >= num_frames:
            start = int(rng.integers(0, len(actions) - num_frames + 1))
            frame_indices = np.arange(start, start + num_frames, dtype=np.int64)
            sequence = actions[frame_indices]
        else:
            start = 0
            frame_indices = np.linspace(0, len(actions) - 1, num_frames, dtype=int)
            sequence = actions[frame_indices]
        sequences.append(sequence.astype(np.int64))
        metadata.append({
            "condition_type": "dmlab_action_sequence",
            "condition_source": "random_real_action_window",
            "source_action_path": action_file,
            "source_start": int(start),
            "source_frame_indices": frame_indices.astype(np.int64).tolist(),
            "action_sequence": sequence.astype(np.int64).tolist(),
        })

    return np.stack(sequences), metadata


def build_condition_labels(args, batch_size, label_idx, device, use_cfg, action_files=None, rng=None):
    if args.extras == 2:
        class_label = args.class_label if args.class_label >= 0 else label_idx % args.num_classes
        y = torch.full((batch_size,), class_label, dtype=torch.long, device=device)
        if use_cfg:
            y_null = torch.full((batch_size,), args.num_classes, dtype=torch.long, device=device)
            y = torch.cat([y, y_null], dim=0)
        metadata = [{"condition_type": "class", "class_label": int(class_label)} for _ in range(batch_size)]
        return y, [f"class_{class_label}"] * batch_size, metadata

    if args.extras == 3:
        action_sequence = parse_action_sequence(args.action_sequence, args.num_frames)
        if action_sequence is None:
            if args.action_label >= 0:
                y = torch.full((batch_size, args.num_frames), args.action_label, dtype=torch.long, device=device)
                label_names = [f"action_{args.action_label}"] * batch_size
                metadata = [{
                    "condition_type": "dmlab_action_sequence",
                    "condition_source": "fixed_action_label",
                    "action_label": int(args.action_label),
                    "action_sequence": [int(args.action_label)] * args.num_frames,
                } for _ in range(batch_size)]
            else:
                action_sequences, metadata = sample_action_sequences(action_files, batch_size, args.num_frames, rng)
                y = torch.tensor(action_sequences, dtype=torch.long, device=device)
                label_names = ["actions_random"] * batch_size
        else:
            y = torch.tensor(action_sequence, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
            label_names = ["actions_" + "-".join(str(action) for action in action_sequence)] * batch_size
            metadata = [{
                "condition_type": "dmlab_action_sequence",
                "condition_source": "explicit_action_sequence",
                "action_sequence": [int(action) for action in action_sequence],
            } for _ in range(batch_size)]

        if use_cfg:
            y_null = torch.full((batch_size, args.num_frames), args.num_classes, dtype=torch.long, device=device)
            y = torch.cat([y, y_null], dim=0)
        return y, label_names, metadata

    return None, ["uncond"] * batch_size, [None] * batch_size


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = True  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    use_dist, rank, world_size, device = setup_runtime(args)
    # print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "Latte-XL/2", "Only Latte-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    args.latent_size = latent_size
    model = get_models(args).to(device)

    # a pre-trained model or load a custom Latte checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    if args.use_compile:
        model = torch.compile(model)
    model.eval()  # important!
    
    print('model', model)
    diffusion = create_diffusion(
        timestep_respacing=str(args.num_sampling_steps),
        name=args.diffusion_name if hasattr(args, 'diffusion_name') else 'gaussian_diffusion',
        noise_schedule="linear",
        use_kl=False,
        sigma_small=args.sigma_small if 'sigma_small' in args else False,
        predict_xstart=args.predict_xstart if 'predict_xstart' in args else False,
        learn_sigma=args.learn_sigma if 'learn_sigma' in args else True,
        adaptive_frequency=args.get('adaptive_frequency', False),
        adaptive_frequency_gamma=args.get('adaptive_frequency_gamma', 0.5),
        adaptive_frequency_learnable_gamma=args.get('adaptive_frequency_learnable_gamma', False),
        adaptive_frequency_gamma_mode=args.get('adaptive_frequency_gamma_mode', 'scalar'),
        adaptive_frequency_power_path=args.get('adaptive_frequency_power_path', None),
        adaptive_frequency_power_exponent=args.get('adaptive_frequency_power_exponent', 2.0),
        adaptive_frequency_num_temporal_bands=args.get('adaptive_frequency_num_temporal_bands', None),
        adaptive_frequency_num_spatial_bands=args.get('adaptive_frequency_num_spatial_bands', None),
        equal_snr=args.get('equal_snr', False),
        equal_snr_power_path=args.get('equal_snr_power_path', None),
        equal_snr_power_scale=args.get('equal_snr_power_scale', 1.0),
        equal_snr_power_exponent=args.get('equal_snr_power_exponent', 2.0),
        equal_snr_use_channelwise=args.get('equal_snr_use_channelwise', True),
        equal_snr_calibrate_schedule=args.get('equal_snr_calibrate_schedule', False),
    )  # default: 1000 steps, linear noise schedule
    diffusion.initialize_adaptive_frequency_for_shape((1, args.num_frames, args.in_channels, latent_size, latent_size), device)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "adaptive_frequency" in checkpoint:
        diffusion.load_adaptive_frequency_state_dict(checkpoint["adaptive_frequency"])
    if isinstance(checkpoint, dict) and "equal_snr" in checkpoint:
        diffusion.load_equal_snr_state_dict(checkpoint["equal_snr"])
    for p in diffusion.adaptive_frequency_parameters():
        p.data = p.data.to(device)
    for p in diffusion.equal_snr_parameters():
        p.data = p.data.to(device)
    print('diffusion', diffusion)
    print('sampling method', args.sample_method)
    print('num_sampling_steps', args.num_sampling_steps)
    if diffusion.equal_snr.enabled and args.sample_method != "ddim":
        raise ValueError("EqualSNR checkpoints must be sampled with --sample-method ddim")

    vae = get_vae(OmegaConf.load(args.vae)).to(device)
    
    if args.use_fp16:
        print('WARNING: using half percision for inferencing!')
        vae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)
        # text_encoder.to(dtype=torch.float16)
    
    args.cfg_scale = getattr(args, 'cfg_scale', 1.0)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    print('args.cfg_scale', args.cfg_scale)
    using_cfg = args.cfg_scale > 1.0
    action_files = None
    action_rng = None
    if args.extras == 3 and not args.action_sequence and args.action_label < 0:
        action_files = load_action_files(args.get("action_path", ""))
        if rank == 0:
            print(f"Loaded {len(action_files)} DMLab action files from {args.get('action_path', '')}")
        action_rng = np.random.default_rng(args.seed * world_size + rank if args.seed is not None else None)

    # Create folder to save samples:
    # model_string_name = args.model.replace("/", "-")
    # ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    # folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
    #               f"cfg-{args.cfg_scale}-seed-{args.seed}"
    # sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    sample_folder_dir = args.save_video_path
    if args.seed:
        sample_folder_dir = args.save_video_path + '-seed-' + str(args.seed)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .mp4 samples at {sample_folder_dir}")
    maybe_barrier(use_dist)
    
    # check existing videos and skip them
    n_existing_video = len([name for name in os.listdir(sample_folder_dir) if os.path.isfile(os.path.join(sample_folder_dir, name)) and name.endswith('.mp4')])
    if n_existing_video > 0:
        print(f"Found {n_existing_video} existing videos in {sample_folder_dir}, skipping them.")
    
    args.num_fvd_samples = max(0, args.num_fvd_samples - n_existing_video)
    
    maybe_barrier(use_dist)

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * world_size
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fvd_samples / global_batch_size) * global_batch_size)

    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % world_size == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // world_size)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = n_existing_video
    label_idx = 0
    for _ in pbar:
        # Sample inputs:
        if args.use_fp16:
            z = torch.randn(n, args.num_frames, args.in_channels, latent_size, latent_size, dtype=torch.float16, device=device)
        else:
            z = torch.randn(n, args.num_frames, args.in_channels, latent_size, latent_size, device=device)
        if diffusion.equal_snr.enabled:
            z = diffusion.equal_snr.colored_noise(z)
        
        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y, label_names, condition_metadata = build_condition_labels(
                args, n, label_idx, device, using_cfg, action_files=action_files, rng=action_rng
            )
            label_idx = (label_idx + 1) % args.num_classes
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, use_fp16=args.use_fp16)
            sample_fn = model.forward_with_cfg
        else:
            y, label_names, condition_metadata = build_condition_labels(
                args, n, label_idx, device, using_cfg, action_files=action_files, rng=action_rng
            )
            if args.extras != 1:
                label_idx = (label_idx + 1) % args.num_classes
            model_kwargs = dict(y=y, use_fp16=args.use_fp16)
            sample_fn = model.forward

        # Sample images:
        if args.sample_method == 'ddim':
            samples = diffusion.ddim_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
        elif args.sample_method == 'ddpm':
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )

        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        if args.use_fp16:
            samples = samples.to(dtype=torch.float16)

        b, f, c, h, w = samples.shape
        samples = decode_video(vae, samples / vae.scaler)

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            sample = ((sample * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
            index = i * world_size + rank + total

            if y is not None:
                sample_save_path = f"{sample_folder_dir}/{index:04d}_{label_names[i]}.mp4"
            else:
                sample_save_path = f"{sample_folder_dir}/{index:04d}.mp4"

            print('sample_save_path', sample_save_path)
            imageio.mimwrite(sample_save_path, sample, fps=8, quality=9)
            if condition_metadata[i] is not None:
                metadata_save_path = os.path.splitext(sample_save_path)[0] + ".json"
                with open(metadata_save_path, "w") as f:
                    json.dump({
                        **condition_metadata[i],
                        "sample_path": sample_save_path,
                        "sample_index": int(index),
                        "seed": None if args.seed is None else int(args.seed),
                    }, f, indent=2)
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    maybe_barrier(use_dist)
    # if rank == 0:
    #     create_npz_from_sample_folder(sample_folder_dir, args.num_fvd_samples)
    #     print("Done.")
    # dist.barrier()
    if use_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/tuneavideo.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--save_video_path", type=str, default="./sample_videos/")

    parser.add_argument('--use-fp16', type=str2bool, nargs='?', const=True, help='Use half precision for inference', default=False)
    parser.add_argument('--seed', type=int, help='Random seed for sampling', default=0)
    parser.add_argument('--sample-method', type=str, help='Sampling method', default='ddpm')
    parser.add_argument('--num-sampling-steps', type=int, help='Number of sampling steps', default=50)
    parser.add_argument('--cfg-scale', type=float, help='Classifier-free guidance scale', default=None)
    parser.add_argument('--negative-name', type=str, help='Negative prompt name', default='')
    parser.add_argument('--batch-size', type=int, help='Batch size for sampling', default=4)
    parser.add_argument('--num-fvd-samples', type=int, help='Number of samples for FVD', default=2048)
    parser.add_argument('--fps', type=int, help='Frames per second for video', default=8)
    parser.add_argument('--video-quality', type=int, help='Quality for video encoding (1-10)', default=9)
    parser.add_argument('--wandb-run-id', type=str, help='W&B run ID for logging', default='')
    parser.add_argument('--class-label', type=int, default=-1, help='Fixed class label for extras=2; -1 cycles labels.')
    parser.add_argument('--action-label', type=int, default=-1, help='Fixed per-frame action label for extras=3; -1 cycles labels.')
    parser.add_argument('--action-sequence', type=str, default='', help='Comma-separated per-frame action labels for extras=3.')
    parser.add_argument('--action-path', type=str, default='', help='Directory containing DMLab .npz action files for extras=3.')

    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    omega_conf.ckpt = args.ckpt
    omega_conf.save_video_path = args.save_video_path

    omega_conf.seed = args.seed
    omega_conf.sample_method = args.sample_method
    omega_conf.num_sampling_steps = args.num_sampling_steps
    if args.cfg_scale is not None:
        omega_conf.cfg_scale = args.cfg_scale

    omega_conf.negative_name = args.negative_name

    omega_conf.use_fp16 = args.use_fp16
    omega_conf.fps = args.fps
    omega_conf.video_quality = args.video_quality

    omega_conf.per_proc_batch_size = args.batch_size
    omega_conf.num_fvd_samples = args.num_fvd_samples
    omega_conf.class_label = args.class_label
    omega_conf.action_label = args.action_label
    omega_conf.action_sequence = args.action_sequence
    if args.action_path:
        omega_conf.action_path = args.action_path

    main(omega_conf)

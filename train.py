"""
Simplified training script using HuggingFace Accelerate.

Replaces the project's original DDP setup with `Accelerator`, uses
`args.mixed_precision` to control mixed precision, and simplifies
training, logging, and checkpointing logic while remaining compatible
with the project's helper functions.
"""

import os
import time
import math
import logging
from glob import glob
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from models import get_models
from datasets import get_dataset
from vae import get_vae, encode_video, scale_latents
from diffusion import create_diffusion
from diffusers.optimization import get_scheduler
from utils import (
    update_ema,
    requires_grad,
    parse_args,
    get_experiment_dir,
)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Tracker:
    """Track and aggregate scalar loss entries from loss dictionaries.

    Usage:
    - Call `update(loss_dict)` after computing losses for a batch.
    - Call `average()` to get per-key averages since last reset.
    - Call `reset()` to clear internal accumulators.
    """
    def __init__(self) -> None:
        self.sums = OrderedDict()
        self.counts = OrderedDict()

    def update(self, loss_dict: Dict[str, Any]) -> None:
        for k, v in loss_dict.items():
            # Accept tensors or numeric scalars
            try:
                val = float(v.mean().detach().cpu().item()) if hasattr(v, 'mean') else float(v)
            except Exception:
                # Fallback: try direct float conversion
                val = float(v)
            if k not in self.sums:
                self.sums[k] = 0.0
                self.counts[k] = 0
            self.sums[k] += val
            self.counts[k] += 1

    def average(self) -> Dict[str, float]:
        return {k: (self.sums[k] / self.counts[k] if self.counts[k] > 0 else 0.0) for k in self.sums}

    def reset(self) -> None:
        self.sums.clear()
        self.counts.clear()


def log_model_stats(model, logger, name: str = "model") -> None:
    """Log basic model statistics: total params, trainable params, size, and top modules.

    Runs on the provided `logger`. Keeps output concise and human-readable.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # dtype counts and size in memory (bytes)
    dtype_counts = {}
    total_bytes = 0
    for p in model.parameters():
        dt = str(p.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + p.numel()
        total_bytes += p.element_size() * p.nelement()
    size_mb = total_bytes / (1024 ** 2)

    module_counts = {}
    for n, p in model.named_parameters():
        top = n.split('.')[0]
        module_counts[top] = module_counts.get(top, 0) + p.numel()

    logger.info(f"[{name}] total_params={total_params:,} trainable={trainable_params:,} param_size={size_mb:.2f}MB dtypes={dtype_counts}")
    # log top 8 modules by param count
    top_modules = sorted(module_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    for mod, cnt in top_modules:
        logger.info(f"[{name}] module {mod}: params={cnt:,}")

def setup_experiment(args):
    if args.debug:
        args.results_dir = os.path.join(args.results_dir, 'debug')

    experiment_index = len(glob(f"{args.results_dir}/{args.dataset}{args.image_size}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., Latte-XL/2 --> Latte-XL-2 (for naming folders)
    num_frame_string = 'F' + str(args.num_frames) + 'S' + str(args.frame_interval)
    os.makedirs(f"{args.results_dir}/{args.dataset}{args.image_size}", exist_ok=True)

    gaussian_name = args.diffusion_name if hasattr(args, "diffusion_name") else None
    experiment_name = f"{experiment_index:03d}-{model_string_name}-{num_frame_string}-{args.dataset}{args.image_size}"
    if hasattr(args, "use_lora") and args.use_lora:
        experiment_name += f"-lora-r{args.lora.r}-alpha{args.lora.lora_alpha}"

    if gaussian_name is not None:
        experiment_name += f"-{gaussian_name}"

    if args.num_frames == 16:
        experiment_dir = f"{args.results_dir}/{args.dataset}{args.image_size}/{experiment_name}"  # Create an experiment folder
    else:
        experiment_dir = f"{args.results_dir}/{args.dataset}{args.image_size}-{args.num_frames}/{experiment_name}"  # Create an experiment folder

    experiment_dir = get_experiment_dir(experiment_dir, args)
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))
    return experiment_name, experiment_dir, checkpoint_dir


def save_checkpoint(accelerator, checkpoint_dir, step, model, ema, optimizer, lr_scheduler, extras=None):
    if not accelerator.is_main_process:
        return
    os.makedirs(checkpoint_dir, exist_ok=True)
    #unwrapped = accelerator.unwrap_model(model)
    ckpt = {
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "train_steps": step,
    }
    if extras:
        ckpt.update(extras)
    path = os.path.join(checkpoint_dir, f"{step:07d}.pt")
    torch.save(ckpt, path)
    return path


def mem(tag="", logger=None):
    a = torch.cuda.memory_allocated()/1e9
    r = torch.cuda.memory_reserved()/1e9
    m = torch.cuda.max_memory_allocated()/1e9
    logger.info(f"[{tag}] alloc={a:.1f}G reserved={r:.1f}G max_alloc={m:.1f}G")


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    set_seed(args.global_seed)
    # Use Accelerator to handle device placement and mixed precision
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dynamo_backend="inductor" if args.use_compile else None,
        log_with="wandb"
    )

    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    logger = get_logger(__name__)
    logging.basicConfig(
        format='[%(asctime)s - %(levelname)s] %(message)s',
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if args.debug:
        logger.info("===============================\nRunning in debug mode.\n===============================")

    logger.info(f"Accelerator initialized. Device: {device}, Rank: {rank}, World Size: {world_size}, Mixed Precision: {accelerator.mixed_precision} Distributed type: {accelerator.state.distributed_type}")
    if args.use_compile:
        logger.info("TorchDynamo compilation enabled with 'inductor' backend.")

    if accelerator.is_main_process:
        experiment_name, experiment_dir, checkpoint_dir = setup_experiment(args)
        logger.info(f"Experiment Name: {experiment_name}")
        logger.info(f"Experiment Directory: {experiment_dir}")
        logger.info(f"Checkpoint Directory: {checkpoint_dir}")
        
        if args.project is not None:
            project = "debug" if args.debug and args.project is not None else args.project

            if hasattr(args, "run_id") and args.run_id is not None:
                init_kwargs = {"wandb": {"dir": experiment_dir, "id": args.run_id}}
            else:
                init_kwargs = {"wandb": {"dir": experiment_dir, "name": experiment_name}}

            accelerator.init_trackers(
                project_name=project,
                config=vars(args) if accelerator.is_main_process else None,
                init_kwargs=init_kwargs
            )
            logger.info(f"WandB tracking initialized for project: {project}, experiment: {experiment_name}")

    # VAE and model
    vae = get_vae(OmegaConf.load(args.vae)).to(device)
    if hasattr(args, 'offload_vae') and args.offload_vae:
        if not args.load_latent:
            logger.warning("Using raw videos, offloading VAE to CPU may slow down training.")
        else:
            logger.info("Offloading VAE to CPU.")
            vae.to('cpu')
    vae.requires_grad_(False)
    vae.eval()

    # TODO: handle it in case different VAE
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    sample_size = args.image_size // 8
    args.latent_size = sample_size
    model_args = deepcopy(args)
    if vae.is_video_vae:
        num_frames = args.num_frames
        model_args.num_frames = (num_frames // 4) + 1  # adjust num frames according to VAE frame factor
        logger.info(f"Using video VAE, adjusted num frames from {num_frames} to {model_args.num_frames}") 

    original_model = get_models(model_args)

    diffusion = create_diffusion(
        name=args.diffusion_name if 'diffusion_name' in args else 'gaussian_diffusion',
        timestep_respacing=None,
        noise_schedule="linear",
        use_kl=False,
        sigma_small=args.sigma_small if 'sigma_small' in args else False,
        predict_xstart=args.predict_xstart if 'predict_xstart' in args else False,
        learn_sigma=args.learn_sigma if 'learn_sigma' in args else True,
    )  # default: 1000 steps, linear noise schedule

    ema = deepcopy(original_model) # Create an EMA of the model for use after training
    if hasattr(args, 'offload_ema') and args.offload_ema:
        logger.info("Offloading EMA model to CPU.")
        ema = ema.to('cpu')
    update_ema(ema, original_model, decay=0)
    requires_grad(ema, False)
    ema = ema.to(device)
    ema.eval()     

    # Log model statistics (only on main process)
    if accelerator.is_main_process:
        log_model_stats(original_model, logger, name="original_model")
        log_model_stats(ema, logger, name="ema")

    model = original_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)

    # dataset and dataloader
    dataset = get_dataset(args)
    logger.info(f"Dataset {dataset}")
    if 'img' in args.dataset:
        assert args.use_image_num > 0, "Must specify use_image_num > 0 when training with image data."
        logger.info("Training with both video and image data, using up to {} images per video.".format(args.use_image_num))
        use_image_num = args.use_image_num
    else:
        use_image_num = 0

    loader = DataLoader(dataset, batch_size=int(args.local_batch_size), shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # scheduler
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # optionally load pretrained weights (simple behavior)
    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location='cpu')
        original_model.load_state_dict({k: v for k, v in ckpt['model'].items() if k in original_model.state_dict()})
        ema.load_state_dict({k: v for k, v in ckpt['ema'].items() if k in ema.state_dict()})
        optimizer.load_state_dict({k: v for k, v in ckpt['optimizer'].items() if k in optimizer.state_dict()})
        lr_scheduler.load_state_dict({k: v for k, v in ckpt['lr_scheduler'].items() if k in lr_scheduler.state_dict()})
        train_steps = ckpt['train_steps']
    else:
        train_steps = 0

    # prepare everything with accelerator
    model, optimizer, loader, lr_scheduler = accelerator.prepare(model, optimizer, loader, lr_scheduler)
    model.train(); vae.eval()
    
    prev_steps = train_steps
    tracker = Tracker()
    total_start_time = time.time()
    start_time = time.time()

    mem("Training", logger)
    logger.info(f'Start training with batch size {args.local_batch_size}, gradient accumulation steps {args.gradient_accumulation_steps}, effective batch size {args.local_batch_size * args.gradient_accumulation_steps}')

    num_update_steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    grad_norm = None
    for epoch in range(num_train_epochs):
        for step, video_data in enumerate(loader):
            # Load data
            x = video_data['video']
            if not args.load_latent:
                x = encode_video(vae, x)
            x = scale_latents(vae, x)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)

            # condition
            if args.dataset == "ucf101_img":
                image_name = video_data['image_name']
                image_names = []
                for caption in image_name:
                    single_caption = [int(item) for item in caption.split('=====')]
                    image_names.append(torch.as_tensor(single_caption))

            if args.extras == 1:
                model_kwargs = dict(y=None)
            elif args.extras == 2:
                model_kwargs = dict(y=video_data.get('video_name'))
                if args.dataset == "ucf101_img":
                    model_kwargs['y_image'] = image_names
            elif args.extras == 3:
                model_kwargs = dict(y=video_data['action'])
            else:
                raise NotImplementedError("extras mode not supported in simplified trainer")

            if use_image_num > 0:
                model_kwargs['use_image_num'] = use_image_num

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                    loss = loss_dict["loss"].mean()

                if torch.isnan(loss).any():
                    logger.warning("Loss is NaN, skipping this step.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if train_steps >= args.start_clip_iter:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                    else:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), float('inf'))  # just compute grad norm without clipping

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                tracker.update(loss_dict)

                if accelerator.sync_gradients:
                    train_steps += 1
                    update_ema(ema, original_model)

                    if train_steps % args.log_every == 0:
                        elapsed = time.time() - total_start_time
                        avg_losses = tracker.average()
                        # Prefer the scalar named 'loss' for concise logging message
                        end_time = time.time()
                        steps_per_sec = (train_steps-prev_steps) / (end_time - start_time)
                        eta = (args.max_train_steps - train_steps) / max(steps_per_sec, 1e-8)
                        prev_steps = train_steps
                        logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_losses.get('loss'):.4f}, Grad Norm: {grad_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Elapsed: {(elapsed):.2f}, ETA: {eta:.2f}")
                        # Log all tracked losses under the 'train/' prefix
                        log_dict = {f"train/{k}": v for k, v in avg_losses.items()}
                        log_dict.update({"train/grad_norm": grad_norm})
                        accelerator.log(log_dict, step=train_steps)
                        tracker.reset()
                        start_time = time.time()

                    #accelerator.wait_for_everyone()
                    if accelerator.is_main_process and train_steps % args.ckpt_every == 0 and train_steps > 0:
                        save_checkpoint(accelerator, checkpoint_dir, train_steps, accelerator.unwrap_model(model, keep_fp32_wrapper=False, keep_torch_compile=False), ema, optimizer, lr_scheduler)
                        logger.info(f"Saved checkpoint at step {train_steps} to {checkpoint_dir}")

                if train_steps >= args.max_train_steps:
                    break
            
            if train_steps >= args.max_train_steps:
                break

    if accelerator.is_main_process:
        logger.info("Training finished")
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    if not hasattr(args, 'mixed_precision'):
        args.mixed_precision = 'no'
    main(args)

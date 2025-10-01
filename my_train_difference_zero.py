# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Latte using PyTorch DDP.
"""


import torch
# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import io
import os
import math
import argparse

import torch.distributed as dist
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from models import get_models
from datasets import get_dataset
from models.clip import TextEmbedder
from vae import get_vae, encode_video
from diffusion import create_diffusion
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import (clip_grad_norm_, create_logger, update_ema, 
                   requires_grad, cleanup, setup_distributed,
                   get_experiment_dir, text_preprocessing)
from models.diff_utils import *
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
import wandb
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # Setup DDP:
    setup_distributed()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)


    # Setup an experiment folder:
    if rank == 0:
        if args.debug:
            args.results_dir = os.path.join(args.results_dir, 'debug')

        os.makedirs(f"{args.results_dir}/{args.dataset}{args.image_size}", exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/{args.dataset}{args.image_size}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., Latte-XL/2 --> Latte-XL-2 (for naming folders)
        num_frame_string = 'F' + str(args.num_frames) + 'S' + str(args.frame_interval)

        experiment_name = f"{experiment_index:03d}-{model_string_name}-{num_frame_string}-{args.dataset}{args.image_size}"
        experiment_dir = f"{args.results_dir}/{args.dataset}{args.image_size}/{experiment_name}"  # Create an experiment folder
        experiment_dir = get_experiment_dir(experiment_dir, args)
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(experiment_dir, exist_ok=True)

        logger = create_logger(experiment_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))
        logger.info(f"Experiment directory created at {experiment_dir}")

        project = "debug" if args.debug and args.project is not None else args.project
        wandb.init(project=project, name=experiment_name, tags=['video_generation', model_string_name, f"{args.dataset}{args.image_size}", "training"]) if args.project else None
    else:
        logger = create_logger(None)
        # tb_writer = None
    
    logger.info('starting main')
    logger.info(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")    
    if args.debug:
        logger.info("===============================\nRunning in debug mode.\n===============================")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    sample_size = args.image_size // 8
    args.latent_size = sample_size
    model = get_models(args)
    
    diffusion = create_diffusion(
        timestep_respacing=None,
        name=args.diffusion_name if hasattr(args, 'diffusion_name') else 'difference_gaussian_diffusion_v0',
        noise_schedule="linear",
        use_kl=False,
        sigma_small=args.sigma_small if 'sigma_small' in args else False,
        predict_xstart=args.predict_xstart if 'predict_xstart' in args else False,
        learn_sigma=args.learn_sigma if 'learn_sigma' in args else True,
    )  # default: 1000 steps, linear noise schedule
    logger.info(f'diffusion {diffusion}')

    vae = get_vae(OmegaConf.load(args.vae)).to(device)

    # # use pretrained model?
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            logger.info('Using ema ckpt!')
            checkpoint = checkpoint["ema"]

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                logger.info('Ignoring: {}'.format(k))
        logger.info('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('Successfully load model at {}!'.format(args.pretrained))

    if args.use_compile:
        model = torch.compile(model)

    # Note that parameter initialization is done within the Latte constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
  
    # set distributed training
    model = DDP(model.to(device), device_ids=[local_rank])

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)

    # Setup data:
    dataset = get_dataset(args)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.local_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} videos ({args.data_path})")

    # Scheduler
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0

    running_metrics = {
        "loss": 0,
        "xs_loss": 0,
        "diff_loss": 0,
        "mse": 0,
        "xs_mse": 0,
        "diff_mse": 0,
        "vb": 0,
        "xs_vb": 0,
        "diff_vb": 0,
    }

    first_epoch = 0
    start_time = time()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(loader))
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # TODO, need to checkout
        # Get the most recent checkpoint
        dirs = os.listdir(os.path.join(experiment_dir, 'checkpoints'))
        dirs = [d for d in dirs if d.endswith("pt")]
        dirs = sorted(dirs, key=lambda x: int(x.split(".")[0]))
        path = dirs[-1]
        logger.info(f"Resuming from checkpoint {path}")
        model.load_state(os.path.join(dirs, path))
        train_steps = int(path.split(".")[0])

        first_epoch = train_steps // num_update_steps_per_epoch
        resume_step = train_steps % num_update_steps_per_epoch

    if args.pretrained:
        train_steps = int(args.pretrained.split("/")[-1].split('.')[0])

    for epoch in range(first_epoch, num_train_epochs):
        sampler.set_epoch(epoch)
        for step, video_data in enumerate(loader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            x = video_data['video'].to(device, non_blocking=True)
            video_name = video_data['video_name']
            
            if not args.load_latent:
                x = encode_video(vae, x)  # (B,F,C,H,W)
            x = x.mul_(vae.scaler)

            # Get difference
            dx = torch.zeros_like(x, dtype=x.dtype)
            #TODO: add diff norm
            # x = interleave_frames(x, diff)

            # TODO: add condition for action dataset
            if args.extras == 78: # text-to-video
                raise 'T2V training are Not supported at this moment!'
            elif args.extras == 2:
                model_kwargs = dict(y=video_name)
            else:
                model_kwargs = dict(y=None)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, dx, t, model_kwargs)
            loss = loss_dict["loss"].mean() / args.gradient_accumulation_steps
            loss.backward()

            if train_steps < args.start_clip_iter: # if train_steps >= start_clip_iter, will clip gradient
                gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False)
            else:
                gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=True)

            lr_scheduler.step()
            if train_steps % args.gradient_accumulation_steps == 0 and train_steps > 0:
                opt.step()
                opt.zero_grad()
                update_ema(ema, model.module)

            # for logging
            running_metrics["vb"] += loss_dict["vb"].mean().item() / args.gradient_accumulation_steps if "vb" in loss_dict else 0.0
            running_metrics["mse"] += loss_dict["mse"].mean().item() / args.gradient_accumulation_steps if "mse" in loss_dict else 0.0
            running_metrics["loss"] += loss.item()

            running_metrics["xs_vb"] += loss_dict['x_vb'].mean().item() / args.gradient_accumulation_steps if 'x_vb' in loss_dict else 0.0
            running_metrics["xs_mse"] += loss_dict['x_mse'].mean().item() / args.gradient_accumulation_steps if 'x_mse' in loss_dict else 0.0
            running_metrics["xs_loss"] += loss_dict['x_loss'].mean().item() / args.gradient_accumulation_steps if 'x_loss' in loss_dict else 0.0

            running_metrics["diff_vb"] += loss_dict['dx_vb'].mean().item() / args.gradient_accumulation_steps if 'dx_vb' in loss_dict else 0.0
            running_metrics["diff_mse"] += loss_dict['dx_mse'].mean().item() / args.gradient_accumulation_steps if 'dx_mse' in loss_dict else 0.0
            running_metrics["diff_loss"] += loss_dict['dx_loss'].mean().item() / args.gradient_accumulation_steps if 'dx_loss' in loss_dict else 0.0

            # Logging
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Reduce running_metrics over all processes
                reduced_metrics = {}
                for k, v in running_metrics.items():
                    t = torch.tensor(v, device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    reduced_metrics[k] = t.item() / dist.get_world_size() / log_steps

                logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Loss: {reduced_metrics['loss']:.4f}, xs_loss: {reduced_metrics['xs_loss']:.4f}, diff_loss: {reduced_metrics['diff_loss']:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                if wandb.run is not None:
                    logging_dict = {
                        "train/loss": reduced_metrics['loss'],
                        "train/grad_norm": gradient_norm
                    }
                    for k, v in reduced_metrics.items():
                        if v != 0:
                            logging_dict[f"train/{k}"] = v

                    wandb.log(logging_dict, step=train_steps)

                # Reset monitoring variables:
                for k in running_metrics:
                    running_metrics[k] = 0.0

                log_steps = 0
                start_time = time()

            # Save Latte checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "train_steps": train_steps
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train Latte with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train.yaml")
    parser.add_argument("--debug", action='store_true', help="If true, run in debug mode.")
    args = parser.parse_args()

    configs = OmegaConf.load(args.config)
    configs.debug = args.debug

    main(configs)

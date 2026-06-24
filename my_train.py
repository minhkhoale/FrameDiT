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
from vae import get_vae, encode_video, scale_latents
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
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
import wandb
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

#################################################################################
#                                  Training Loop                                #
#################################################################################
BIN_EDGES = [0, 200, 400, 600, 800, 1000]
BIN_LABELS = ["0-199", "200-399", "400-599", "600-799", "800-999"]
NUM_BINS = 5


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    print('starting main')
    # Setup DDP:
    setup_distributed()
    # dist.init_process_group("nccl")
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    # device = rank % torch.cuda.device_count()
    # local_rank = rank

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")
    if args.debug:
        print("===============================\nRunning in debug mode.\n===============================")

    # Setup an experiment folder:
    if rank == 0:
        if args.debug:
            args.results_dir = os.path.join(args.results_dir, 'debug')

        os.makedirs(f"{args.results_dir}/{args.dataset}{args.image_size}", exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/{args.dataset}{args.image_size}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., Latte-XL/2 --> Latte-XL-2 (for naming folders)
        num_frame_string = 'F' + str(args.num_frames) + 'S' + str(args.frame_interval)

        gaussian_name = args.diffusion_name if 'diffusion_name' in args else None
        experiment_name = f"{experiment_index:03d}-{model_string_name}-{num_frame_string}-{args.dataset}{args.image_size}"

        if gaussian_name is not None:
            experiment_name += f"-{gaussian_name}"

        if args.num_frames == 16:
            experiment_dir = f"{args.results_dir}/{args.dataset}{args.image_size}/{experiment_name}"  # Create an experiment folder
        else:
            experiment_dir = f"{args.results_dir}/{args.dataset}{args.image_size}-{args.num_frames}/{experiment_name}"  # Create an experiment folder
        experiment_dir = get_experiment_dir(experiment_dir, args)

        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)

        logger = create_logger(experiment_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))
        logger.info(f"Experiment directory created at {experiment_dir}")

        project = "debug" if args.debug and args.project is not None else args.project

        if hasattr(args, "run_id") and args.run_id is not None:
            wandb.init(project=project, id=args.run_id, resume="must") if args.project else None
        else:
            wandb.init(project=project, name=experiment_name, tags=['video_generation', model_string_name, f"{args.dataset}{args.image_size}", "training"]) if args.project else None
    else:
        logger = create_logger(None)
        # tb_writer = None

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    sample_size = args.image_size // 8
    args.latent_size = sample_size

    vae = get_vae(OmegaConf.load(args.vae)).to(device)

    # clone args
    model_args = deepcopy(args)
    if vae.is_video_vae:
        num_frames = args.num_frames
        model_args.num_frames = (num_frames // 4) + 1  # adjust num frames according to VAE frame factor
        logger.info(f"Using video VAE, adjusted num frames from {num_frames} to {model_args.num_frames}") 
    model = get_models(model_args)
    # print('Model', model)
    
    diffusion = create_diffusion(
        name=args.diffusion_name if 'diffusion_name' in args else 'gaussian_diffusion',
        timestep_respacing=None,
        noise_schedule="linear",
        use_kl=False,
        sigma_small=args.sigma_small if 'sigma_small' in args else False,
        predict_xstart=args.predict_xstart if 'predict_xstart' in args else False,
        learn_sigma=args.learn_sigma if 'learn_sigma' in args else True,
    )  # default: 1000 steps, linear noise schedule
    print('diffusion', diffusion)

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

    # Note that parameter initialization is done within the Latte constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
  
    # set distributed training
    model = DDP(model.to(device), device_ids=[local_rank])
    if args.use_compile:
        model = torch.compile(model)

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    vae.eval()

    # Setup data:
    dataset = get_dataset(args)
    logger.info(f"Dataset {dataset}")
    if args.load_latent:
        logger.info(f"Loading latent from {args.data_path}")
    else:
        logger.info(f"Loading video from {args.data_path}")

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
    logger.info(f"Num frames per video: {args.num_frames}, frame interval: {args.frame_interval}")
    logger.info(f"Batch size per GPU: {args.local_batch_size}, global batch size: {args.local_batch_size * dist.get_world_size()}")

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

    # if args.mixed_precision_16bit:
    #     scaler = torch.amp.GradScaler()

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_loss_mse = 0
    running_loss_vb = 0
    running_bins = {
        "x_loss_sum":  [0.0] * NUM_BINS,
        "count":       [0.0] * NUM_BINS,
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

    total_start_time = time()
    for epoch in range(first_epoch, num_train_epochs):
        sampler.set_epoch(epoch)

        end = time()
        for step, video_data in enumerate(loader):
            data_time = time() - end
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            x = video_data['video'].to(device, non_blocking=True)
            video_name = video_data['video_name']
            
            if not args.load_latent:
                x = encode_video(vae, x)  # (B,F,C,H,W)

            x = scale_latents(vae, x)

            if args.extras == 78: # text-to-video
                raise 'T2V training are Not supported at this moment!'
            elif args.extras == 2:
                model_kwargs = dict(y=video_name)
            else:
                model_kwargs = dict(y=None)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            if args.mixed_precision:
                with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
                    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                    loss = loss_dict["loss"].mean() / args.gradient_accumulation_steps
                loss.backward()
                #scaler.scale(loss).backward()
            else:
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean() / args.gradient_accumulation_steps
                loss.backward()

            # for logging
            mse = loss_dict["mse"].mean().item() / args.gradient_accumulation_steps if "mse" in loss_dict else 0.0
            vb = loss_dict["vb"].mean().item() / args.gradient_accumulation_steps if "vb" in loss_dict else 0.0
            lr_scheduler.step()

            if train_steps % args.gradient_accumulation_steps == 0 and train_steps > 0:
                #scaler.unscale_(opt)
                if train_steps < args.start_clip_iter: # if train_steps >= start_clip_iter, will clip gradient
                    gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False)
                else:
                    gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=True)
            
                # if args.mixed_precision_16bit:
                #     scaler.step(opt)
                #     scaler.update()
                # else:
                opt.step()

                opt.zero_grad()
                update_ema(ema, model.module)

            # Logging
            running_loss += loss.item()
            running_loss_mse += mse
            running_loss_vb += vb

            # track bins
            with torch.no_grad():
                x_loss_ps  = loss_dict.get('loss',  None)
                if x_loss_ps is not None:
                    for i in range(NUM_BINS):
                        lo, hi = BIN_EDGES[i], BIN_EDGES[i+1]
                        mask = (t >= lo) & (t < hi)
                        if mask.any():
                            # Sum the masked losses (as floats), and count
                            running_bins["x_loss_sum"][i]  += x_loss_ps[mask].sum().item() / args.gradient_accumulation_steps
                            running_bins["count"][i]       += mask.sum().item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, ETA: {(time()-total_start_time):.2f}")
                x_sums  = torch.tensor(running_bins["x_loss_sum"],  device=device, dtype=torch.float64)
                counts  = torch.tensor(running_bins["count"],       device=device, dtype=torch.float64)
                dist.all_reduce(x_sums,  op=dist.ReduceOp.SUM)
                dist.all_reduce(counts,  op=dist.ReduceOp.SUM)

                x_means  = (x_sums  / (counts + 1e-12)).tolist()


                if wandb.run is not None:
                    logging_dict = {
                        "train/loss": avg_loss,
                        "train/xs_loss": avg_loss,
                        "train/grad_norm": gradient_norm
                    }
                    if 'mse' in loss_dict:
                        logging_dict["train/mse"] = running_loss_mse / log_steps
                        logging_dict["train/xs_mse"] = running_loss_mse / log_steps
                    if 'vb' in loss_dict:
                        logging_dict["train/loss_vb"] = running_loss_vb / log_steps

                    for i, lbl in enumerate(BIN_LABELS):
                        if counts[i] > 0:
                            logging_dict[f"train/bin_xs_loss/{lbl}"]  = x_means[i]

                    wandb.log(logging_dict, step=train_steps)

                # Reset monitoring variables:
                running_loss = 0
                running_loss_mse = 0
                running_loss_vb = 0
                running_bins = {
                    "x_loss_sum":  [0.0] * NUM_BINS,
                    "count":       [0.0] * NUM_BINS,
                }
                log_steps = 0
                start_time = time()

            # Save Latte checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
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
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()

    configs = OmegaConf.load(args.config)
    configs.debug = args.debug
    if args.num_workers is not None:
        configs.num_workers = args.num_workers

    main(configs)

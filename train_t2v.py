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
from models.utils import load_pretrained_latte_into_framedith, freeze_model_for_matrix_training
from datasets import get_dataset
from tokenizer import get_tokenizer, _text_preprocessing
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
                   get_experiment_dir, get_torch_dtype)
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
import wandb
# import torch._inductor.config as cfg
# cfg.triton.cudagraphs = False
import torch._inductor.config as cfg
cfg.triton.cudagraphs = False
# cfg.max_autotune = False   # only if your torch build supports it
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

# torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(True)
# torch.backends.cuda.enable_math_sdp(True)  # keep fallback


def mem(tag="", logger=None):
    a = torch.cuda.memory_allocated()/1e9
    r = torch.cuda.memory_reserved()/1e9
    m = torch.cuda.max_memory_allocated()/1e9
    logger.info(f"[{tag}] alloc={a:.1f}G reserved={r:.1f}G max_alloc={m:.1f}G")


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
    if hasattr(args, 'offload_vae') and args.offload_vae:
        if not args.load_latent:
            logger.warning("Loading latent videos, offloading VAE to CPU may slow down training.")
        else:
            logger.info("Offloading VAE to CPU.")
            vae.to('cpu')

    tokenizer, text_encoder = get_tokenizer(OmegaConf.load(args.tokenizer))
    text_encoder = text_encoder.to(device)

    # clone args
    model_args = deepcopy(args)
    # if vae.is_video_vae:
    #     num_frames = args.num_frames
    #     model_args.num_frames = (num_frames // 4) + 1  # adjust num frames according to VAE frame factor
    #     logger.info(f"Using video VAE, adjusted num frames from {num_frames} to {model_args.num_frames}") 
    base_model = get_models(model_args)
    base_model.channel_first = True
    if args.gradient_checkpointing:
        n_blocks = args.n_checkpointing_blocks if 'n_checkpointing_blocks' in args else 999
        logger.info('enable gradient checkpointing for first {} blocks'.format(n_blocks))
        base_model.enable_gradient_checkpointing(n_blocks)

    ema_update_every = args.ema_update_every if hasattr(args, "ema_update_every") else 1

    logger.info('load latte model')
    load_pretrained_latte_into_framedith(base_model, args.pretrained_latte, logger, device)
    # freeze all layers except framedit_h_t2v layers
    # for name, param in model.named_parameters():
    # print('Model', model)
    # LORA
    if hasattr(args, 'use_lora') and args.use_lora:
        from peft import get_peft_model, LoraConfig, TaskType
        logger.info('use lora finetune')
        peft_config = LoraConfig(
            r=args.lora.r,
            lora_alpha=args.lora.lora_alpha,
            target_modules=args.lora.target_modules,
            lora_dropout=args.lora.lora_dropout,
        )
        base_model = get_peft_model(base_model, peft_config)
        trainable_params, all_param = base_model.get_nb_trainable_parameters()
        logger.info(f'Number of trainable parameters (LoRA): {trainable_params:,} ({trainable_params/all_param:.2%})')
        # for name, module in model.named_modules():
        #     if name.endswith(("to_q","to_k","to_v","to_out.0")):
        #         print(name, type(module))

    freeze_model_for_matrix_training(base_model, logger)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         logger.info(f'Trainable parameter: {name}, shape: {param.shape}')
    # 3. Validation Stats
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    
    logger.info("-" * 50)
    # print('Trainable Parameters:', trainable_param_names)
    logger.info(f"Total Parameters:     {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    logger.info("-" * 50)

    # count params of attention modules
    logger.info('-------------------------')
    attention_params = 0
    for name, param in base_model.named_parameters():
        # print(name)
        if 'attention' in name  and (not 'matrix_attention' in name):
            attention_params += param.numel()
            # print(name)
    logger.info(f"Local Factorized Attention Parameters: {attention_params:,} ({attention_params/total_params:.2%})")
    #   exit(0)
    
    diffusion = create_diffusion(
        name=args.diffusion_name if 'diffusion_name' in args else 'gaussian_diffusion',
        timestep_respacing=None,
        noise_schedule="linear",
        use_kl=False,
        sigma_small=args.sigma_small if 'sigma_small' in args else False,
        predict_xstart=args.predict_xstart if 'predict_xstart' in args else False,
        learn_sigma=args.learn_sigma if 'learn_sigma' in args else True,
        adaptive_frequency=args.get('adaptive_frequency', False),
        adaptive_frequency_gamma=args.get('adaptive_frequency_gamma', 0.5),
        adaptive_frequency_learnable_gamma=args.get('adaptive_frequency_learnable_gamma', False),
        adaptive_frequency_power_path=args.get('adaptive_frequency_power_path', None),
        adaptive_frequency_power_exponent=args.get('adaptive_frequency_power_exponent', 2.0),
        adaptive_frequency_num_temporal_bands=args.get('adaptive_frequency_num_temporal_bands', None),
        adaptive_frequency_num_spatial_bands=args.get('adaptive_frequency_num_spatial_bands', None),
    )  # default: 1000 steps, linear noise schedule
    for p in diffusion.adaptive_frequency_parameters():
        p.data = p.data.to(device)
    logger.info(f'diffusion: {diffusion}')

    # # use pretrained model?
    logger.info('load pretrained model')
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        if isinstance(checkpoint, dict) and "adaptive_frequency" in checkpoint:
            diffusion.load_adaptive_frequency_state_dict(checkpoint["adaptive_frequency"])
        if "ema" in checkpoint:  # supports checkpoints from train.py
            logger.info('Using ema ckpt!')
            checkpoint = checkpoint["ema"]

        model_dict = base_model.state_dict()
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
        base_model.load_state_dict(model_dict)
        logger.info('Successfully load model at {}!'.format(args.pretrained))

    # Note that parameter initialization is done within the Latte constructor
    ema = deepcopy(base_model)  # Create an EMA of the model for use after training
    if hasattr(args, 'offload_ema') and args.offload_ema:
        logger.info("Offloading EMA model to CPU.")
        ema = ema.to('cpu')

    requires_grad(ema, False)
    if hasattr(args, 'gate_initialziation'):
        logger.info(f"Reset content gate with {args.gate_initialziation} initialization")
        base_model.reset_content_gate(args.gate_initialziation)

    if args.enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available
        if is_xformers_available():
            logger.info("Enabling xformers memory efficient attention.")
            base_model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        
    # set distributed training
    if args.use_compile:
        logger.info("Using torch.compile for model compilation.")
        model = torch.compile(base_model)
    else:
        model = base_model
    model = DDP(model.to(device), device_ids=[local_rank])

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(diffusion.adaptive_frequency_parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay if 'weight_decay' in args else 0,
    )
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    vae.eval()

    text_encoder.requires_grad_(False)
    text_encoder.eval()

    # Setup data:
    dataset = get_dataset(args)
    logger.info(f"Dataset {dataset}")
    if args.load_latent:
        logger.info(f"Loading latent from {args.latent_path}")
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
        #pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        #pin_memory_device="cuda"
    )
    logger.info(f"Dataset contains {len(dataset):,} videos ({args.latent_path})")
    logger.info(f"Num frames per video: {args.num_frames}, frame interval: {args.frame_interval}")
    logger.info(f"Batch size per GPU: {args.local_batch_size}, global batch size: {args.local_batch_size * dist.get_world_size()}")
    logger.info(f"Learning rate: {args.learning_rate}, gradient accumulation steps: {args.gradient_accumulation_steps}")

    # Scheduler
    # lr_scheduler = get_scheduler(
    #     name="constant",
    #     optimizer=opt,
    #     num_warmup_steps=args.lr_warmup_steps,
    #     num_training_steps=args.max_train_steps,
    # )

    # Prepare models for training:
    update_ema(ema, base_model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    use_scaler = args.mixed_precision == 'float16'
    if use_scaler:
        scaler = torch.amp.GradScaler()

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
        ckpt_dirs = os.listdir(os.path.join(experiment_dir, 'checkpoints'))
        ckpt_dirs = [d for d in ckpt_dirs if d.endswith("pt")]
        ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split(".")[0]))
        path = ckpt_dirs[-1]
        logger.info(f"Resuming from checkpoint {path}")
        resume_checkpoint = torch.load(os.path.join(experiment_dir, 'checkpoints', path), map_location=device)
        if isinstance(resume_checkpoint, dict) and "model" in resume_checkpoint:
            base_model.load_state_dict(resume_checkpoint["model"])
            if "adaptive_frequency" in resume_checkpoint:
                diffusion.load_adaptive_frequency_state_dict(resume_checkpoint["adaptive_frequency"])
        else:
            base_model.load_state_dict(resume_checkpoint)
        train_steps = int(path.split(".")[0])

        first_epoch = train_steps // num_update_steps_per_epoch
        resume_step = train_steps % num_update_steps_per_epoch

    if args.pretrained:
        train_steps = int(args.pretrained.split("/")[-1].split('.')[0])

    # for n, p in model.named_parameters():
    #     if "content_gate" in n:
    #         logger.info(f"{n} - {p.mean()} - {p.std()}, {p.requires_grad}")

    total_start_time = time()
    for epoch in range(first_epoch, num_train_epochs):
        sampler.set_epoch(epoch)

        # end = time()
        for step, video_data in enumerate(loader):
            # data_time = time() - end
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue
            
            # encode video to latents
            if not args.load_latent:
                x = video_data['video'].to(device, non_blocking=True)
                x = encode_video(vae, x)  # (B,F,C,H,W)
                prompt = video_data['prompt']
                prompt = _text_preprocessing(prompt, args.text_cleaning)
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=120,
                    truncation=True,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                attention_mask = text_inputs.attention_mask.to(device)
                prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=attention_mask)[0]
            else:
                x = video_data['video_latent'].to(device, non_blocking=True)  # (B,F,C,H,W)
                prompt_embeds = video_data['prompt_embedding'].to(device)
                if torch.isnan(prompt_embeds).any():
                    logger.warning("Prompt embeddings contain NaN values.")
                    continue

            x = scale_latents(vae, x)

            model_kwargs = {
                "encoder_hidden_states": prompt_embeds,
            }
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            if args.mixed_precision:
                with torch.amp.autocast(dtype=get_torch_dtype(args.mixed_precision), device_type='cuda'):
                    loss_dict = diffusion.training_losses(model, x, t, model_kwargs, channel_first=True)
                    loss = loss_dict["loss"].mean()
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs, channel_first=True)
                loss = loss_dict["loss"].mean()
                loss.backward()

            # for logging
            mse = loss_dict["mse"].mean().item() if "mse" in loss_dict else 0.0
            vb = loss_dict["vb"].mean().item() if "vb" in loss_dict else 0.0

            if (train_steps+1) % args.gradient_accumulation_steps == 0:
                # mem('start', logger)
                # logger.info('------------------------------------------------------------')
                # for n, p in model.named_parameters():
                #     if "module.base_model.model.temporal_transformer_blocks.0.attn1.content_gate" in n and p.grad is not None:
                #         logger.info(f"{n} - {p.grad.data.mean().item()}, std {p.grad.data.std().item()}, current_weights: {p.data.mean().item()}, std {p.data.std().item()}")

                if use_scaler:
                    scaler.unscale_(opt)

                if train_steps < args.start_clip_iter: # if train_steps >= start_clip_iter, will clip gradient
                    gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False)
                else:
                    gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=True)

                #logger.info(f'grad temporal_transformer_blocks.14.attn1.content_gate.proj.weight - mean: {model.module.temporal_transformer_blocks[14].attn1.content_gate.proj.weight.grad.mean().item()}, std: {model.module.temporal_transformer_blocks[14].attn1.content_gate.proj.weight.grad.std().item()}')
                # logger.info(f'grad temporal_transformer_blocks.14.attn1.content_gate.proj.bias - mean: {model.module.temporal_transformer_blocks[14].attn1.content_gate.proj.bias.grad.mean().item()}, std: {model.module.temporal_transformer_blocks[14].attn1.content_gate.proj.bias.grad.std().item()}')
                # logger.info(f'temporal_transformer_blocks.14.attn1.content_gate.proj.bias - mean: {model.module.temporal_transformer_blocks[14].attn1.content_gate.proj.bias.mean().item()}, std: {model.module.temporal_transformer_blocks[14].attn1.content_gate.proj.bias.std().item()}')

                # logger.info(f'grad temporal_transformer_blocks.14.attn1.content_gate.proj.weight - mean: {model.module.temporal_transformer_blocks[14].attn1.to_q.lora_A.default.weight.grad.mean().item()}, std: {model.module.temporal_transformer_blocks[14].attn1.to_q.lora_A.default.weight.grad.std().item()}')
            
                diffusion.synchronize_adaptive_frequency_gradients()

                if use_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()

                opt.zero_grad()
                lr_scheduler.step()
            
                # Update EMA:
                if rank == 0 and train_steps % ema_update_every == 0:
                    update_ema(ema, base_model, decay=0.9999**ema_update_every)

            avg_loss =  loss.detach()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            logger.info(f'loss: {avg_loss}')
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
            lr_scheduler.step()

            if train_steps % args.log_every == 0:
                # Measure training speed:
                # torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                # avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss =  loss.detach()
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
                        "model": base_model.state_dict(),
                        "ema": ema.state_dict(),
                        "train_steps": train_steps,
                        "adaptive_frequency": diffusion.adaptive_frequency_state_dict(),
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

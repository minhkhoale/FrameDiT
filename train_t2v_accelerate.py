# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Latte using PyTorch DDP.
"""

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import TorchDynamoPlugin
import logging

import torch
# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import io
import os
import math
import argparse

from glob import glob
from time import time
from copy import deepcopy
from models import get_models
from models.utils import load_pretrained_latte_into_framedith, freeze_model_for_matrix_training
from datasets import get_dataset
from tokenizer import get_tokenizer, _text_preprocessing
from vae import get_vae, encode_video, scale_latents
from diffusion import create_diffusion
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from utils import update_ema, requires_grad, cleanup, get_experiment_dir
import torch._inductor.config as cfg
cfg.triton.cudagraphs = False
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

# torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(True)
# torch.backends.cuda.enable_math_sdp(True)  # keep fallback


def mem(tag="", logger=None):
    a = torch.cuda.memory_allocated()/1e9
    r = torch.cuda.memory_reserved()/1e9
    m = torch.cuda.max_memory_allocated()/1e9
    logger.info(f"[{tag}] alloc={a:.1f}G reserved={r:.1f}G max_alloc={m:.1f}G")


def get_experiment_name(args):
        if args.debug:
            args.results_dir = os.path.join(args.results_dir, 'debug')

        experiment_index = len(glob(f"{args.results_dir}/{args.dataset}{args.image_size}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., Latte-XL/2 --> Latte-XL-2 (for naming folders)
        num_frame_string = 'F' + str(args.num_frames) + 'S' + str(args.frame_interval)
        os.makedirs(f"{args.results_dir}/{args.dataset}{args.image_size}", exist_ok=True)

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
        OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))

        return experiment_name, experiment_dir, checkpoint_dir


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    if args.use_compile:
        dynamo_plugin = TorchDynamoPlugin(
            # backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
            # mode="max-autotune",      # Options: "default", "reduce-overhead", "max-autotune"
            # fullgraph=True,
            # dynamic=False
        )
    else:
        dynamo_plugin = None

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_dir=args.results_dir,
        dynamo_backend="inductor" if args.use_compile else None,
        #   dynamo_plugin=dynamo_plugin
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

    logger.info(f"Accelerator initialized. Device: {device}, Rank: {rank}, World Size: {world_size}, Mixed Precision: {accelerator.mixed_precision}")
    if args.use_compile:
        logger.info("TorchDynamo compilation enabled with 'inductor' backend.")

    if rank == 0:
        experiment_name, experiment_dir, checkpoint_dir = get_experiment_name(args)
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

    seed = args.global_seed + rank
    torch.manual_seed(seed)
    if args.debug:
        logger.info("===============================\nRunning in debug mode.\n===============================")

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
    vae.requires_grad_(False)
    vae.eval()

    if not args.load_latent:
        logger.info('Load tokenizer and text encoder')
        tokenizer, text_encoder = get_tokenizer(OmegaConf.load(args.tokenizer))
        text_encoder = text_encoder.to(device)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    # clone args
    model_args = deepcopy(args)
    base_model = get_models(model_args)

    if args.gradient_checkpointing:
        n_blocks = args.n_checkpointing_blocks if 'n_checkpointing_blocks' in args else 999
        logger.info('enable gradient checkpointing for first {} blocks'.format(n_blocks))
        base_model.enable_gradient_checkpointing(n_blocks)

    logger.info('load latte model')
    load_pretrained_latte_into_framedith(base_model, args.pretrained_latte, logger, device)

    if args.pretrained:
        logger.info('load pretrained model')
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        if hasattr(args, 'using_pretrained_ema') and args.using_pretrained_ema:
            key = 'ema'
        else:
            key = 'model'
        
        logger.info(f'Using {key} ckpt!')
        checkpoint = checkpoint[key]
        
        ckpt_state_dict = {}
        for k in checkpoint.keys():
            ckpt_state_dict[k.replace('_orig_mod.', '')] = checkpoint[k]

        checkpoint = ckpt_state_dict

        # if "ema" in checkpoint:  # supports checkpoints from train.py
        #     logger.info('Using ema ckpt!')
        #     checkpoint = checkpoint["ema"]

        model_dict = base_model.state_dict()
        # LORA
        # if hasattr(args, 'use_lora') and args.use_lora:
        #     prefix_key = 'base_model.model.'
        # else:
        prefix_key = ''

        pretrained_dict = {}
        for k, v in checkpoint.items():
            if f"{prefix_key}{k}" in model_dict.keys():
                pretrained_dict[f"{prefix_key}{k}"] = v
            else:
                logger.info('Ignoring: {}'.format(f"{prefix_key}{k}"))
        logger.info(f'Successfully Load {len(pretrained_dict) / len(checkpoint.items()) * 100:.2f}% original pretrained model weights ')
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        base_model.load_state_dict(model_dict)
        logger.info('Successfully load model at {}!'.format(args.pretrained))

        # resume_step = checkpoint["train_steps"] if "train_steps" in checkpoint else 0
        # logger.info(f"Resuming training from step {resume_step}.")


    else:
        logger.info('Training from scratch!')
        
    resume_step = 0

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

    # for k in base_model.state_dict().keys():
    #     print(k)
    # exit(0)
    # Freeze model parameters for matrix training (if applicable)
    freeze_model_for_matrix_training(base_model, logger)
    if hasattr(args, 'gate_initialziation'):
        logger.info(f"Reset content gate with {args.gate_initialziation} initialization")
        base_model.reset_content_gate(args.gate_initialziation)

    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    
    logger.info("-" * 50)
    # print('Trainable Parameters:', trainable_param_names)
    logger.info(f"Total Parameters:     {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    logger.info("-" * 50)

    # for name, param in base_model.named_parameters():
    #     #if param.requires_grad:
    #     logger.info(f"{name}")

    # exit(0)

    diffusion = create_diffusion(
        name=args.diffusion_name if 'diffusion_name' in args else 'gaussian_diffusion',
        timestep_respacing=None,
        noise_schedule="linear",
        use_kl=False,
        sigma_small=args.sigma_small if 'sigma_small' in args else False,
        predict_xstart=args.predict_xstart if 'predict_xstart' in args else False,
        learn_sigma=args.learn_sigma if 'learn_sigma' in args else True,
    )  # default: 1000 steps, linear noise schedule
    logger.info(f'diffusion: {diffusion}')

    ema = None
    # EMA only for rank 0
    if rank == 0:    
        ema = deepcopy(base_model)  # Create an EMA of the model for use after training
        if hasattr(args, 'offload_ema') and args.offload_ema:
            logger.info("Offloading EMA model to CPU.")
            ema = ema.to('cpu')
        update_ema(ema, base_model, decay=0)
        requires_grad(ema, False)
        ema.eval()     

    if args.enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available
        if is_xformers_available():
            logger.info("Enabling xformers memory efficient attention.")
            base_model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        
    # set distributed training
    # if args.use_compile:
    #     logger.info("Using torch.compile for model compilation.")
    #     model = torch.compile(base_model)
    # else:
    #     model = base_model
    model = base_model
    model.train()   
    
    ema_update_every = args.ema_update_every if hasattr(args, "ema_update_every") else 1

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay if 'weight_decay' in args else 0)
    lr_scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Setup data
    dataset = get_dataset(args)
    logger.info(f"Dataset {dataset}")
    if args.load_latent:
        logger.info(f"Loading latent from {args.latent_path}")
    else:
        logger.info(f"Loading video from {args.data_path}")

    loader = DataLoader(
        dataset,
        batch_size=int(args.local_batch_size),
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    logger.info(f"Dataset contains {len(dataset):,} videos ({args.latent_path})")
    logger.info(f"Num frames per video: {args.num_frames}, frame interval: {args.frame_interval}")
    logger.info(f"Batch size per GPU: {args.local_batch_size}, global batch size: {args.local_batch_size * world_size}")
    logger.info(f"Learning rate: {args.learning_rate}, gradient accumulation steps: {args.gradient_accumulation_steps}")

    model, opt, loader, lr_scheduler = accelerator.prepare(model, opt, loader, lr_scheduler)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    grad_norm = 0.0
    first_epoch = 0
    prev_train_steps = 0
    running_loss = []

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(loader))
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # if args.pretrained:
    #     train_steps = int(args.pretrained.split("/")[-1].split('.')[0])

    logger.info(f"Memory allocated before training: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    total_start_time = time()
    start_time = time()

    for epoch in range(first_epoch, num_train_epochs):
        logger.info(f"Starting epoch {epoch+1}/{num_train_epochs} (train steps: {train_steps}/{args.max_train_steps})")
        for step, video_data in enumerate(loader):
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
                attention_mask = text_inputs.attention_mask
                prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=attention_mask.to(device))[0]
            else:
                x = video_data['video_latent']
                prompt_embeds = video_data['prompt_embedding']
                if torch.isnan(prompt_embeds).any():
                    logger.warning("Prompt embeddings contain NaN values.")
                    continue
            x = scale_latents(vae, x)

            with accelerator.accumulate(model):
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                loss_dict = diffusion.training_losses(model, x, t, {"encoder_hidden_states": prompt_embeds}, channel_first=True)
                loss = loss_dict["loss"].mean()

                if loss.isnan():
                    logger.warning("Loss is NaN, skipping this step.")
                    continue

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if train_steps >= args.start_clip_iter:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                    else:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), float('inf'))  # just compute grad norm without clipping

                    # for name, param in model.named_parameters():
                    #     if 'attn1' in name and param.grad is not None:
                    #         print(f"{name} grad norm: {param.grad.norm().item()}")

                    opt.step()
                    lr_scheduler.step()
                    opt.zero_grad()

                    if accelerator.is_main_process and train_steps % ema_update_every == 0:
                        update_ema(ema, base_model, decay=0.9999**ema_update_every)

            # logger.info(f'loss: {accelerator.gather_for_metrics(loss.detach()).item():.4f}')
            # # For logging
            # logger.info(f'accelerator.gather_for_metrics(loss.detach()) {accelerator.gather_for_metrics(loss.detach()).item():.4f}')
            #exit(0)
            # running_loss.append(accelerator.gather_for_metrics(loss.detach()).item())
            # running_loss_mse.append(loss_dict["mse"].mean().item() if "mse" in loss_dict else 0.0)
            # running_loss_vb.append(loss_dict["vb"].mean().item() if "vb" in loss_dict else 0.0)

            # Logging (only on rank 0):
            if train_steps % args.log_every == 0:
                reduced_loss = accelerator.gather_for_metrics(loss.detach())
                avg_loss = reduced_loss.mean().item()
                # avg_loss = sum(running_loss) / len(running_loss) if running_loss else 0.0  # mean of running_loss
                # running_loss = []  # reset running loss

                if 'mse' in loss_dict:
                    loss_mse_for_log = loss_dict["mse"].detach()
                    reduced_loss_mse = accelerator.gather_for_metrics(loss_mse_for_log)
                    avg_loss_mse = reduced_loss_mse.mean().item()
                else:
                    avg_loss_mse = None
                
                if 'vb' in loss_dict:
                    loss_vb_for_log = loss_dict["vb"].detach()
                    reduced_loss_vb = accelerator.gather_for_metrics(loss_vb_for_log)
                    avg_loss_vb = reduced_loss_vb.mean().item()
                else:
                    avg_loss_vb = None

                if accelerator.is_main_process:
                    total_elapsed_time = time() - total_start_time
                    elapsed_time = time() - start_time
                    logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, MSE: {avg_loss_mse:.4f} VB: {avg_loss_vb:.4f}, Gradient Norm: {grad_norm:.4f}, Train Steps/Sec: {(train_steps - prev_train_steps)/elapsed_time:.2f}")
                    mem("Training", logger)

                    logging_dict = {
                        "train/loss": avg_loss,
                        "train/grad_norm": grad_norm,
                    }
                    if avg_loss_mse is not None:
                        logging_dict["train/mse"] = avg_loss_mse
                    if avg_loss_vb is not None:
                        logging_dict["train/vb"] = avg_loss_vb

                    accelerator.log(logging_dict, step=train_steps)
                start_time = time()
                prev_train_steps = train_steps

            train_steps += 1

            # Save Latte checkpoint:
            if accelerator.is_main_process and train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "train_steps": train_steps
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("Done!")


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

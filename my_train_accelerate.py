# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Latte using Hugging Face Accelerate.
"""

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list

import argparse
import logging
import math
import os
from copy import deepcopy
from glob import glob
from time import time

import torch

# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import get_dataset
from diffusion import create_diffusion
from diffusers.optimization import get_scheduler
from models import get_models
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils import clip_grad_norm_, get_experiment_dir, requires_grad, update_ema
from vae import encode_video, get_vae, scale_latents

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

#################################################################################
#                                  Training Loop                                #
#################################################################################
BIN_EDGES = [0, 200, 400, 600, 800, 1000]
BIN_LABELS = ["0-199", "200-399", "400-599", "600-799", "800-999"]
NUM_BINS = 5


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _accelerate_mixed_precision(args):
    mixed_precision = getattr(args, "mixed_precision", False)
    if isinstance(mixed_precision, str):
        lowered = mixed_precision.lower()
        if lowered in {"no", "fp16", "bf16", "fp8"}:
            return lowered
        if lowered in {"false", "none"}:
            return "no"
        if lowered == "true":
            return "bf16"

    if _as_bool(mixed_precision):
        return "bf16"
    if _as_bool(getattr(args, "mixed_precision_16bit", False)):
        return "fp16"
    return "no"


def _config_for_tracker(args):
    return OmegaConf.to_container(args, resolve=True)


def get_experiment_name(args):
    if args.debug:
        args.results_dir = os.path.join(args.results_dir, "debug")

    os.makedirs(f"{args.results_dir}/{args.dataset}{args.image_size}", exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/{args.dataset}{args.image_size}/*"))
    model_string_name = args.model.replace("/", "-")
    num_frame_string = "F" + str(args.num_frames) + "S" + str(args.frame_interval)

    gaussian_name = args.diffusion_name if "diffusion_name" in args else None
    experiment_name = f"{experiment_index:03d}-{model_string_name}-{num_frame_string}-{args.dataset}{args.image_size}"

    if gaussian_name is not None:
        experiment_name += f"-{gaussian_name}"
    if args.get("experiment_suffix", None):
        experiment_name += f"-{args.experiment_suffix}"

    if args.num_frames == 16:
        experiment_dir = f"{args.results_dir}/{args.dataset}{args.image_size}/{experiment_name}"
    else:
        experiment_dir = f"{args.results_dir}/{args.dataset}{args.image_size}-{args.num_frames}/{experiment_name}"

    experiment_dir = get_experiment_dir(experiment_dir, args)
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    OmegaConf.save(args, os.path.join(experiment_dir, "config.yaml"))

    return experiment_name, experiment_dir, checkpoint_dir


def _latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None

    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.endswith(".pt")]
    if not checkpoints:
        return None

    checkpoints = sorted(checkpoints, key=lambda x: int(x.split(".")[0]))
    return os.path.join(checkpoint_dir, checkpoints[-1])


def _unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    mixed_precision = _accelerate_mixed_precision(args)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb" if getattr(args, "project", None) else None,
        project_dir=args.results_dir,
        dynamo_backend="inductor" if getattr(args, "use_compile", False) else None,
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

    experiment_dir = None
    checkpoint_dir = None
    if accelerator.is_main_process:
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

    experiment_dir = broadcast_object_list([experiment_dir])[0]
    checkpoint_dir = broadcast_object_list([checkpoint_dir])[0]

    seed = args.global_seed + rank
    torch.manual_seed(seed)
    logger.info(
        f"Starting rank={rank}, seed={seed}, world_size={world_size}, "
        f"mixed_precision={accelerator.mixed_precision}."
    )
    if args.debug:
        logger.info("===============================\nRunning in debug mode.\n===============================")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    sample_size = args.image_size // 8
    args.latent_size = sample_size

    vae = get_vae(OmegaConf.load(args.vae)).to(device)

    model_args = deepcopy(args)
    if vae.is_video_vae:
        num_frames = args.num_frames
        model_args.num_frames = (num_frames // 4) + 1
        logger.info(f"Using video VAE, adjusted num frames from {num_frames} to {model_args.num_frames}")

    model = get_models(model_args)

    diffusion = create_diffusion(
        name=args.diffusion_name if "diffusion_name" in args else "gaussian_diffusion",
        timestep_respacing=None,
        noise_schedule="linear",
        use_kl=False,
        sigma_small=args.sigma_small if "sigma_small" in args else False,
        predict_xstart=args.predict_xstart if "predict_xstart" in args else False,
        learn_sigma=args.learn_sigma if "learn_sigma" in args else True,
        adaptive_frequency=args.get("adaptive_frequency", False),
        adaptive_frequency_gamma=args.get("adaptive_frequency_gamma", 0.5),
        adaptive_frequency_learnable_gamma=args.get("adaptive_frequency_learnable_gamma", False),
        adaptive_frequency_gamma_mode=args.get("adaptive_frequency_gamma_mode", "scalar"),
        adaptive_frequency_power_path=args.get("adaptive_frequency_power_path", None),
        adaptive_frequency_power_exponent=args.get("adaptive_frequency_power_exponent", 2.0),
        adaptive_frequency_num_temporal_bands=args.get("adaptive_frequency_num_temporal_bands", None),
        adaptive_frequency_num_spatial_bands=args.get("adaptive_frequency_num_spatial_bands", None),
        equal_snr=args.get("equal_snr", False),
        equal_snr_power_path=args.get("equal_snr_power_path", None),
        equal_snr_power_scale=args.get("equal_snr_power_scale", 1.0),
        equal_snr_power_exponent=args.get("equal_snr_power_exponent", 2.0),
        equal_snr_calibrate_schedule=args.get("equal_snr_calibrate_schedule", False),
    )
    diffusion.initialize_adaptive_frequency_for_shape((1, args.num_frames, args.in_channels, args.latent_size, args.latent_size), device)
    for p in diffusion.adaptive_frequency_parameters():
        p.data = p.data.to(device)
    for p in diffusion.equal_snr_parameters():
        p.data = p.data.to(device)
    logger.info(f"diffusion: {diffusion}")
    logger.info(
        "Adaptive frequency: enabled=%s, gamma=%.6f, gamma_mode=%s, learnable_gamma=%s, power_path=%s, temporal_bands=%s, spatial_bands=%s",
        diffusion.adaptive_frequency.enabled,
        diffusion.adaptive_frequency.gamma(device=device).detach().float().item(),
        diffusion.adaptive_frequency.gamma_mode,
        diffusion.adaptive_frequency.learnable_gamma,
        getattr(diffusion.adaptive_frequency, "power_path", None),
        getattr(diffusion.adaptive_frequency, "num_temporal_bands", None),
        getattr(diffusion.adaptive_frequency, "num_spatial_bands", None),
    )
    logger.info(
        "EqualSNR: enabled=%s, power_path=%s, power_scale=%s",
        diffusion.equal_snr.enabled,
        getattr(diffusion.equal_snr, "power_path", None),
        getattr(diffusion.equal_snr, "power_scale", None),
    )

    if getattr(args, "pretrained", None):
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        if isinstance(checkpoint, dict) and "adaptive_frequency" in checkpoint:
            diffusion.load_adaptive_frequency_state_dict(checkpoint["adaptive_frequency"])
            logger.info("Loaded adaptive frequency state from pretrained checkpoint.")
        if isinstance(checkpoint, dict) and "equal_snr" in checkpoint:
            diffusion.load_equal_snr_state_dict(checkpoint["equal_snr"])
            logger.info("Loaded EqualSNR state from pretrained checkpoint.")
        if "ema" in checkpoint:
            logger.info("Using ema ckpt!")
            checkpoint = checkpoint["ema"]

        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                logger.info(f"Ignoring: {k}")
        logger.info(f"Successfully Load {len(pretrained_dict) / len(checkpoint.items()) * 100:.2f}% original pretrained model weights")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(f"Successfully load model at {args.pretrained}!")

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    vae.requires_grad_(False)
    vae.eval()

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    adaptive_frequency_params = list(diffusion.adaptive_frequency_parameters())
    equal_snr_params = list(diffusion.equal_snr_parameters())
    if adaptive_frequency_params:
        logger.info(f"Adaptive Frequency Parameters: {sum(p.numel() for p in adaptive_frequency_params):,}")
    if equal_snr_params:
        logger.info(f"EqualSNR Parameters: {sum(p.numel() for p in equal_snr_params):,}")
    learning_rate = getattr(args, "learning_rate", 1e-4)
    weight_decay = getattr(args, "weight_decay", 0)
    opt = torch.optim.AdamW(
        list(model.parameters()) + adaptive_frequency_params + equal_snr_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Setup data:
    dataset = get_dataset(args)
    logger.info(f"Dataset {dataset}")
    if args.load_latent:
        logger.info(f"Loading latent from {getattr(args, 'latent_path', args.data_path)}")
    else:
        logger.info(f"Loading video from {args.data_path}")

    loader = DataLoader(
        dataset,
        batch_size=int(args.local_batch_size),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    logger.info(f"Dataset contains {len(dataset):,} videos ({args.data_path})")
    logger.info(f"Num frames per video: {args.num_frames}, frame interval: {args.frame_interval}, image_size: {args.image_size}")
    logger.info(f"Batch size per GPU: {args.local_batch_size}, global batch size: {args.local_batch_size * world_size}, accumulation step: {args.gradient_accumulation_steps}")

    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model, opt, loader, lr_scheduler = accelerator.prepare(model, opt, loader, lr_scheduler)

    update_ema(ema, _unwrap_model(accelerator, model), decay=0)
    model.train()
    ema.eval()

    train_steps = 0
    log_steps = 0
    running_count = 0
    gradient_norm = torch.tensor(0.0, device=device)
    running_loss = 0
    running_loss_mse = 0
    running_loss_vb = 0
    running_bins = {
        "x_loss_sum": [0.0] * NUM_BINS,
        "count": [0.0] * NUM_BINS,
    }
    first_epoch = 0
    resume_step = 0
    start_time = time()

    num_update_steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if getattr(args, "resume_from_checkpoint", False):
        checkpoint_path = _latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None:
            logger.warning(f"No checkpoint found in {checkpoint_dir}; starting from scratch.")
        else:
            logger.info(f"Resuming from checkpoint {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            _unwrap_model(accelerator, model).load_state_dict(checkpoint["model"])
            ema.load_state_dict(checkpoint["ema"])
            if "adaptive_frequency" in checkpoint:
                diffusion.load_adaptive_frequency_state_dict(checkpoint["adaptive_frequency"])
                logger.info("Loaded adaptive frequency state from resume checkpoint.")
            if "equal_snr" in checkpoint:
                diffusion.load_equal_snr_state_dict(checkpoint["equal_snr"])
                logger.info("Loaded EqualSNR state from resume checkpoint.")
            train_steps = int(checkpoint.get("train_steps", os.path.basename(checkpoint_path).split(".")[0]))
            first_epoch = train_steps // num_update_steps_per_epoch
            resume_step = (train_steps % num_update_steps_per_epoch) * args.gradient_accumulation_steps

    if getattr(args, "pretrained", None):
        try:
            train_steps = int(os.path.basename(args.pretrained).split(".")[0])
        except ValueError:
            logger.warning("Could not infer train_steps from pretrained checkpoint filename.")

    total_start_time = time()
    for epoch in range(first_epoch, num_train_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_train_epochs} (train steps: {train_steps}/{args.max_train_steps})")

        for step, video_data in enumerate(loader):
            if getattr(args, "resume_from_checkpoint", False) and epoch == first_epoch and step < resume_step:
                continue

            # logger.info(f'step: {step} idx: {video_data['video_name']}')

            if args.load_latent:
                x = video_data["video"] if "video" in video_data else video_data["video_latent"]
                x = x.to(device, non_blocking=True)
            else:
                x = video_data["video"].to(device, non_blocking=True)
                x = encode_video(vae, x)

            x = scale_latents(vae, x)

            if args.extras == 78:
                raise NotImplementedError("T2V training is not supported in my_train_accelerate.py.")
            if args.extras == 2:
                model_kwargs = dict(y=video_data["video_name"])
            else:
                model_kwargs = dict(y=None)

            did_update = False
            with accelerator.accumulate(model):
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()

                if loss.isnan():
                    logger.warning("Loss is NaN, skipping this step.")
                    continue

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    unwrapped_model = _unwrap_model(accelerator, model)
                    if train_steps < args.start_clip_iter:
                        gradient_norm = clip_grad_norm_(unwrapped_model.parameters(), args.clip_max_norm, clip_grad=False)
                    else:
                        gradient_norm = accelerator.clip_grad_norm_(model.parameters(), args.clip_max_norm)

                    diffusion.synchronize_adaptive_frequency_gradients()
                    opt.step()
                    lr_scheduler.step()
                    opt.zero_grad()
                    update_ema(ema, unwrapped_model)
                    log_steps += 1
                    train_steps += 1
                    did_update = True

            mse = loss_dict["mse"].mean().detach() if "mse" in loss_dict else None
            vb = loss_dict["vb"].mean().detach() if "vb" in loss_dict else None

            running_loss += accelerator.gather(loss.detach()).mean().item()
            running_count += 1
            if mse is not None:
                running_loss_mse += accelerator.gather(mse).mean().item()
            if vb is not None:
                running_loss_vb += accelerator.gather(vb).mean().item()

            with torch.no_grad():
                x_loss_ps = loss_dict.get("loss", None)
                if x_loss_ps is not None:
                    gathered_t = accelerator.gather(t.detach())
                    gathered_loss = accelerator.gather(x_loss_ps.detach())
                    for i in range(NUM_BINS):
                        lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
                        mask = (gathered_t >= lo) & (gathered_t < hi)
                        if mask.any():
                            running_bins["x_loss_sum"][i] += gathered_loss[mask].sum().item()
                            running_bins["count"][i] += mask.sum().item()

            if did_update and train_steps % args.log_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / max(running_count, 1)
                avg_loss_mse = running_loss_mse / max(running_count, 1) if mse is not None else None
                avg_loss_vb = running_loss_vb / max(running_count, 1) if vb is not None else None

                if accelerator.is_main_process:
                    grad_norm_value = gradient_norm.item() if torch.is_tensor(gradient_norm) else float(gradient_norm)
                    mse_text = f", MSE: {avg_loss_mse:.4f}" if avg_loss_mse is not None else ""
                    vb_text = f", VB: {avg_loss_vb:.4f}" if avg_loss_vb is not None else ""
                    logger.info(
                        f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}{mse_text}{vb_text}, "
                        f"Gradient Norm: {grad_norm_value:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, "
                        f"Elapsed: {(time() - total_start_time):.2f}"
                    )

                    logging_dict = {
                        "train/loss": avg_loss,
                        "train/xs_loss": avg_loss,
                        "train/grad_norm": grad_norm_value,
                        "train/steps_per_sec": steps_per_sec,
                    }
                    if diffusion.adaptive_frequency.enabled:
                        gamma_stats = diffusion.adaptive_frequency.gamma_stats(device=device)
                        logging_dict["adaptive_frequency/gamma_mean"] = gamma_stats["mean"]
                        logging_dict["adaptive_frequency/gamma_min"] = gamma_stats["min"]
                        logging_dict["adaptive_frequency/gamma_max"] = gamma_stats["max"]
                        logging_dict["adaptive_frequency/gamma_std"] = gamma_stats["std"]
                        logging_dict["adaptive_frequency/gamma"] = gamma_stats["mean"]
                        logger.info(
                            "Adaptive Frequency Gamma: mean=%.6f, min=%.6f, max=%.6f, std=%.6f",
                            gamma_stats["mean"],
                            gamma_stats["min"],
                            gamma_stats["max"],
                            gamma_stats["std"],
                        )
                    if diffusion.equal_snr.enabled:
                        logging_dict["equal_snr/power_scale"] = diffusion.equal_snr.power_scale
                    if avg_loss_mse is not None:
                        logging_dict["train/mse"] = avg_loss_mse
                        logging_dict["train/xs_mse"] = avg_loss_mse
                    if avg_loss_vb is not None:
                        logging_dict["train/loss_vb"] = avg_loss_vb

                    for i, lbl in enumerate(BIN_LABELS):
                        if running_bins["count"][i] > 0:
                            logging_dict[f"train/bin_xs_loss/{lbl}"] = running_bins["x_loss_sum"][i] / running_bins["count"][i]

                    accelerator.log(logging_dict, step=train_steps)

                running_loss = 0
                running_loss_mse = 0
                running_loss_vb = 0
                running_count = 0
                running_bins = {
                    "x_loss_sum": [0.0] * NUM_BINS,
                    "count": [0.0] * NUM_BINS,
                }
                log_steps = 0
                start_time = time()

            should_save = did_update and train_steps % args.ckpt_every == 0 and train_steps > 0
            if should_save:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": _unwrap_model(accelerator, model).state_dict(),
                        "ema": ema.state_dict(),
                        "train_steps": train_steps,
                        "adaptive_frequency": diffusion.adaptive_frequency_state_dict(),
                        "equal_snr": diffusion.equal_snr_state_dict(),
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                accelerator.wait_for_everyone()

            if train_steps >= args.max_train_steps:
                break

        if train_steps >= args.max_train_steps:
            break

    model.eval()
    accelerator.end_training()
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train.yaml")
    parser.add_argument("--debug", action="store_true", help="If true, run in debug mode.")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()

    configs = OmegaConf.load(args.config)
    configs.debug = args.debug
    if args.num_workers is not None:
        configs.num_workers = args.num_workers

    main(configs)

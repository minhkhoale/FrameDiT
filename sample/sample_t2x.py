import os
import sys
import math
import torch
import imageio
import argparse
import torchvision
from omegaconf import OmegaConf
import torch.distributed as dist
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler, 
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler, 
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from pipeline_latte import LattePipeline

sys.path.append(os.path.split(sys.path[0])[0])
from models import get_models
from models.utils import load_pretrained_latte_into_framedith
from utils import save_video_grid, setup_distributed

def print(*args, **kwargs):
    __builtins__.print(*args, flush=True, **kwargs)


def build_jobs(prompt_list, num_videos: int):
    """
    Creates a flat job list: (job_id, prompt_idx, prompt_str, video_idx)
    """
    jobs = []
    job_id = 0
    for p_i, prompt in enumerate(prompt_list):
        for v_i in range(num_videos):
            jobs.append((job_id, p_i, prompt, v_i))
            job_id += 1
    return jobs

def shard_jobs(jobs, rank: int, world_size: int):
    """
    Simple contiguous sharding by rank.
    """
    n = len(jobs)
    if world_size <= 1:
        return jobs
    per = int(math.ceil(n / world_size))
    start = rank * per
    end = min(start + per, n)
    return jobs[start:end]


def main(args):
    setup_distributed()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    if args.seed is not None:
        seed = args.seed + rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.set_grad_enabled(False)

    transformer_model = get_models(args).to(device, dtype=torch.float16)
    load_pretrained_latte_into_framedith(transformer_model, args.pretrained_model_path, None, device)
    if hasattr(args, 'use_lora') and args.use_lora:
        from peft import get_peft_model, LoraConfig, TaskType
        print('use lora finetune')
        peft_config = LoraConfig(
            # task_type=TaskType.FEATURE_EXTRACTION,
            r=args.lora.r,
            lora_alpha=args.lora.lora_alpha,
            target_modules=args.lora.target_modules,
            lora_dropout=args.lora.lora_dropout,
        )
        transformer_model = get_peft_model(transformer_model, peft_config)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        ckpt_state_dict = {}
        ema_state_dict = {}

        model_key = 'model'
        for k in checkpoint[model_key].keys():
            if k.startswith('_orig_mod.'):
                new_k = k[len('_orig_mod.'):]
                ckpt_state_dict[new_k] = checkpoint[model_key][k]
            else:
                ckpt_state_dict[k] = checkpoint[model_key][k]
        checkpoint[model_key] = ckpt_state_dict
        transformer_model.load_state_dict(checkpoint[model_key], strict=True)
        print(f"Loaded pretrained model from {args.pretrained}")

    # print model to check
    # for name, param in transformer_model.named_parameters():
    #     if "lora" in name.lower():
    #         print(name, param.abs().mean().item(), param.abs().std().item())


    if args.enable_vae_temporal_decoder:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_path, subfolder="vae_temporal_decoder", torch_dtype=torch.float16).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae", torch_dtype=torch.float16).to(device)

    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    scheduler = get_sampling_scheduler(args)

    videogen_pipeline = LattePipeline(vae=vae, 
                                 text_encoder=text_encoder, 
                                 tokenizer=tokenizer, 
                                 scheduler=scheduler, 
                                 transformer=transformer_model).to(device)


    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path)

    if hasattr(args, 'disable_progress_bar') and args.disable_progress_bar:
        videogen_pipeline.set_progress_bar_config(disable=True)

    # Load prompts
    if isinstance(args.text_prompt, str):
        if os.path.exists(args.text_prompt):
            with open(args.text_prompt, 'r') as f:
                prompt_list = f.readlines()
            prompt_list = [prompt.strip() for prompt in prompt_list]
        else:
            print("Warning: Treat text_prompt as text")
            prompt_list = [args.text_prompt]
    else:
        prompt_list = args.text_prompt

    num_videos = args.num_videos if hasattr(args, 'num_videos') and args.num_videos is not None else 1
    jobs = build_jobs(prompt_list, num_videos)
    my_jobs = shard_jobs(jobs, rank, world_size)

    if rank == 0:
        print(f"Total jobs: {len(jobs)} | world_size: {world_size}")
    print(f"[rank {rank}] Running {len(my_jobs)} jobs on device={device}")
    
    for (job_id, prompt_id, prompt, video_id) in my_jobs:
        print(f'Prompt: {prompt} ({video_id})')
        file_path = os.path.join(args.save_img_path, f"{prompt}-{video_id}.mp4")
        if os.path.exists(file_path):
            print('\tExist, skip...')
            continue

        videos = videogen_pipeline(prompt, 
                                video_length=args.video_length,
                                height=args.image_size[0], 
                                width=args.image_size[1], 
                                num_inference_steps=args.num_sampling_steps,
                                guidance_scale=args.guidance_scale,
                                enable_temporal_attentions=args.enable_temporal_attentions,
                                num_images_per_prompt=1,
                                mask_feature=True,
                                enable_vae_temporal_decoder=args.enable_vae_temporal_decoder
                                ).video
    
        print('\tSaving to \"{}\"'.format(file_path))
        try:
            imageio.mimwrite(file_path, videos[0], fps=8, quality=10) # highest quality is 10, lowest is 0
        except:
            print('\tError when saving {}'.format(prompt))


def get_sampling_scheduler(args):
    if args.sample_method == 'DDIM':
        scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type,
                                                  clip_sample=False)
    elif args.sample_method == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_path, 
                                                        subfolder="scheduler",
                                                        beta_start=args.beta_start, 
                                                        beta_end=args.beta_end, 
                                                        beta_schedule=args.beta_schedule,
                                                        variance_type=args.variance_type)
    elif args.sample_method == 'DDPM':
        scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type,
                                                  clip_sample=False)
    elif args.sample_method == 'DPMSolverMultistep':
        scheduler = DPMSolverMultistepScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'DPMSolverSinglestep':
        scheduler = DPMSolverSinglestepScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'PNDM':
        scheduler = PNDMScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'HeunDiscrete':
        scheduler = HeunDiscreteScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'EulerAncestralDiscrete':
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'DEISMultistep':
        scheduler = DEISMultistepScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'KDPM2AncestralDiscrete':
        scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)

    return scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/wbv10m_train.yaml")
    parser.add_argument("--save-img-path", type=str)
    args = parser.parse_args()

    configs = OmegaConf.load(args.config)
    if args.save_img_path is not None:
        configs.save_img_path = args.save_img_path
    main(configs)


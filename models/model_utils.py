# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!


import os
import math
import torch

import numpy as np
import torch.nn as nn

from einops import repeat

from .latte_t2v import LatteT2V
from .framedit_h_t2v import FusedMatrixAttention


#################################################################################
#                                  Unet Utils                                   #
#################################################################################

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim).contiguous()
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def load_pretrained_latte_into_framedith(framedith_model, latte_pretrained_path, logger, device='cpu'):
    """
    Loads weights from a pretrained LatteT2V model into a FrameDiTHT2V model.
    
    Mapping Strategy:
    1. Spatial Blocks & Embeddings: Direct copy (keys match).
    2. Temporal Blocks: 
       - Latte's 'attn1' (Standard Attention) -> FrameDiT's 'attn1.attention' (Local Branch).
       - Matrix Attention branch and Fusion gates remain initialized from scratch.
    """
    if logger:
        logger.info(f"Loading Latte weights from: {latte_pretrained_path}")
    
    # 1. Load the source Latte model
    # We use the class definition provided in your context or diffusers
    # Assuming the config is compatible or loading state_dict directly
    try:
        latte_model = LatteT2V.from_pretrained(latte_pretrained_path, subfolder="transformer", video_length=16).to(device)
    except Exception as e:
        if logger:
            logger.info(f"Could not load as pipeline, trying to load state dict directly...")
        latte_state_dict = torch.load(os.path.join(latte_pretrained_path, "diffusion_pytorch_model.bin"), map_location=device)
    else:
        latte_state_dict = latte_model.state_dict()

    target_state_dict = framedith_model.state_dict()
    loaded_keys = []
    missing_keys = []
    shape_mismatch_keys = []

    if logger:
        logger.info("Starting weight injection...")

    for key, target_param in target_state_dict.items():
        source_key = key
        
        # --- Logic for Mapping Keys ---
        # Case A: Temporal Attention Injection
        # FrameDiT: temporal_transformer_blocks.0.attn1.attention.to_q.weight
        # Latte:    temporal_transformer_blocks.0.attn1.to_q.weight
        if "temporal_transformer_blocks" in key and "attn1.attention." in key:
            # Remap keys to pull from Latte's standard attn1
            source_key = key.replace("attn1.attention.", "attn1.")
        
        # Case B: Matrix Attention (New Components) -> SKIP
        elif "temporal_transformer_blocks" in key and "matrix_attention" in key:
            continue
        
        # Case C: Fusion Gates/Norms (New Components) -> SKIP
        # FusedMatrixAttention has 'norm_local', 'norm_global', 'alpha', 'gamma'
        elif "temporal_transformer_blocks" in key and any(x in key for x in ['norm_local', 'norm_global', 'alpha', 'gamma', 'output_linear']):
            continue

        # --- Attempt Load ---
        
        if source_key in latte_state_dict:
            source_param = latte_state_dict[source_key]
            
            if source_param.shape == target_param.shape:
                # Copy data
                with torch.no_grad():
                    target_param.copy_(source_param)
                loaded_keys.append(key)
            else:
                shape_mismatch_keys.append(f"{key} (Target: {target_param.shape} vs Source: {source_param.shape})")
        else:
            missing_keys.append(key)

    if logger:
        logger.info("-" * 50)
        logger.info(f"Successfully loaded {len(loaded_keys)} keys.")
    
    if logger:
        if len(shape_mismatch_keys) > 0:
            logger.info(f"\n[WARNING] Shape Mismatches (Skipped {len(shape_mismatch_keys)} keys):")
            for k in shape_mismatch_keys[:5]: logger.info(f" - {k}")
            if len(shape_mismatch_keys) > 5: logger.info(" ...")
        
    # Filter missing keys to strictly show unexpected missing keys
    # We expect matrix attention keys to be missing
    # unexpected_missing = [k for k in missing_keys if "matrix_attention" not in k and "gamma" not in k and "alpha" not in k and "norm_local" not in k and "norm_global" not in k and "output_linear" not in k]
    unexpected_missing = [k for k in missing_keys if all(x not in k for x in ["matrix_attention", "gamma", "alpha", "norm_local", "norm_global", "output_linear"])]
    
    if logger:
        if len(unexpected_missing) > 0:
            logger.info(f"[WARNING] Unexpected Missing Keys in Source ({len(unexpected_missing)} keys):")
            for k in unexpected_missing[:10]: logger.info(f" - {k}")
        else:
            logger.info("[SUCCESS] All base parameters loaded. Only new Matrix Attention layers are uninitialized.")

    return framedith_model


def freeze_framedit_h_for_training(model, logger):
    # 1) Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    trainable = []

    # 2) Unfreeze Matrix Attention + fusion inside fused blocks
    for name, m in model.named_modules():
        if isinstance(m, FusedMatrixAttention):
            # logger.info(f"Unfreezing Matrix block: {name}")

            if hasattr(m, "matrix_attention"):
                for n, p in m.matrix_attention.named_parameters():
                    p.requires_grad = True
                    trainable.append(f"{name}.matrix_attention.{n}")

            for attr in ("norm_local", "norm_global", "output_linear", "content_gate"):
                if hasattr(m, attr) and getattr(m, attr) is not None:
                    for n, p in getattr(m, attr).named_parameters():
                        p.requires_grad = True
                        trainable.append(f"{name}.{attr}.{n}")

            for n, p in m.named_parameters(recurse=False):
                if n in ("alpha", "gamma_local", "gamma_global"):
                    p.requires_grad = True
                    trainable.append(f"{name}.{n}")

    # 3) Unfreeze LoRA params everywhere
    for n, p in model.named_parameters():
        if ".A" in n or ".B" in n or "lora" in n.lower():
            p.requires_grad = True
            trainable.append(n)

    # logger.info(f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return sorted(set(trainable))

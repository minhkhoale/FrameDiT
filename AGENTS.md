# Agent Notes

This repository is a modified Latte video diffusion codebase. The original
project provides PyTorch implementations for latent diffusion transformers
for video generation; this fork adds several matrix-factorized attention and
FrameDiT/FrameDiTH text-to-video variants, plus preprocessing, sampling,
SLURM, and evaluation utilities.

## High-Level Purpose

- Train and sample latent diffusion transformer models for video generation.
- Support unconditional, class-conditional, image/video joint, and text-to-video
  workflows.
- Experiment with efficient temporal/spatial attention variants:
  `MatLatte`, `FusedMatLatte`, `DiffLatte`, `SpatialDiffLatteV2`,
  `TemporalDiffLatteV2`, and `FrameDiTHT2V`.
- Train newer T2V models by loading Latte-1 weights, adding matrix/global
  temporal attention branches, optionally adding LoRA, and freezing most base
  parameters.

## Main Entry Points

- `train.py`: original DDP Latte training loop for non-T2V latent/video data.
  It builds the model through `models.get_models`, the dataset through
  `datasets.get_dataset`, the diffusion object through `diffusion.create_diffusion`,
  and a configured VAE through `vae.get_vae`.
- `train_with_img.py`: image/video joint training variant.
- `train_t2v.py`: main local T2V training path. It uses tokenizer/text encoder
  support, latent text-video datasets, optional mixed precision, W&B logging,
  LoRA, gradient checkpointing, and FrameDiT/FrameDiTH weight initialization.
- `train_t2v_accelerate.py`: Accelerate-based T2V training variant.
- `train_t2v_lora.py`: LoRA-focused T2V training variant.
- `sample/sample.py` and `sample/sample_ddp.py`: original Latte sampling.
- `sample/sample_t2x.py`: T2V/T2I sampling path around `LattePipeline`,
  diffusers schedulers, Latte-1 VAE/tokenizer/text encoder, and local
  FrameDiTHT2V checkpoints.
- `scripts/`: latent preprocessing and dataset preparation scripts.
- `tools/`: metrics, preprocessing, tracking, and utility code.
- `slurm_scripts/`: cluster launch scripts for training, sampling, latent
  extraction, and paper experiments.

## Configuration

Most runs are driven by OmegaConf YAML files under `configs/`.

- Dataset fields include `dataset`, `data_path`, `latent_path`,
  `prompt_latent_path`, `load_latent`, `num_frames`, `frame_interval`, and
  `image_size`.
- Model fields include `model`, `in_channels`, `learn_sigma`, `extras`,
  `attention_mode`, `gradient_checkpointing`, and T2V-specific fields such as
  `pretrained_latte`, `pretrained_model_path`, `tokenizer`, and `vae`.
- Training fields include `learning_rate`, `local_batch_size`,
  `gradient_accumulation_steps`, `max_train_steps`, `ckpt_every`,
  `clip_max_norm`, `mixed_precision`, `use_compile`, and W&B `project`.

Representative configs:

- `configs/ffs/ffs_train.yaml`: original FaceForensics Latte training.
- `configs/bair/matlatte_train.yaml`: BAIR latent training with `MatLatte`.
- `configs/t2v/framedit_h.yaml`: OpenVid latent T2V training with
  `FrameDiTHT2V`.
- `configs/pexels/*` and `configs/multiple/*`: additional local T2V datasets
  and LoRA/global/matrix attention experiments.

## Model System

`models/__init__.py` is the central factory. `get_models(args)` maps
`args.model` to the correct registry:

- `Latte_models` in `models/latte.py`
- `LatteIMG_models` in `models/latte_img.py`
- `LatteT2V` in `models/latte_t2v.py`
- `MatLatte_models` in `models/mat_latte.py`
- `MatLatteV2_models` in `models/mat_lattev2.py`
- `FusedMatLatte_models` in `models/fused_mat_latte.py`
- `FusedMatLatteIMG_models` in `models/fused_mat_latte_img.py`
- `FusedMatLatte1D_models` in `models/fused_mat_latte_1d.py`
- `DiffLatte` and V2 variants
- `DiT3D_models`
- `FrameDiTHT2V_models` in `models/framedit_h_t2v.py`

The baseline Latte architecture patchifies latent video, adds spatial and
temporal positional embeddings, applies alternating spatial/temporal
transformer blocks, conditions on timesteps and optional labels/text, and
unpatchifies to latent-space noise or noise-plus-variance predictions.

The matrix variants replace or augment dense projections/attention with
`MatrixLinear` and `MatrixAttention`, which factor row/token and channel
dimensions through learned matrices. Several variants support `param`,
`softmax`, `normalized_l1`, `normalized_l2`, `identity`, and sparse row-mixing
matrices.

`FrameDiTHT2V` is a diffusers-style `ModelMixin`/`ConfigMixin` transformer.
It extends Latte-style temporal blocks with local attention, matrix/global
attention branches, fusion gates, content gates, and optional LoRA.

Important helper functions:

- `models.utils.load_pretrained_latte_into_framedith`: loads LatteT2V weights
  into FrameDiTHT2V, remapping temporal `attn1` weights into the local branch
  and leaving new matrix/gating parameters initialized from scratch.
- `models.utils.freeze_model_for_matrix_training`: freezes all base weights,
  then unfreezes `FusedMatrixAttention` matrix/global/fusion/content-gate
  parameters plus LoRA parameters.

## Diffusion System

`diffusion/__init__.py` provides `create_diffusion(...)`. It builds beta
schedules and selects a spaced diffusion wrapper:

- `gaussian_diffusion`: standard `SpacedDiffusion`
- `difference_gaussian_diffusion_v0`
- `difference_gaussian_diffusion_v1`
- `mean_gaussian_diffusion_v0`
- `spatial_difference_gaussian_diffusion_v2`
- `temporal_difference_gaussian_diffusion_v2`
- `gaussian_diffusion_v2`

Training generally samples random timesteps, calls
`diffusion.training_losses(model, x, t, model_kwargs, ...)`, backpropagates the
mean loss, clips gradients after a configured threshold, steps AdamW, and
updates EMA parameters.

## Dataset System

`datasets/__init__.py` provides `get_dataset(args)`. It chooses dataset classes
with Python `match` on `args.dataset`, configures video transforms, and returns
video, latent-video, image, or text-video datasets.

Supported names include:

- `ffs`, `ffs_img`, `ffs_whole`
- `ucf101`, `ucf101_img`, `ucf101_whole`
- `taichi`, `taichi_img`, `taichi_whole`
- `sky`, `sky_img`, `sky_whole`
- `bair`
- `kinetics600`
- `latent_text_video`, `pexels`, `multiple`
- `openvid`

Latent T2V training typically consumes `video_latent` plus
`prompt_embedding`; pixel T2V training encodes video through the VAE and
computes prompt embeddings online.

## VAE System

`vae/__init__.py` provides:

- `get_vae(args)`: builds configured VAE wrappers.
- `encode_video(model, x)`: converts `(B,F,C,H,W)` videos to latent videos.
- `scale_latents(model, latents)`: applies the correct scaling/statistics.
- `decode_video(model, latents)`: converts latent videos back to pixels.

Supported VAE paths include:

- `autoencoder_kl`
- `lattet2v_autoencoder_kl`
- `video_vae`
- `titok_kl`
- `dc_ae` appears partially disabled because `MyAutoencoderDC` is commented
  out in the current `vae/__init__.py`.

## Sampling

Original Latte sampling uses local scripts under `sample/`.

T2V sampling uses `sample/sample_t2x.py`, which:

1. Initializes DDP and shards prompt/video jobs by rank.
2. Builds a transformer through `get_models(args)`.
3. Loads Latte weights into FrameDiTHT2V and optionally applies LoRA.
4. Loads a local checkpoint if `args.pretrained` is set.
5. Loads Latte-1 VAE, tokenizer, and T5 text encoder.
6. Chooses a diffusers scheduler such as DDIM, DDPM, PNDM, Euler, or DPM-Solver.
7. Runs `LattePipeline` and writes MP4 files with `imageio`.

## Utilities

`utils.py` contains common DDP setup, logging, experiment directory naming,
EMA updates, gradient clipping, video saving, and helper functions. It supports
both SLURM and torchrun-style environment variables.

`tools/` contains dataset conversion, preprocessing, metric implementations,
tracking utilities, and W&B logging helpers. `vbench_eval/` contains VBench
evaluation wrappers.

## Local Caveats Noted While Reading

- The sandbox in this environment could not start because `bubblewrap` is not
  installed, so inspection commands had to be run outside the sandbox.
- `rg` is not installed in this environment; `find` and `grep` were used.
- `git status --short` shows existing deleted files:
  `my_train_difference.py`, `my_train_difference_v0.py`,
  `my_train_difference_v1.py`, `my_train_difference_v2.py`,
  `my_train_difference_v2_bins.py`, `my_train_difference_zero.py`, and
  `my_train_mean.py`. These were not touched.
- Some dataset branches reference classes that are not imported in
  `datasets/__init__.py`, such as `UCF101Latent`, `TaichiPreprocess`, and
  `Kinetics600Latent`.
- `vae/__init__.py` references `MyAutoencoderDC`, but its import is commented
  out. Configs using `dc_ae` may fail unless that import is restored.
- `train_t2v.py` appears to call `lr_scheduler.step()` both inside the
  optimizer-step block and again once per loop iteration, so learning-rate
  schedule behavior should be checked before relying on exact step counts.
- Several configs contain machine-specific absolute paths under `/scratch`.

## Mental Model For Changes

When adding a new model variant:

1. Implement the module under `models/`.
2. Add its registry dictionary or constructor function.
3. Register it in `models/__init__.py`.
4. Add a config selecting `model: YourModel-...`.
5. Verify tensor layout expectations: most latent videos are `(B,F,C,H,W)`,
   but transformer internals often rearrange between frame, spatial-token, and
   channel dimensions.

When adding a new dataset:

1. Implement a dataset class under `datasets/`.
2. Return dictionaries with keys expected by the chosen training script:
   `video` for pixel video, `video_latent` for latent video, `prompt` or
   `prompt_embedding` for T2V, and optional `video_name`/labels.
3. Add a branch in `datasets.get_dataset`.
4. Add a config and update preprocessing scripts if latents are required.

When changing T2V matrix-attention training:

1. Start at `models/framedit_h_t2v.py` for architecture behavior.
2. Check `models.utils.load_pretrained_latte_into_framedith` for weight
   compatibility and remapping.
3. Check `models.utils.freeze_model_for_matrix_training` for which parameters
   are trainable.
4. Use `train_t2v.py` as the authoritative DDP training path unless the
   Accelerate script is explicitly being targeted.

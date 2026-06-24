# Adaptive-Frequency Gamma Implementation Notes

This note records what is implemented in this repo for adaptive-frequency
diffusion, with emphasis on the two experiment families we want to focus on
next:

- fixed scalar gammas
- learnable noise-level-wise gammas, currently named `timestep` gamma mode

## Core Method

The implementation lives in `diffusion/adaptive_frequency.py` as
`AdaptiveFrequencyTimesteps`.

For a normal VP/DDPM timestep, the code starts from the usual global scalars:

```text
x_t = alpha_t x_0 + sigma_t eps
alpha_t = sqrt(alpha_bar_t)
sigma_t = sqrt(1 - alpha_bar_t)
```

Adaptive frequency keeps the model conditioning as one scalar timestep `t`,
but changes the actual noising trajectory per Fourier frequency. The video
latent is FFT-transformed over `(F, H, W)`, then each frequency uses local
coefficients:

```text
local_sigma_b^2 =
    sigma_t^2 / (sigma_t^2 + alpha_t^2 * (P_bar / P_b)^gamma)

local_alpha_b = sqrt(1 - local_sigma_b^2)
```

where:

- `P_b` is the dataset-prior power for frequency `b`
- `P_bar` is the mean prior power
- `gamma` controls how strongly frequencies are equalized

Interpretation:

```text
gamma = 0.0  -> original global diffusion schedule
gamma = 1.0  -> hard EqualSNR-style local schedules
0 < gamma < 1 -> soft interpolation between them
```

The adaptive path is used in:

- `q_sample(...)`: forward noising during training
- `predict_xstart_from_eps(...)`: epsilon-to-x0 conversion
- `predict_eps_from_xstart(...)`: x0-to-epsilon conversion
- `ddim_step(...)`: deterministic DDIM sampling step, currently only `eta=0`

The FFT computation uses float32 for half/bfloat16 inputs and returns to the
original dtype afterward.

## Dataset Frequency Prior

The prior can be loaded from `adaptive_frequency_power_path`. The expected
dense key is:

```text
frequency_power_mean: [F, H, W]
```

If `adaptive_frequency_num_temporal_bands` and
`adaptive_frequency_num_spatial_bands` are set, the dense prior is rebinned
into a temporal-by-spatial grid. Runtime then builds a `[F, H, W]` bin map and
uses one power value per bin.

If no power file is provided, the code falls back to a synthetic radial power
law controlled by `adaptive_frequency_power_exponent`.

The stats script is:

```text
scripts/adaptive_frequency/compute_adaptive_frequency_power.py
```

It supports latent inputs, UCF101 Gaussian posterior latents, RGB fallback, and
saves both dense and pre-binned statistics.

## Fixed Gamma Runs

Fixed gamma is the cleanest current baseline. In config:

```yaml
adaptive_frequency: True
adaptive_frequency_gamma: 0.5
adaptive_frequency_learnable_gamma: False
adaptive_frequency_power_path: results_adaptive_schedule/...
adaptive_frequency_num_temporal_bands: 4
adaptive_frequency_num_spatial_bands: 8
```

Existing fixed-gamma sweeps include:

- `configs/ucf101_adaptive_schedule/ucf101_128_matlatte_gamma_{0.0,0.25,0.5,0.75,1.0}_train.yaml`
- `configs/ucf101_adaptive_schedule/precomputed_freq/ucf101_128_matlatte_gamma_{0.0,0.25,0.5,0.75,1.0}_train_precomputed_freq.yaml`
- `configs/ucf101_256_adaptive_schedule/precomputed_freq/ucf101_256_matlatte_gamma_{0.0,0.25,0.5,0.75,1.0}_train_precomputed_freq.yaml`

The precomputed-prior configs are the better comparison target because they use
dataset frequency statistics instead of the synthetic radial fallback.

## Learnable Noise-Level-Wise Gamma

The noise-level-wise version is implemented as:

```yaml
adaptive_frequency_learnable_gamma: True
adaptive_frequency_gamma_mode: timestep
```

Internally this creates:

```text
_raw_gamma_timesteps: [num_train_timesteps]
gamma_t = sigmoid(_raw_gamma_timesteps[t])
```

So every diffusion timestep/noise level has its own constrained gamma in
`[0, 1]`. The config value `adaptive_frequency_gamma` is treated as the
initial gamma before the logit transform.

Current configs:

- `configs/ucf101_adaptive_schedule/learnable_gamma/ucf101_128_matlatte_learnable_timestep_gamma_train.yaml`
- `configs/ucf101_256_adaptive_schedule/learnable_gamma/ucf101_256_matlatte_learnable_timestep_gamma_train.yaml`
- additional 256-resolution frequency-bin-count variants under
  `configs/ucf101_256_adaptive_schedule/freq_bin_*/learnable_gamma/`

This is the variant that matches "learnable noiselevel-wise gammas" most
closely. The current name is timestep-wise because the implementation indexes
by integer diffusion timestep, but semantically it is one gamma per noise
level in the training schedule.

## Other Gamma Modes Already Present

These are implemented but are lower priority for the next focus:

- `scalar`: one learnable scalar gamma shared by all timesteps and frequencies
- `frequency_bin`: one learnable gamma per frequency bin
- `data_dependent`: gamma is predicted from per-sample FFT energy using a
  learned bias and scale

All learnable modes use sigmoid-constrained raw parameters so the effective
gamma stays in `[0, 1]`.

## Training Integration

The adaptive-frequency options are threaded through `create_diffusion(...)` in:

- `train.py`
- `train_with_img.py`
- `train_t2v.py`
- `my_train_accelerate.py`

Learnable gamma parameters are exposed by:

```text
diffusion.adaptive_frequency_parameters()
```

and added to the optimizer. In DDP paths, gradients for these diffusion-owned
parameters are synchronized manually with:

```text
diffusion.synchronize_adaptive_frequency_gradients()
```

Checkpoints save and load:

```text
checkpoint["adaptive_frequency"] = diffusion.adaptive_frequency_state_dict()
```

`my_train_accelerate.py` also logs:

```text
adaptive_frequency/gamma_mean
adaptive_frequency/gamma_min
adaptive_frequency/gamma_max
adaptive_frequency/gamma_std
```

For timestep-wise gamma, these stats summarize all 1000 timestep gammas.

## Practical Next Focus

For the next experiments, keep the comparison narrow:

1. Fixed gamma sweep with dataset-prior power:
   `gamma in {0.0, 0.25, 0.5, 0.75, 1.0}`.
2. Learnable noise-level-wise gamma:
   `adaptive_frequency_gamma_mode: timestep`.
3. Use the same dataset prior, band counts, model, resolution, seed, and
   training budget across both families.
4. Track learned `gamma_t` curves, not only mean/min/max/std. The existing
   stats are useful for monitoring but do not show where in the noise schedule
   gamma changes.

The main missing piece for the noise-level-wise analysis is a small logging or
checkpoint-inspection utility that writes the full learned `gamma_t` vector as
a curve over diffusion timestep/noise level.

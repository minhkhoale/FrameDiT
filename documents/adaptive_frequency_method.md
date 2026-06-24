# Adaptive Frequency Timesteps

This document describes the current adaptive-frequency timestep method and how
it is implemented in this FastVideo branch. The method is experimental and is
implemented for Wan-style latent video inference.

## Motivation

Standard diffusion and flow-matching samplers use one scalar timestep for the
entire sample. In Fourier space this means all spatial and temporal frequency
components share the same noise level. This is not well matched to natural
image and video spectra: low frequencies have much larger power than high
frequencies, so their signal-to-noise ratios evolve differently under white
noise.

For a flow-matching latent at global noise level `sigma`, with signal
coefficient

```text
alpha = 1 - sigma,
```

the per-frequency SNR is

```text
SNR_b(sigma) = C_b * (1 - sigma)^2 / sigma^2,
```

where `b` indexes a spatiotemporal frequency bin and `C_b` is the dataset-prior
Fourier power of that bin. Because `C_b` differs strongly by frequency, a single
global timestep does not put all frequencies in the same denoising regime.

The core idea is therefore:

```text
Use one scalar model timestep for the transformer,
but update different frequency bands with different local sigma trajectories.
```

The model still receives normal latent tensors in `[B, C, F, H, W]` space. The
scheduler update is applied in FFT space.

## Dataset Prior

The adaptive schedule needs a dataset prior over latent-video frequency power.
The current preferred artifact is:

```text
experiments/adaptive_frequency_stats/openvidhd/dataset_prior_frequency_stats.npz
```

The runtime expects the `.npz` to contain:

```text
frequency_power_mean: [F_lat, H_lat, W_lat]
shape_fhw:            [3]
```

The source script that creates this artifact is:

```text
fastvideo/entrypoints/adaptive_frequency_stats.py
```

It can read latent tensors directly or read RGB videos with Decord. When
`--encode-with-vae` is enabled, it loads the model VAE, encodes videos into
latent space, and computes the FFT power there. Latent-space stats are preferred
because inference also updates Wan latents, not RGB pixels.

For each input latent video `x` shaped `[B, C, F, H, W]`, the script computes:

```text
X = FFT(x, axes=(F,H,W), norm="ortho")
power(F,H,W) = mean_{B,C} |X|^2
```

Then it averages `power` over all processed videos and saves only the raw
per-frequency power. Band masks are not stored. At inference time, the raw power
is re-binned on the fly into the requested number of temporal and spatial
frequency bands.

## Frequency Binning

Runtime loading is implemented in:

```text
fastvideo/models/schedulers/adaptive_frequency.py
```

The function `load_frequency_power_stats_for_inference()` loads
`frequency_power_mean` and calls `rebin_frequency_power()`.

For a latent FFT grid `[F, H, W]`, the code defines:

```text
temporal_radius = abs(fftfreq(F) * F)
spatial_radius  = sqrt((fftfreq(H) * H)^2 + (fftfreq(W) * W)^2)
```

The temporal and spatial edges are linear bins from zero to each radius maximum.
For every `(temporal_band, spatial_band)` bin, the code computes:

```text
C_mean[b]        = mean frequency_power_mean over the bin
num_frequencies[b] = number of FFT coordinates in the bin
```

The bin map used during sampling is built by `build_frequency_bin_map()`. It
returns a tensor shaped:

```text
[F_lat, H_lat, W_lat]
```

where each FFT coordinate stores its flattened frequency-bin id.

## Local Sigma Schedules

Local schedules are built by `build_local_sigma_schedules()`.

The function takes the original scheduler sigmas as `global_sigmas`. For Wan,
these sigmas come from `FlowUniPCMultistepScheduler` after the configured flow
shift has already been applied:

```text
sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
```

The local schedule tensor has shape:

```text
[num_bins, num_inference_steps + 1]
```

Each row is the local sigma trajectory for one spatiotemporal frequency bin.

### EqualSNR

For EqualSNR, the target global SNR at global sigma `sigma_k` is:

```text
rho_k = P_bar * (1 - sigma_k)^2 / sigma_k^2
```

where

```text
P_bar = sum_b num_frequencies[b] * C_b / sum_b num_frequencies[b].
```

The local sigma for bin `b` is chosen so that this bin has SNR `rho_k`:

```text
C_b * (1 - sigma_{b,k})^2 / sigma_{b,k}^2 = rho_k.
```

Solving gives the implemented flow-matching form:

```text
sigma_{b,k}
= sigma_k /
  (sigma_k + (1 - sigma_k) * sqrt(P_bar / C_b)).
```

This makes low-power and high-power frequency bins follow different effective
timestep trajectories while sharing the same global endpoints.

### Soft EqualSNR

Hard EqualSNR can move frequency bands aggressively. The soft version adds a
continuous interpolation strength:

```text
gamma in [0, 1]
```

The implemented local sigma is:

```text
sigma_{b,k}
= sigma_k /
  (sigma_k + (1 - sigma_k) * (P_bar / C_b)^(gamma / 2)).
```

Interpretation:

```text
gamma = 0: all bins use the original global sigma schedule
gamma = 1: hard EqualSNR
0 < gamma < 1: partial frequency-dependent schedule deformation
```

The inference knob is:

```text
adaptive_frequency_soft_equal_snr_gamma
```

or, in the source inference script:

```text
--af-soft-equal-snr-gamma
```

### LogSNR and Rho Modes

The config also exposes:

```text
adaptive_frequency_method = logsnr | rho | equal_snr | soft_equal_snr
```

In the current source, `equal_snr` and `soft_equal_snr` are the active
frequency-dependent methods. The current `logsnr` and `rho` paths compute SNR
from the original global sigma grid and then invert it back to sigma. With the
present formulas, this mostly reconstructs the original global schedule and is
best viewed as scaffolding for future alternative schedule construction.

## Scalar Model Timestep

The transformer still expects one scalar timestep per denoising call. The code
therefore stores two timestep concepts:

```text
local_timesteps:  [num_bins, num_inference_steps + 1]
global_timesteps: [num_inference_steps]
```

`local_timesteps` are debug conversions from local sigmas:

```text
t_{b,k} = round(num_train_timesteps * sigma_{b,k})
```

`global_timesteps` are the scalar timesteps passed to the model.

For `equal_snr` and `soft_equal_snr`, the current implementation keeps
`global_timesteps` equal to the original scheduler timesteps. This is deliberate
in the current code path: the model conditioning remains aligned to the Wan
base scheduler, while the frequency-dependent behavior happens in the FFT
latent update.

For the `logsnr` and `rho` paths, `compute_global_timesteps()` can derive a
scalar global timestep from the mixed local state. It supports two reductions.

The direct SNR reduction computes:

```text
signal_power(k) = sum_b N_b * (1 - sigma_{b,k})^2 * C_b
noise_power(k)  = sum_b N_b * sigma_{b,k}^2
SNR_global(k)   = signal_power(k) / noise_power(k)
```

Then it maps that global SNR back to a scalar flow sigma:

```text
C_global = sum_b N_b * C_b / sum_b N_b

sigma_global(k)
= 1 / (1 + sqrt(SNR_global(k) / C_global)).
```

The `logsnr` reduction averages local log-SNRs instead:

```text
logSNR_b(k)
= log C_b + 2 log(1 - sigma_{b,k}) - 2 log sigma_{b,k}

logSNR_global(k)
= sum_b N_b * logSNR_b(k) / sum_b N_b.
```

Both reductions can optionally blend with the original timestep:

```text
t_model = blend * t_adaptive + (1 - blend) * t_original.
```

The knob is:

```text
adaptive_frequency_global_timestep_blend
```

## Scheduler Update in FFT Space

The adaptive schedulers are implemented as scheduler classes under:

```text
fastvideo/models/schedulers/
```

The shared base class is:

```text
adaptive_frequency_base.py
```

It wraps a normal base scheduler, delegates unknown attributes to it, and adds:

```text
set_adaptive_frequency_schedule(schedule)
```

When the schedule is set, the adaptive scheduler exposes:

```text
self.timesteps = schedule.global_timesteps
self.sigmas    = schedule.local_sigmas
```

so the denoising loop can keep treating it as `self.scheduler`.

### Adaptive Euler

`AdaptiveFrequencyEulerScheduler` implements the simplest update:

```text
latent_fft = FFT(latents)
pred_fft   = FFT(model_output)

delta_sigma_b = local_sigma[b, k+1] - local_sigma[b, k]
latent_fft[w] = latent_fft[w] + delta_sigma_{bin(w)} * pred_fft[w]

latents = IFFT(latent_fft).real
```

The model prediction is still produced from the pixel/latent-domain tensor.
Only the scheduler update is frequency-specific.

### Adaptive UniPC

`AdaptiveFrequencyUniPCMultistepScheduler` implements a frequency-map version
of Flow UniPC. It supports solver order 1 or 2.

For each FFT coordinate, the scheduler uses the local sigma of that coordinate's
frequency bin. It computes local:

```text
alpha = 1 - sigma
lambda = log(alpha) - log(sigma)
```

and applies UniPC predictor/corrector coefficients as FFT-shaped maps. The
method stores recent model outputs in FFT space and updates each frequency
coordinate according to its own local sigma trajectory.

This is closer to the original Wan sampler than Euler, but it is still an
experimental adaptation because the original UniPC multistep state was designed
for one global timestep trajectory, not multiple local trajectories.

## Pipeline Integration

The Wan pipeline initializes the scheduler in:

```text
fastvideo/pipelines/basic/wan/wan_pipeline.py
```

The default scheduler is:

```text
FlowUniPCMultistepScheduler(shift=flow_shift)
```

If `adaptive_frequency_timesteps` is false, the pipeline returns this base
scheduler unchanged.

If `adaptive_frequency_timesteps` is true, Wan wraps the base scheduler as:

```text
AdaptiveFrequencyEulerScheduler(base_scheduler)
```

or:

```text
AdaptiveFrequencyUniPCMultistepScheduler(base_scheduler)
```

depending on:

```text
adaptive_frequency_scheduler = euler | unipc
```

The normal `TimestepPreparationStage` is still used. It asks the scheduler to
prepare the original Wan timesteps and sigmas. Then `DenoisingStage` calls
`prepare_adaptive_frequency_schedule()` before the denoising loop if adaptive
mode is enabled.

That preparation step:

1. Requires Wan latents shaped `[B, C, F, H, W]`.
2. Rejects sequence-parallel latent sharding for v1.
3. Requires one scheduler timestep per inference step.
4. Requires `adaptive_frequency_stats_path`.
5. Reads the original scheduler sigmas.
6. Loads and re-bins dataset frequency-power stats.
7. Builds local sigma schedules and a latent FFT bin map.
8. Calls `self.scheduler.set_adaptive_frequency_schedule(...)`.
9. Stores debug tensors on `ForwardBatch`.

The debug fields are:

```text
adaptive_frequency_local_sigmas
adaptive_frequency_local_timesteps
adaptive_frequency_global_timesteps
adaptive_frequency_bin_map
adaptive_frequency_debug
```

## Config and CLI Knobs

The main config fields live in:

```text
fastvideo/configs/pipelines/base.py
```

The current knobs are:

```text
adaptive_frequency_timesteps: bool
adaptive_frequency_scheduler: "euler" | "unipc"
adaptive_frequency_method: "logsnr" | "rho" | "equal_snr" | "soft_equal_snr"
adaptive_frequency_stats_path: str
adaptive_frequency_num_temporal_bands: int
adaptive_frequency_num_spatial_bands: int
adaptive_frequency_global_timestep_method: "snr" | "logsnr"
adaptive_frequency_global_timestep_blend: float
adaptive_frequency_soft_equal_snr_gamma: float
adaptive_frequency_eps: float
adaptive_frequency_save_debug: bool
```

The direct source-code inference script is:

```text
scripts/inference/run_wan_adaptive_frequency.py
```

Example:

```bash
python -m scripts.inference.run_wan_adaptive_frequency \
    --model-path /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt-path assets/prompt.txt \
    --shift 8.0 \
    --af-timesteps \
    --af-scheduler unipc \
    --af-method soft_equal_snr \
    --af-soft-equal-snr-gamma 0.5 \
    --af-stats experiments/adaptive_frequency_stats/openvidhd/dataset_prior_frequency_stats.npz \
    --af-temporal-bands 4 \
    --af-spatial-bands 8 \
    --output-dir experiments/adaptive_frequency_inference/test
```

## Visualization Experiments

The main diagnostic script is:

```text
experiments/sigma_schedule_plot/plot_sigma_schedule.py
```

It visualizes:

1. Local sigma curves.
2. Local timestep curves.
3. Local timestep heatmaps.
4. Local log-SNR heatmaps.
5. Global SNR reconstructed from local sigmas versus the real global SNR.

It sweeps:

```text
gamma = 0.0, 0.1, ..., 1.0
shift = 1, 2, ..., 8
```

and writes one figure per shift:

```text
experiments/sigma_schedule_plot/soft_equalsnr_schedule_shift_<shift>.png
```

The global SNR diagnostic uses the same total signal/noise computation as the
runtime helper:

```text
SNR_from_local(k)
= sum_b N_b (1 - sigma_{b,k})^2 C_b
  / sum_b N_b sigma_{b,k}^2.
```

This plot is useful because soft EqualSNR can deform local schedules while not
exactly preserving the original global SNR for intermediate `gamma` values.
Empirically, `gamma=0` and `gamma=1` preserve global SNR closely, while middle
values can create a controlled mismatch.

## Current Limitations

This implementation is intentionally narrow.

- It is implemented for Wan-style video latents `[B, C, F, H, W]`.
- It requires single-process, single-latent-shard inference.
- It requires precomputed dataset-prior frequency power stats.
- The model still receives one scalar timestep, so the denoiser is not explicitly
  told about the per-frequency local timesteps.
- `equal_snr` and `soft_equal_snr` currently keep the original scalar model
  timesteps and only adapt the FFT update sigmas.
- Adaptive UniPC is a frequency-map adaptation of UniPC; it is not mathematically
  identical to the original global-trajectory UniPC derivation.
- The code currently uses simple radial temporal/spatial bins. It does not yet
  model anisotropic spatial directions or phase-specific behavior.

## Research Interpretation

The method can be viewed as a generalization of the global flow-shift schedule.
The usual shift changes the global sigma trajectory for all frequencies:

```text
sigma -> shift * sigma / (1 + (shift - 1) * sigma).
```

Adaptive frequency timesteps instead make the effective sigma trajectory depend
on the frequency power:

```text
sigma_k -> sigma_{b,k}.
```

The method keeps the pretrained model interface unchanged while testing whether
frequency-dependent scheduler dynamics can improve video details, motion, and
temporal stability.

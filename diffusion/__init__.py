# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

import numpy as np

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, DifferenceSpacedDiffusion_v0, DifferenceSpacedDiffusion_v1, MeanSpacedDiffusion_v0, SpatialDifferenceSpacedDiffusionV2, TemporalDifferenceSpacedDiffusionV2, space_timesteps
from .respace_v2 import SpacedGaussianDiffusionV2



def create_diffusion(
    timestep_respacing,
    name='gaussian_diffusion',
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    # learn_sigma=False,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    adaptive_frequency=False,
    adaptive_frequency_gamma=0.5,
    adaptive_frequency_learnable_gamma=False,
    adaptive_frequency_gamma_mode="scalar",
    adaptive_frequency_power_path=None,
    adaptive_frequency_power_exponent=2.0,
    adaptive_frequency_num_temporal_bands=None,
    adaptive_frequency_num_spatial_bands=None,
    equal_snr=False,
    equal_snr_power_path=None,
    equal_snr_power_scale=1.0,
    equal_snr_power_exponent=2.0,
    equal_snr_calibrate_schedule=False,
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if equal_snr and equal_snr_calibrate_schedule:
        betas = calibrate_equal_snr_betas(betas, equal_snr_power_path, equal_snr_power_scale)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    
    class_name = {
        'gaussian_diffusion': SpacedDiffusion,
        'difference_gaussian_diffusion_v0': DifferenceSpacedDiffusion_v0,
        'difference_gaussian_diffusion_v1': DifferenceSpacedDiffusion_v1,
        'mean_gaussian_diffusion_v0': MeanSpacedDiffusion_v0,
        'spatial_difference_gaussian_diffusion_v2': SpatialDifferenceSpacedDiffusionV2,
        'temporal_difference_gaussian_diffusion_v2': TemporalDifferenceSpacedDiffusionV2,

        'gaussian_diffusion_v2': SpacedGaussianDiffusionV2,
    }[name]
    return class_name(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        adaptive_frequency_kwargs=dict(
            enabled=adaptive_frequency,
            gamma=adaptive_frequency_gamma,
            learnable_gamma=adaptive_frequency_learnable_gamma,
            gamma_mode=adaptive_frequency_gamma_mode,
            num_train_timesteps=diffusion_steps,
            power_path=adaptive_frequency_power_path,
            power_exponent=adaptive_frequency_power_exponent,
            num_temporal_bands=adaptive_frequency_num_temporal_bands,
            num_spatial_bands=adaptive_frequency_num_spatial_bands,
        ),
        equal_snr_kwargs=dict(
            enabled=equal_snr,
            power_path=equal_snr_power_path,
            power_scale=equal_snr_power_scale,
            power_exponent=equal_snr_power_exponent,
        ),
        # rescale_timesteps=rescale_timesteps,
    )


def calibrate_equal_snr_betas(betas, power_path, power_scale):
    if not power_path:
        raise ValueError("equal_snr_calibrate_schedule requires equal_snr_power_path")
    with np.load(power_path) as data:
        if "channel_frequency_power_mean" in data:
            power = np.asarray(data["channel_frequency_power_mean"], dtype=np.float64)
        elif "frequency_power_mean" in data:
            power = np.asarray(data["frequency_power_mean"], dtype=np.float64)
        else:
            raise KeyError(f"{power_path} does not contain EqualSNR power statistics")

    mean_c = float(power.mean() * float(power_scale))
    if not np.isfinite(mean_c) or mean_c <= 0:
        raise ValueError(f"invalid EqualSNR mean covariance for calibration: {mean_c}")

    alpha_bar = np.cumprod(1.0 - np.asarray(betas, dtype=np.float64))
    ddpm_snr = alpha_bar / np.maximum(1.0 - alpha_bar, 1e-20)
    calibrated_snr = ddpm_snr * mean_c
    calibrated_alpha_bar = calibrated_snr / (1.0 + calibrated_snr)
    calibrated_alpha_bar = np.clip(calibrated_alpha_bar, 1e-12, 1.0 - 1e-12)

    calibrated_betas = np.empty_like(alpha_bar)
    calibrated_betas[0] = 1.0 - calibrated_alpha_bar[0]
    calibrated_betas[1:] = 1.0 - calibrated_alpha_bar[1:] / calibrated_alpha_bar[:-1]
    return np.clip(calibrated_betas, 1e-12, 0.999)

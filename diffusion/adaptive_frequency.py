import math
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def _logit(x):
    x = min(max(float(x), 1e-6), 1.0 - 1e-6)
    return math.log(x / (1.0 - x))


class AdaptiveFrequencyTimesteps:
    """
    Soft EqualSNR local timesteps for VP diffusion.

    The model still receives one scalar timestep. The noisy latent construction
    and DDIM update use per-frequency local alpha/sigma values in FFT space:

        sigma_b^2 = sigma^2 / (sigma^2 + alpha^2 * (P_bar / P_b)^gamma)

    gamma=0 recovers the original global schedule, while gamma=1 is hard
    EqualSNR. Gamma can be a fixed value or a learnable scalar constrained to
    [0, 1] by a sigmoid.
    """

    def __init__(
        self,
        *,
        enabled=False,
        gamma=0.5,
        learnable_gamma=False,
        gamma_mode="scalar",
        num_train_timesteps=1000,
        power_path=None,
        power_exponent=2.0,
        num_temporal_bands=None,
        num_spatial_bands=None,
        eps=1e-8,
    ):
        self.enabled = bool(enabled)
        self.learnable_gamma = bool(learnable_gamma)
        self.gamma_mode = str(gamma_mode)
        self.num_train_timesteps = int(num_train_timesteps)
        self.power_path = power_path
        self.power_exponent = power_exponent
        self.num_temporal_bands = num_temporal_bands
        self.num_spatial_bands = num_spatial_bands
        self.eps = float(eps)
        if self.gamma_mode == "global":
            self.gamma_mode = "scalar"
        if self.gamma_mode in {"per_bin", "frequency", "frequency_wise"}:
            self.gamma_mode = "frequency_bin"
        if self.gamma_mode not in {"scalar", "frequency_bin", "timestep", "data_dependent"}:
            raise ValueError(f"unknown adaptive frequency gamma_mode: {self.gamma_mode}")
        self._raw_gamma = None
        self._raw_gamma_bins = None
        self._raw_gamma_timesteps = None
        self._data_gamma_bias = None
        self._data_gamma_scale = None
        if self.learnable_gamma:
            raw_init = th.tensor(_logit(gamma), dtype=th.float32)
            if self.gamma_mode == "scalar":
                self._raw_gamma = nn.Parameter(raw_init)
            elif self.gamma_mode == "timestep":
                self._raw_gamma_timesteps = nn.Parameter(raw_init.repeat(self.num_train_timesteps))
            elif self.gamma_mode == "data_dependent":
                self._data_gamma_bias = nn.Parameter(raw_init)
                self._data_gamma_scale = nn.Parameter(th.tensor(0.0, dtype=th.float32))
        self._fixed_gamma = float(gamma)
        self._cache = {}
        self._pending_state_dict = None

    def parameters(self):
        params = []
        for param in (
            self._raw_gamma,
            self._raw_gamma_bins,
            self._raw_gamma_timesteps,
            self._data_gamma_bias,
            self._data_gamma_scale,
        ):
            if param is not None:
                params.append(param)
        return params

    def state_dict(self):
        state = {"gamma_mode": self.gamma_mode}
        if self._raw_gamma is not None:
            state["raw_gamma"] = self._raw_gamma.detach().cpu()
        if self._raw_gamma_bins is not None:
            state["raw_gamma_bins"] = self._raw_gamma_bins.detach().cpu()
        if self._raw_gamma_timesteps is not None:
            state["raw_gamma_timesteps"] = self._raw_gamma_timesteps.detach().cpu()
        if self._data_gamma_bias is not None:
            state["data_gamma_bias"] = self._data_gamma_bias.detach().cpu()
        if self._data_gamma_scale is not None:
            state["data_gamma_scale"] = self._data_gamma_scale.detach().cpu()
        return state

    def load_state_dict(self, state_dict):
        self._pending_state_dict = state_dict
        if self._raw_gamma is not None and "raw_gamma" in state_dict:
            with th.no_grad():
                self._raw_gamma.copy_(state_dict["raw_gamma"].to(self._raw_gamma.device))
        if self._raw_gamma_bins is not None and "raw_gamma_bins" in state_dict:
            with th.no_grad():
                self._raw_gamma_bins.copy_(state_dict["raw_gamma_bins"].to(self._raw_gamma_bins.device))
        if self._raw_gamma_timesteps is not None and "raw_gamma_timesteps" in state_dict:
            with th.no_grad():
                self._raw_gamma_timesteps.copy_(state_dict["raw_gamma_timesteps"].to(self._raw_gamma_timesteps.device))
        if self._data_gamma_bias is not None and "data_gamma_bias" in state_dict:
            with th.no_grad():
                self._data_gamma_bias.copy_(state_dict["data_gamma_bias"].to(self._data_gamma_bias.device))
        if self._data_gamma_scale is not None and "data_gamma_scale" in state_dict:
            with th.no_grad():
                self._data_gamma_scale.copy_(state_dict["data_gamma_scale"].to(self._data_gamma_scale.device))

    def initialize_for_shape(self, shape, device, dtype=th.float32):
        if not self.enabled:
            return
        power, _, bin_map, num_bins = self.power_and_global_mean(shape, device, dtype)
        self._maybe_init_frequency_gamma(power, device, bin_map=bin_map, num_bins=num_bins)

    def gamma(self, device=None, dtype=None):
        if self._raw_gamma is not None:
            value = th.sigmoid(self._raw_gamma)
        else:
            value = th.tensor(self._fixed_gamma, dtype=th.float32)
        if device is not None:
            value = value.to(device=device)
        if dtype is not None:
            value = value.to(dtype=dtype)
        return value

    def gamma_stats(self, device=None):
        values = []
        for param in (
            self._raw_gamma,
            self._raw_gamma_bins,
            self._raw_gamma_timesteps,
            self._data_gamma_bias,
        ):
            if param is not None:
                values.append(th.sigmoid(param.detach()).flatten())
        if not values:
            values = [self.gamma(device=device).detach().flatten()]
        values = th.cat([v.to(device=device) for v in values])
        fallback_gamma = float(th.sigmoid(th.tensor(_logit(self._fixed_gamma))).item())
        values = th.nan_to_num(values, nan=fallback_gamma, posinf=1.0, neginf=0.0)
        return {
            "mean": values.mean().item(),
            "min": values.min().item(),
            "max": values.max().item(),
            "std": values.std(unbiased=False).item() if values.numel() > 1 else 0.0,
        }

    def sanitize_parameters(self):
        fill = _logit(self._fixed_gamma)
        for param in self.parameters():
            with th.no_grad():
                finite = th.isfinite(param)
                if not finite.all():
                    param.masked_fill_(~finite, fill)

    def power_and_global_mean(self, shape, device, dtype):
        f, h, w = self._fhw_from_shape(shape)
        key = (f, h, w, str(device), str(dtype))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if self.power_path:
            power, p_bar, bin_map, num_bins = self._load_power(f, h, w)
        else:
            power = self._build_radial_power(f, h, w)
            p_bar = power.mean()
            bin_map = None
            num_bins = power.numel()

        power = power.to(device=device, dtype=dtype).clamp_min(self.eps)
        p_bar = th.as_tensor(p_bar, device=device, dtype=dtype).clamp_min(self.eps)
        if bin_map is not None:
            bin_map = bin_map.to(device=device)
        self._cache[key] = (power, p_bar, bin_map, int(num_bins))
        return power, p_bar, bin_map, int(num_bins)

    def _maybe_init_frequency_gamma(self, power, device, bin_map=None, num_bins=None):
        if not self.learnable_gamma or self.gamma_mode != "frequency_bin" or self._raw_gamma_bins is not None:
            return
        if bin_map is not None:
            if num_bins is None:
                num_bins = int(bin_map.max().item()) + 1
            raw_init = th.full((int(num_bins),), _logit(self._fixed_gamma), dtype=th.float32, device=device)
        else:
            raw_init = th.full_like(power, _logit(self._fixed_gamma), dtype=th.float32, device=device)
        self._raw_gamma_bins = nn.Parameter(raw_init)
        if self._pending_state_dict is not None and "raw_gamma_bins" in self._pending_state_dict:
            with th.no_grad():
                pending = self._pending_state_dict["raw_gamma_bins"].to(device=device)
                if pending.shape != self._raw_gamma_bins.shape:
                    raise ValueError(
                        "checkpoint raw_gamma_bins shape "
                        f"{tuple(pending.shape)} does not match expected {tuple(self._raw_gamma_bins.shape)}"
                    )
                self._raw_gamma_bins.copy_(pending)

    def _gamma_map(self, power, shape, device, dtype, t=None, reference=None, bin_map=None, num_bins=None):
        if not self.learnable_gamma:
            gamma = self.gamma(device=device, dtype=dtype)
        elif self.gamma_mode == "scalar":
            gamma = self.gamma(device=device, dtype=dtype)
        elif self.gamma_mode == "frequency_bin":
            self._maybe_init_frequency_gamma(power, device, bin_map=bin_map, num_bins=num_bins)
            raw = th.nan_to_num(self._raw_gamma_bins, nan=_logit(self._fixed_gamma))
            gamma = th.sigmoid(raw).to(device=device, dtype=dtype)
            if bin_map is not None:
                gamma = gamma[bin_map]
        elif self.gamma_mode == "timestep":
            if t is None:
                raw = th.nan_to_num(self._raw_gamma_timesteps[-1], nan=_logit(self._fixed_gamma))
                gamma = th.sigmoid(raw).to(device=device, dtype=dtype)
            else:
                t_idx = t.clamp(0, self.num_train_timesteps - 1).long()
                raw = self._raw_gamma_timesteps.to(device=device)[t_idx]
                raw = th.nan_to_num(raw, nan=_logit(self._fixed_gamma))
                gamma = th.sigmoid(raw).to(dtype=dtype)
        elif self.gamma_mode == "data_dependent":
            gamma = self._data_dependent_gamma(reference, device, dtype)
        else:
            raise NotImplementedError(self.gamma_mode)

        if gamma.ndim == 0:
            return gamma
        if gamma.shape == power.shape:
            return gamma
        if t is not None and gamma.shape == t.shape:
            return gamma.view(-1, *([1] * (len(shape) - 1)))
        if reference is not None and gamma.shape[:1] == reference.shape[:1]:
            return gamma.view(-1, *([1] * (len(shape) - 1)))
        return gamma

    def _data_dependent_gamma(self, reference, device, dtype):
        if reference is None:
            return th.sigmoid(self._data_gamma_bias).to(device=device, dtype=dtype)
        fft_dtype = th.float32 if reference.dtype in (th.float16, th.bfloat16) else reference.dtype
        ref_fft = th.fft.fftn(reference.detach().to(fft_dtype), dim=(-4, -2, -1), norm="ortho")
        energy = ref_fft.abs().square().mean(dim=tuple(range(1, ref_fft.ndim))).clamp_min(self.eps)
        feature = th.log(energy)
        feature = feature - feature.mean()
        feature = feature / feature.std(unbiased=False).clamp_min(self.eps)
        raw = self._data_gamma_bias.to(device=reference.device) + self._data_gamma_scale.to(device=reference.device) * feature
        return th.sigmoid(raw).to(device=device, dtype=dtype)

    def local_alpha_sigma(self, alpha, sigma, shape, device, dtype, t=None, reference=None):
        power, p_bar, bin_map, num_bins = self.power_and_global_mean(shape, device, dtype)
        gamma = self._gamma_map(
            power,
            shape,
            device,
            dtype,
            t=t,
            reference=reference,
            bin_map=bin_map,
            num_bins=num_bins,
        )
        base_ratio = (p_bar / power).clamp_min(self.eps)
        if gamma.shape == power.shape:
            ratio = base_ratio.pow(gamma)
        else:
            ratio = self._frequency_view(base_ratio, shape).pow(gamma)

        while alpha.ndim < len(shape):
            alpha = alpha[..., None]
            sigma = sigma[..., None]

        if ratio.shape == power.shape:
            ratio = self._frequency_view(ratio, shape)

        sigma2 = sigma.square()
        alpha2 = alpha.square()
        local_sigma2 = sigma2 / (sigma2 + alpha2 * ratio).clamp_min(self.eps)
        local_sigma = local_sigma2.clamp(0.0, 1.0).sqrt()
        local_alpha = (1.0 - local_sigma2).clamp(self.eps, 1.0).sqrt()
        return local_alpha, local_sigma

    def q_sample(self, x_start, t, noise, alpha, sigma):
        if not self.enabled:
            return alpha * x_start + sigma * noise

        dtype = x_start.dtype
        fft_dtype = th.float32 if dtype in (th.float16, th.bfloat16) else dtype
        local_alpha, local_sigma = self.local_alpha_sigma(
            alpha.to(fft_dtype),
            sigma.to(fft_dtype),
            x_start.shape,
            x_start.device,
            fft_dtype,
            t=t,
            reference=x_start,
        )
        x_fft = th.fft.fftn(x_start.to(fft_dtype), dim=(-4, -2, -1), norm="ortho")
        n_fft = th.fft.fftn(noise.to(fft_dtype), dim=(-4, -2, -1), norm="ortho")
        out = th.fft.ifftn(local_alpha * x_fft + local_sigma * n_fft, dim=(-4, -2, -1), norm="ortho").real
        return out.to(dtype)

    def predict_xstart_from_eps(self, x_t, eps, alpha, sigma, t=None):
        if not self.enabled:
            return (x_t - sigma * eps) / alpha

        dtype = x_t.dtype
        fft_dtype = th.float32 if dtype in (th.float16, th.bfloat16) else dtype
        local_alpha, local_sigma = self.local_alpha_sigma(
            alpha.to(fft_dtype),
            sigma.to(fft_dtype),
            x_t.shape,
            x_t.device,
            fft_dtype,
            t=t,
            reference=x_t,
        )
        x_fft = th.fft.fftn(x_t.to(fft_dtype), dim=(-4, -2, -1), norm="ortho")
        e_fft = th.fft.fftn(eps.to(fft_dtype), dim=(-4, -2, -1), norm="ortho")
        out = th.fft.ifftn((x_fft - local_sigma * e_fft) / local_alpha.clamp_min(self.eps), dim=(-4, -2, -1), norm="ortho").real
        return out.to(dtype)

    def predict_eps_from_xstart(self, x_t, pred_xstart, alpha, sigma, t=None):
        if not self.enabled:
            return (x_t - alpha * pred_xstart) / sigma

        dtype = x_t.dtype
        fft_dtype = th.float32 if dtype in (th.float16, th.bfloat16) else dtype
        local_alpha, local_sigma = self.local_alpha_sigma(
            alpha.to(fft_dtype),
            sigma.to(fft_dtype),
            x_t.shape,
            x_t.device,
            fft_dtype,
            t=t,
            reference=x_t,
        )
        x_fft = th.fft.fftn(x_t.to(fft_dtype), dim=(-4, -2, -1), norm="ortho")
        x0_fft = th.fft.fftn(pred_xstart.to(fft_dtype), dim=(-4, -2, -1), norm="ortho")
        out = th.fft.ifftn((x_fft - local_alpha * x0_fft) / local_sigma.clamp_min(self.eps), dim=(-4, -2, -1), norm="ortho").real
        return out.to(dtype)

    def ddim_step(self, pred_xstart, eps, alpha_prev, sigma_prev, t=None):
        if not self.enabled:
            return alpha_prev * pred_xstart + sigma_prev * eps

        dtype = pred_xstart.dtype
        fft_dtype = th.float32 if dtype in (th.float16, th.bfloat16) else dtype
        local_alpha_prev, local_sigma_prev = self.local_alpha_sigma(
            alpha_prev.to(fft_dtype),
            sigma_prev.to(fft_dtype),
            pred_xstart.shape,
            pred_xstart.device,
            fft_dtype,
            t=t,
            reference=pred_xstart,
        )
        x0_fft = th.fft.fftn(pred_xstart.to(fft_dtype), dim=(-4, -2, -1), norm="ortho")
        e_fft = th.fft.fftn(eps.to(fft_dtype), dim=(-4, -2, -1), norm="ortho")
        out = th.fft.ifftn(local_alpha_prev * x0_fft + local_sigma_prev * e_fft, dim=(-4, -2, -1), norm="ortho").real
        return out.to(dtype)

    def _load_power(self, f, h, w):
        path = Path(self.power_path)
        data = np.load(path)
        # if all(k in data for k in ("C_mean", "num_frequencies", "temporal_edges", "spatial_edges")):
        #     c_mean = np.asarray(data["C_mean"], dtype=np.float64)
        #     counts = np.asarray(data["num_frequencies"], dtype=np.float64)
        #     temporal_edges = np.asarray(data["temporal_edges"], dtype=np.float64)
        #     spatial_edges = np.asarray(data["spatial_edges"], dtype=np.float64)
        # elif "frequency_power_mean" in data:
        if "frequency_power_mean" in data:
            if self.num_temporal_bands is None or self.num_spatial_bands is None:
                power = th.from_numpy(np.asarray(data["frequency_power_mean"], dtype=np.float32))
                if tuple(power.shape) != (f, h, w):
                    power = F.interpolate(
                        power[None, None],
                        size=(f, h, w),
                        mode="trilinear",
                        align_corners=False,
                    )[0, 0]
                return power, power.mean(), None, power.numel()
            c_mean, counts, temporal_edges, spatial_edges = self._rebin_frequency_power(
                np.asarray(data["frequency_power_mean"], dtype=np.float64),
                int(self.num_temporal_bands),
                int(self.num_spatial_bands),
            )
        else:
            raise KeyError(f"{path} does not contain adaptive frequency statistics")

        self._validate_banded_stats(c_mean, counts, temporal_edges, spatial_edges)
        bin_map = self._build_frequency_bin_map(f, h, w, temporal_edges, spatial_edges)
        flat_power = th.from_numpy(c_mean.reshape(-1).astype(np.float32))
        power = flat_power[bin_map]
        p_bar = float((c_mean.reshape(-1) * counts.reshape(-1)).sum() / counts.sum())
        return power, p_bar, bin_map, c_mean.size

    def _rebin_frequency_power(self, frequency_power_mean, num_temporal_bands, num_spatial_bands):
        if frequency_power_mean.ndim != 3:
            raise ValueError(f"frequency_power_mean must be [F,H,W], got {frequency_power_mean.shape}")
        if num_temporal_bands <= 0 or num_spatial_bands <= 0:
            raise ValueError("num_temporal_bands and num_spatial_bands must be positive")
        if not np.all(np.isfinite(frequency_power_mean)) or np.any(frequency_power_mean <= self.eps):
            raise ValueError("frequency_power_mean must contain finite positive values")

        temporal_radius, spatial_radius = self._frequency_grids_np(*frequency_power_mean.shape)
        temporal_edges = np.linspace(0.0, float(temporal_radius.max()) + 1e-6, num_temporal_bands + 1)
        spatial_edges = np.linspace(0.0, float(spatial_radius.max()) + 1e-6, num_spatial_bands + 1)
        c_mean = np.zeros((num_temporal_bands, num_spatial_bands), dtype=np.float64)
        counts = np.zeros_like(c_mean)
        for temporal_idx in range(num_temporal_bands):
            t_lo = temporal_edges[temporal_idx]
            t_hi = temporal_edges[temporal_idx + 1]
            temporal_mask = (temporal_radius >= t_lo) & (
                temporal_radius <= t_hi if temporal_idx == num_temporal_bands - 1 else temporal_radius < t_hi
            )
            for spatial_idx in range(num_spatial_bands):
                s_lo = spatial_edges[spatial_idx]
                s_hi = spatial_edges[spatial_idx + 1]
                spatial_mask = (spatial_radius >= s_lo) & (
                    spatial_radius <= s_hi if spatial_idx == num_spatial_bands - 1 else spatial_radius < s_hi
                )
                values = frequency_power_mean[temporal_mask & spatial_mask]
                if values.size == 0:
                    raise ValueError(f"empty adaptive-frequency bin ({temporal_idx}, {spatial_idx})")
                c_mean[temporal_idx, spatial_idx] = max(float(values.mean()), self.eps)
                counts[temporal_idx, spatial_idx] = float(values.size)
        return c_mean, counts, temporal_edges, spatial_edges

    def _build_frequency_bin_map(self, f, h, w, temporal_edges, spatial_edges):
        temporal_edges = th.as_tensor(temporal_edges, dtype=th.float32)
        spatial_edges = th.as_tensor(spatial_edges, dtype=th.float32)
        ft = th.fft.fftfreq(f) * f
        fy = th.fft.fftfreq(h) * h
        fx = th.fft.fftfreq(w) * w
        temporal, yy, xx = th.meshgrid(ft.abs(), fy, fx, indexing="ij")
        spatial = th.sqrt(yy.square() + xx.square())
        temporal_band = th.bucketize(temporal.reshape(-1), temporal_edges[1:-1]).reshape(f, h, w)
        spatial_band = th.bucketize(spatial.reshape(-1), spatial_edges[1:-1]).reshape(f, h, w)
        temporal_band = th.clamp(temporal_band, max=temporal_edges.numel() - 2)
        spatial_band = th.clamp(spatial_band, max=spatial_edges.numel() - 2)
        return (temporal_band * (spatial_edges.numel() - 1) + spatial_band).long()

    def _validate_banded_stats(self, c_mean, counts, temporal_edges, spatial_edges):
        if c_mean.ndim != 2:
            raise ValueError(f"C_mean must be 2D, got {c_mean.shape}")
        if counts.shape != c_mean.shape:
            raise ValueError("num_frequencies must match C_mean shape")
        if temporal_edges.shape != (c_mean.shape[0] + 1,):
            raise ValueError("temporal_edges length must equal temporal bands + 1")
        if spatial_edges.shape != (c_mean.shape[1] + 1,):
            raise ValueError("spatial_edges length must equal spatial bands + 1")
        if not np.all(np.isfinite(c_mean)) or np.any(c_mean <= self.eps):
            raise ValueError("C_mean must contain finite positive values")
        if not np.all(np.isfinite(counts)) or np.any(counts <= 0):
            raise ValueError("num_frequencies must contain finite positive values")
        if not np.all(np.isfinite(temporal_edges)) or not np.all(temporal_edges[1:] > temporal_edges[:-1]):
            raise ValueError("temporal_edges must be finite and strictly increasing")
        if not np.all(np.isfinite(spatial_edges)) or not np.all(spatial_edges[1:] > spatial_edges[:-1]):
            raise ValueError("spatial_edges must be finite and strictly increasing")

    @staticmethod
    def _frequency_grids_np(frames, height, width):
        ft = np.fft.fftfreq(frames) * frames
        fy = np.fft.fftfreq(height) * height
        fx = np.fft.fftfreq(width) * width
        tt, yy, xx = np.meshgrid(ft, fy, fx, indexing="ij")
        return np.abs(tt), np.sqrt(yy**2 + xx**2)

    def _resize_power(self, power, f, h, w):
        if tuple(power.shape) != (f, h, w):
            power = F.interpolate(
                power[None, None],
                size=(f, h, w),
                mode="trilinear",
                align_corners=False,
            )[0, 0]
        return power

    def _build_radial_power(self, f, h, w):
        ft = th.fft.fftfreq(f) * f
        fy = th.fft.fftfreq(h) * h
        fx = th.fft.fftfreq(w) * w
        tt, yy, xx = th.meshgrid(ft, fy, fx, indexing="ij")
        radius2 = tt.square() + yy.square() + xx.square()
        power = (1.0 + radius2).pow(-0.5 * self.power_exponent)
        return power / power.mean()

    @staticmethod
    def _frequency_view(tensor, shape):
        view_shape = [1] * len(shape)
        view_shape[-4] = tensor.shape[0]
        view_shape[-2] = tensor.shape[1]
        view_shape[-1] = tensor.shape[2]
        return tensor.view(view_shape)

    @staticmethod
    def _fhw_from_shape(shape):
        if len(shape) < 5:
            raise ValueError(f"adaptive frequency expects video latents, got shape {shape}")
        return int(shape[-4]), int(shape[-2]), int(shape[-1])


class EqualSNRFourier(AdaptiveFrequencyTimesteps):
    """
    Hard EqualSNR forward process from "A Fourier Space Perspective on
    Diffusion Models".

    The process uses y_t = sqrt(alpha_t) y_0 + sqrt(1 - alpha_t) eps_C in
    Fourier space, with eps_C having diagonal covariance C. The loss is
    || C^{-1/2} (y_0 - yhat_0) ||^2. C is loaded from the same dense
    frequency_power_mean statistics used by the adaptive-frequency runs.
    """

    def __init__(
        self,
        *,
        enabled=False,
        power_path=None,
        power_scale=1.0,
        power_exponent=2.0,
        eps=1e-8,
    ):
        super().__init__(
            enabled=enabled,
            gamma=1.0,
            learnable_gamma=False,
            power_path=power_path,
            power_exponent=power_exponent,
            num_temporal_bands=None,
            num_spatial_bands=None,
            eps=eps,
        )
        self.power_scale = float(power_scale)

    def parameters(self):
        return []

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return

    def power_and_global_mean(self, shape, device, dtype):
        power, p_bar, bin_map, num_bins = self._equal_snr_power_and_global_mean(shape, device, dtype)
        scale = th.as_tensor(self.power_scale, device=device, dtype=dtype).clamp_min(self.eps)
        return power * scale, p_bar * scale, bin_map, num_bins

    def _power_view(self, shape, device, dtype):
        power, _, _, _ = self.power_and_global_mean(shape, device, dtype)
        view_shape = [1] * len(shape)
        if power.ndim == 4:
            view_shape[-4] = power.shape[0]
            view_shape[-3] = power.shape[1]
            view_shape[-2] = power.shape[2]
            view_shape[-1] = power.shape[3]
        else:
            view_shape[-4] = power.shape[0]
            view_shape[-2] = power.shape[1]
            view_shape[-1] = power.shape[2]
        return power.view(view_shape).clamp_min(self.eps)

    def _equal_snr_power_and_global_mean(self, shape, device, dtype):
        if not self.power_path:
            return super().power_and_global_mean(shape, device, dtype)

        f, h, w = self._fhw_from_shape(shape)
        c = int(shape[-3])
        key = ("equal_snr", c, f, h, w, str(device), str(dtype))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        path = Path(self.power_path)
        data = np.load(path)
        if "channel_frequency_power_mean" not in data:
            return super().power_and_global_mean(shape, device, dtype)

        power = th.from_numpy(np.asarray(data["channel_frequency_power_mean"], dtype=np.float32))
        if power.ndim != 4:
            raise ValueError(f"channel_frequency_power_mean must be [C,F,H,W], got {tuple(power.shape)}")
        if power.shape[0] != c:
            raise ValueError(f"channel_frequency_power_mean has {power.shape[0]} channels, expected {c}")
        if tuple(power.shape[1:]) != (f, h, w):
            power = F.interpolate(
                power[None],
                size=(f, h, w),
                mode="trilinear",
                align_corners=False,
            )[0]
        power = power.permute(1, 0, 2, 3).contiguous()
        power = power.to(device=device, dtype=dtype).clamp_min(self.eps)
        p_bar = power.mean().clamp_min(self.eps)
        cached = (power, p_bar, None, power.numel())
        self._cache[key] = cached
        return cached

    def colored_noise(self, noise):
        if not self.enabled:
            return noise
        dtype = noise.dtype
        fft_dtype = th.float32 if dtype in (th.float16, th.bfloat16) else dtype
        power = self._power_view(noise.shape, noise.device, fft_dtype)
        noise_fft = th.fft.fftn(noise.to(fft_dtype), dim=(-4, -2, -1), norm="ortho")
        out = th.fft.ifftn(power.sqrt() * noise_fft, dim=(-4, -2, -1), norm="ortho").real
        return out.to(dtype)

    def q_sample(self, x_start, t, noise, alpha, sigma):
        if not self.enabled:
            return alpha * x_start + sigma * noise
        colored_noise = self.colored_noise(noise)
        return alpha * x_start + sigma * colored_noise

    def predict_eps_from_xstart(self, x_t, pred_xstart, alpha, sigma):
        if not self.enabled:
            return (x_t - alpha * pred_xstart) / sigma
        # This returns eps_C in data space, not the underlying white eps.
        return (x_t - alpha * pred_xstart) / sigma.clamp_min(self.eps)

    def ddim_step(self, pred_xstart, eps_c, alpha_prev, sigma_prev):
        if not self.enabled:
            return alpha_prev * pred_xstart + sigma_prev * eps_c
        return alpha_prev * pred_xstart + sigma_prev * eps_c

    def xstart_loss(self, x_start, pred_xstart):
        if not self.enabled:
            return None
        dtype = x_start.dtype
        fft_dtype = th.float32 if dtype in (th.float16, th.bfloat16) else dtype
        power = self._power_view(x_start.shape, x_start.device, fft_dtype)
        target_fft = th.fft.fftn(x_start.to(fft_dtype), dim=(-4, -2, -1), norm="ortho")
        pred_fft = th.fft.fftn(pred_xstart.to(fft_dtype), dim=(-4, -2, -1), norm="ortho")
        weighted = (target_fft - pred_fft).abs().square() / power
        return weighted.flatten(1).mean(dim=1).to(dtype)

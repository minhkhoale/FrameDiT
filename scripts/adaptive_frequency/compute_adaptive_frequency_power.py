#!/usr/bin/env python3
"""Compute adaptive-frequency power statistics for latent video datasets.

Preferred input is latent video tensors shaped [C, F, H, W] or
[B, C, F, H, W]. The output .npz contains both the dense frequency prior used
by adaptive schedulers and pre-binned statistics compatible with
diffusion.adaptive_frequency:

    frequency_power_mean[F, H, W]
    shape_fhw[3]
    C_mean[temporal_band, spatial_band]
    num_frequencies[temporal_band, spatial_band]
    temporal_edges
    spatial_edges
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


PREFERRED_KEYS = ("latents", "latent", "video_latent", "samples", "video")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--latent", action="append", type=Path, help="Latent .pt/.pth/.npz file. Can be repeated.")
    inputs.add_argument("--latent-dir", type=Path, help="Directory containing latent .pt/.pth/.npz files.")
    inputs.add_argument("--video", action="append", type=Path, help="RGB video file for debugging. Can be repeated.")
    inputs.add_argument("--video-list", type=Path, help="Text file with one RGB video path per line.")
    parser.add_argument("--output-dir", type=Path, default=Path("results_adaptive_schedule/frequency_power"))
    parser.add_argument("--output-name", type=str, default="adaptive_frequency_power")
    parser.add_argument("--latent-key", type=str, default=None, help="Array key for .npz or dict-like .pt latent files.")
    parser.add_argument(
        "--latent-format",
        choices=("auto", "array", "ucf101_gaussian"),
        default="auto",
        help=(
            "How to interpret latent files. ucf101_gaussian matches datasets/ucf101_latent_datasets.py: "
            "torch.load posterior parameters, DiagonalGaussianDistribution, then [F,C,H,W]."
        ),
    )
    parser.add_argument(
        "--posterior-stat",
        choices=("sample", "mean"),
        default="sample",
        help="For ucf101_gaussian latents, use posterior sample to match the dataset or posterior mean for deterministic stats.",
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "bcfhw", "bfchw", "cfhw", "fchw", "bfhwc", "fhwc"),
        default="auto",
        help="Input tensor layout. Auto handles common latent and RGB video layouts.",
    )
    parser.add_argument(
        "--num-frames",
        "--frames",
        dest="num_frames",
        type=int,
        default=None,
        help="Number of frames to sample per latent/video. For latents, this should match configs.num_frames.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Temporal crop interval used for latent frame sampling, matching configs.frame_interval.",
    )
    parser.add_argument(
        "--temporal-sample",
        choices=("random", "center", "uniform"),
        default="random",
        help=(
            "How to choose latent frames. random matches UCF101Latent's random crop; "
            "center is deterministic; uniform spans the whole latent."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for temporal sampling and posterior sampling.")
    parser.add_argument("--resolution", type=int, default=None, help="Square resolution for RGB video fallback.")
    parser.add_argument("--num-temporal-bands", type=int, default=4)
    parser.add_argument("--num-spatial-bands", type=int, default=6)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true", help="Skip writing the PNG heatmap.")
    return parser.parse_args()


def collect_input_paths(args: argparse.Namespace) -> tuple[list[Path], str]:
    if args.latent:
        return args.latent, "latent"
    if args.latent_dir:
        paths = sorted(
            path
            for path in args.latent_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".npz"}
        )
        return paths, "latent"
    if args.video:
        return args.video, "rgb_video"

    assert args.video_list is not None
    paths = []
    for line in args.video_list.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            paths.append(Path(line))
    return paths, "rgb_video"


def load_latent(path: Path, key: str | None, layout: str, latent_format: str, posterior_stat: str) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as payload:
            array_key = key or first_array_key(payload.keys())
            array = payload[array_key]
    elif path.suffix in {".pt", ".pth"}:
        import torch

        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            if should_decode_ucf101_gaussian(payload, latent_format):
                array = sample_ucf101_gaussian(payload, posterior_stat).detach().cpu().numpy()
                return normalize_video_array(array, "fchw" if layout == "auto" else layout)
            array = payload.detach().cpu().numpy()
        elif isinstance(payload, dict):
            array_key = key or first_tensor_key(payload, torch)
            value = payload[array_key]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            if should_decode_ucf101_gaussian(value, latent_format):
                array = sample_ucf101_gaussian(value, posterior_stat).detach().cpu().numpy()
                return normalize_video_array(array, "fchw" if layout == "auto" else layout)
            array = value.detach().cpu().numpy()
        else:
            array = torch.as_tensor(payload).detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported latent file suffix: {path}")
    return normalize_video_array(array, layout)


def should_decode_ucf101_gaussian(tensor, latent_format: str) -> bool:
    if latent_format == "ucf101_gaussian":
        return True
    if latent_format == "array":
        return False

    # UCF101Latent stores DiagonalGaussianDistribution parameters as
    # [frames, 2 * latent_channels, height, width]. Sampled latents are
    # [frames, latent_channels, height, width].
    return tensor.ndim == 4 and tensor.shape[1] % 2 == 0 and tensor.shape[1] > 4


def sample_ucf101_gaussian(parameters, posterior_stat: str):
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    gaussian_dist = DiagonalGaussianDistribution(parameters=parameters)
    if posterior_stat == "mean":
        return gaussian_dist.mean
    return gaussian_dist.sample()


def first_array_key(keys: Iterable[str]) -> str:
    keys = list(keys)
    if not keys:
        raise ValueError("No arrays found in npz file")
    for preferred in PREFERRED_KEYS:
        if preferred in keys:
            return preferred
    return keys[0]


def first_tensor_key(payload: dict, torch_module) -> str:
    for preferred in PREFERRED_KEYS:
        if preferred in payload:
            return preferred
    for key, value in payload.items():
        if isinstance(value, torch_module.Tensor):
            return key
    raise ValueError("No tensor values found in latent checkpoint dictionary")


def normalize_video_array(array: np.ndarray, layout: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if layout != "auto":
        return normalize_known_layout(array, layout)

    if array.ndim == 5:
        if array.shape[1] <= 16:
            return array
        if array.shape[2] <= 16:
            return np.moveaxis(array, 2, 1)
        if array.shape[-1] in {1, 3, 4}:
            return np.moveaxis(array, -1, 1)
    elif array.ndim == 4:
        if array.shape[0] <= 16:
            return array[None, ...]
        if array.shape[1] <= 16:
            return np.moveaxis(array, 1, 0)[None, ...]
        if array.shape[-1] in {1, 3, 4}:
            return np.moveaxis(array, -1, 0)[None, ...]

    raise ValueError(
        "Could not infer layout. Expected one of [B,C,F,H,W], [B,F,C,H,W], "
        f"[C,F,H,W], [F,C,H,W], [B,F,H,W,C], or [F,H,W,C], got {array.shape}. "
        "Pass --layout explicitly if needed."
    )


def normalize_known_layout(array: np.ndarray, layout: str) -> np.ndarray:
    if layout == "bcfhw":
        if array.ndim != 5:
            raise ValueError(f"--layout bcfhw expects 5 dims, got {array.shape}")
        return array
    if layout == "bfchw":
        if array.ndim != 5:
            raise ValueError(f"--layout bfchw expects 5 dims, got {array.shape}")
        return np.moveaxis(array, 2, 1)
    if layout == "cfhw":
        if array.ndim != 4:
            raise ValueError(f"--layout cfhw expects 4 dims, got {array.shape}")
        return array[None, ...]
    if layout == "fchw":
        if array.ndim != 4:
            raise ValueError(f"--layout fchw expects 4 dims, got {array.shape}")
        return np.moveaxis(array, 1, 0)[None, ...]
    if layout == "bfhwc":
        if array.ndim != 5:
            raise ValueError(f"--layout bfhwc expects 5 dims, got {array.shape}")
        return np.moveaxis(array, -1, 1)
    if layout == "fhwc":
        if array.ndim != 4:
            raise ValueError(f"--layout fhwc expects 4 dims, got {array.shape}")
        return np.moveaxis(array, -1, 0)[None, ...]
    raise ValueError(f"Unknown layout: {layout}")


def load_video(path: Path, frames: int | None, resolution: int | None) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("RGB video input requires opencv-python. Use latent input or install cv2.") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = frames or max(frame_count, 1)
    indices = np.linspace(0, max(frame_count - 1, 0), target_frames).round().astype(int).tolist()

    loaded = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = center_crop_square(frame)
        if resolution is not None:
            frame = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)
        loaded.append(frame.astype(np.float32) / 127.5 - 1.0)
    capture.release()

    if not loaded:
        raise RuntimeError(f"No frames decoded from video: {path}")
    while len(loaded) < target_frames:
        loaded.append(loaded[-1].copy())
    return normalize_video_array(np.stack(loaded[:target_frames], axis=0), "fhwc")


def sample_frames_bcfhw(
    array: np.ndarray,
    num_frames: int | None,
    frame_interval: int,
    sample_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_frames is None:
        return array
    if num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if frame_interval <= 0:
        raise ValueError("--frame-interval must be positive")

    total_frames = int(array.shape[2])
    if total_frames < num_frames:
        raise ValueError(f"Cannot sample {num_frames} frames from latent with only {total_frames} frames")

    if sample_mode == "uniform":
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        crop_size = num_frames * frame_interval
        if total_frames < crop_size:
            raise ValueError(
                f"Cannot sample {num_frames} frames with frame_interval={frame_interval} "
                f"from latent with only {total_frames} frames"
            )
        if sample_mode == "center":
            start = (total_frames - crop_size) // 2
        elif sample_mode == "random":
            start = int(rng.integers(0, total_frames - crop_size + 1))
        else:
            raise ValueError(f"Unknown temporal sample mode: {sample_mode}")
        end = start + crop_size
        frame_indices = np.linspace(start, end - 1, num_frames, dtype=int)

    return array[:, :, frame_indices, :, :]


def center_crop_square(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    return frame[y0 : y0 + side, x0 : x0 + side]


def fft_power_bcfhw(video_bcfhw: np.ndarray) -> np.ndarray:
    coeffs = np.fft.fftn(video_bcfhw, axes=(-3, -2, -1), norm="ortho")
    return np.mean(np.abs(coeffs) ** 2, axis=(0, 1)).astype(np.float64)


def frequency_grids(frames: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    ft = np.fft.fftfreq(frames) * frames
    fy = np.fft.fftfreq(height) * height
    fx = np.fft.fftfreq(width) * width
    tt, yy, xx = np.meshgrid(ft, fy, fx, indexing="ij")
    return np.abs(tt), np.sqrt(yy**2 + xx**2)


def band_edges(values: np.ndarray, num_bands: int) -> np.ndarray:
    if num_bands <= 0:
        raise ValueError("num_bands must be positive")
    return np.linspace(0.0, float(values.max()) + 1e-6, num_bands + 1, dtype=np.float64)


def compute_banded_stats(
    frequency_power_mean: np.ndarray,
    num_temporal_bands: int,
    num_spatial_bands: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    temporal_radius, spatial_radius = frequency_grids(*frequency_power_mean.shape)
    temporal_edges = band_edges(temporal_radius, num_temporal_bands)
    spatial_edges = band_edges(spatial_radius, num_spatial_bands)
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
                raise ValueError(f"Empty frequency bin at temporal={temporal_idx}, spatial={spatial_idx}")
            c_mean[temporal_idx, spatial_idx] = max(float(values.mean()), eps)
            counts[temporal_idx, spatial_idx] = float(values.size)

    return c_mean, counts, temporal_edges, spatial_edges


def validate_stats(
    frequency_power_mean: np.ndarray,
    c_mean: np.ndarray,
    counts: np.ndarray,
    temporal_edges: np.ndarray,
    spatial_edges: np.ndarray,
    eps: float,
) -> None:
    if frequency_power_mean.ndim != 3:
        raise ValueError(f"frequency_power_mean must be [F,H,W], got {frequency_power_mean.shape}")
    if not np.all(np.isfinite(frequency_power_mean)) or np.any(frequency_power_mean <= 0):
        raise ValueError("frequency_power_mean must contain finite positive values")
    if c_mean.ndim != 2:
        raise ValueError(f"C_mean must be 2D, got {c_mean.shape}")
    if counts.shape != c_mean.shape:
        raise ValueError("num_frequencies must match C_mean shape")
    if temporal_edges.shape != (c_mean.shape[0] + 1,):
        raise ValueError("temporal_edges length must equal temporal bands + 1")
    if spatial_edges.shape != (c_mean.shape[1] + 1,):
        raise ValueError("spatial_edges length must equal spatial bands + 1")
    if not np.all(np.isfinite(c_mean)) or np.any(c_mean <= eps):
        raise ValueError("C_mean must contain finite positive values")
    if not np.all(np.isfinite(counts)) or np.any(counts <= 0):
        raise ValueError("num_frequencies must contain finite positive values")
    if not np.all(np.isfinite(temporal_edges)) or not np.all(temporal_edges[1:] > temporal_edges[:-1]):
        raise ValueError("temporal_edges must be finite and strictly increasing")
    if not np.all(np.isfinite(spatial_edges)) or not np.all(spatial_edges[1:] > spatial_edges[:-1]):
        raise ValueError("spatial_edges must be finite and strictly increasing")


def write_csv(path: Path, c_mean: np.ndarray, counts: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["temporal_band", "spatial_band", "C_mean", "num_frequencies"])
        writer.writeheader()
        for temporal_idx in range(c_mean.shape[0]):
            for spatial_idx in range(c_mean.shape[1]):
                writer.writerow(
                    {
                        "temporal_band": temporal_idx,
                        "spatial_band": spatial_idx,
                        "C_mean": float(c_mean[temporal_idx, spatial_idx]),
                        "num_frequencies": int(counts[temporal_idx, spatial_idx]),
                    }
                )


def plot_power(path: Path, c_mean: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping PNG heatmap")
        return

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    im = ax.imshow(np.log10(np.maximum(c_mean, 1e-12)), origin="lower", aspect="auto", cmap="magma")
    ax.set_title("adaptive frequency power")
    ax.set_xlabel("spatial frequency band")
    ax.set_ylabel("temporal frequency band")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10 C_mean")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def iter_progress(items):
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items)


def main() -> None:
    args = parse_args()
    if args.eps <= 0:
        raise ValueError("--eps must be positive")

    paths, input_kind = collect_input_paths(args)
    if args.max_items is not None:
        paths = paths[: args.max_items]
    if not paths:
        raise ValueError("No input files found")

    rng = np.random.default_rng(args.seed)
    power_sum = None
    shape_fhw = None
    processed = 0
    skipped = []
    for path in iter_progress(paths):
        try:
            array = (
                load_latent(path, args.latent_key, args.layout, args.latent_format, args.posterior_stat)
                if input_kind == "latent"
                else load_video(path, args.num_frames, args.resolution)
            )
            if input_kind == "latent":
                array = sample_frames_bcfhw(array, args.num_frames, args.frame_interval, args.temporal_sample, rng)
            cur_shape = tuple(int(dim) for dim in array.shape[-3:])
            if shape_fhw is None:
                shape_fhw = cur_shape
                power_sum = np.zeros(shape_fhw, dtype=np.float64)
            elif cur_shape != shape_fhw:
                raise ValueError(f"expected [F,H,W]={shape_fhw}, got {cur_shape}")

            assert power_sum is not None
            power_sum += fft_power_bcfhw(array)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            skipped.append({"path": str(path), "reason": reason})
            print(f"Skipping {path}: {reason}")
            continue
        processed += 1

    if power_sum is None or shape_fhw is None or processed == 0:
        raise ValueError(f"No valid inputs were processed; skipped {len(skipped)} files")
    frequency_power_mean = np.maximum(power_sum / max(processed, 1), args.eps)
    c_mean, counts, temporal_edges, spatial_edges = compute_banded_stats(
        frequency_power_mean,
        args.num_temporal_bands,
        args.num_spatial_bands,
        args.eps,
    )
    validate_stats(frequency_power_mean, c_mean, counts, temporal_edges, spatial_edges, args.eps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / f"{args.output_name}.npz"
    csv_path = args.output_dir / f"{args.output_name}.csv"
    json_path = args.output_dir / f"{args.output_name}.json"
    png_path = args.output_dir / f"{args.output_name}.png"

    np.savez_compressed(
        npz_path,
        frequency_power_mean=frequency_power_mean,
        shape_fhw=np.asarray(shape_fhw, dtype=np.int64),
        C_mean=c_mean,
        num_frequencies=counts,
        temporal_edges=temporal_edges,
        spatial_edges=spatial_edges,
    )
    write_csv(csv_path, c_mean, counts)
    if not args.no_plot:
        plot_power(png_path, c_mean)

    metadata = {
        "input_kind": input_kind,
        "latent_format": args.latent_format if input_kind == "latent" else None,
        "posterior_stat": args.posterior_stat if input_kind == "latent" else None,
        "num_items": processed,
        "num_skipped": len(skipped),
        "skipped": skipped,
        "shape_fhw": list(shape_fhw),
        "num_frames": args.num_frames,
        "frame_interval": args.frame_interval,
        "temporal_sample": args.temporal_sample,
        "seed": args.seed,
        "num_temporal_bands": args.num_temporal_bands,
        "num_spatial_bands": args.num_spatial_bands,
        "eps": args.eps,
        "outputs": [str(npz_path), str(csv_path), str(json_path)]
        + ([] if args.no_plot else [str(png_path)]),
        "inputs": [str(path) for path in paths],
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

"""
python scripts/adaptive_frequency/compute_adaptive_frequency_power.py \
    --latent-dir /scratch/s224075134/temporal_diffusion/datasets/video/ucf101_latent_16_kl_f8_autoencoder_bilinear_flip \
    --latent-format ucf101_gaussian \
    --posterior-stat sample \
    --output-dir results_adaptive_schedule/ucf101128/adaptive_frequency_power \
    --output-name dataset_prior_frequency_stats \
    --num-frames 16 \
    --frame-interval 3 \
    --num-temporal-bands 8 \
    --num-temporal-bands 4 \
    --max-items 3000
"""
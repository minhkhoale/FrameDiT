import sys; sys.path.extend(['.', 'tools'])

import argparse
import json
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from tools.new_metrics import VideoMetric, SharedVideoMetricModelRegistry


def load_fake_video(path: Path) -> np.ndarray:
    frames = imageio.mimread(path)
    if len(frames) == 0:
        raise ValueError(f"No frames read from {path}")
    video = np.stack(frames, axis=0)
    if video.shape[-1] == 4:
        video = video[..., :3]
    if video.shape[-1] == 1:
        video = np.repeat(video, 3, axis=-1)
    return video.astype(np.uint8, copy=False)


def load_real_condition_window(metadata: dict) -> np.ndarray:
    source_path = metadata.get("source_action_path")
    frame_indices = metadata.get("source_frame_indices")
    if not source_path or frame_indices is None:
        raise ValueError(
            "Metadata does not contain source_action_path/source_frame_indices. "
            "Exact action-matched FVD requires samples generated with random real action windows."
        )

    with np.load(source_path) as data:
        video = data["video"][np.asarray(frame_indices, dtype=np.int64)]
        actions = data["actions"][np.asarray(frame_indices, dtype=np.int64)]

    expected_actions = np.asarray(metadata.get("action_sequence", []), dtype=np.int64)
    if expected_actions.size and not np.array_equal(actions.reshape(-1), expected_actions.reshape(-1)):
        raise ValueError(f"Action sequence mismatch for {source_path}")

    return video.astype(np.uint8, copy=False)


def to_metric_tensor(videos: list[np.ndarray], device: torch.device) -> torch.Tensor:
    array = np.stack(videos, axis=0)
    tensor = torch.from_numpy(array).permute(0, 1, 4, 2, 3).to(device)
    return tensor.float() / 255.0


def compute_rgb_stats(pairs: list[tuple[Path, Path]]) -> tuple[np.ndarray, ...]:
    fake_sum = np.zeros(3, dtype=np.float64)
    fake_square_sum = np.zeros(3, dtype=np.float64)
    real_sum = np.zeros(3, dtype=np.float64)
    real_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    print("Computing dataset-level RGB statistics for fake normalization")
    for fake_path, metadata_path in tqdm(pairs):
        with open(metadata_path) as f:
            metadata = json.load(f)

        fake_video = load_fake_video(fake_path).astype(np.float64)
        real_video = load_real_condition_window(metadata).astype(np.float64)
        if fake_video.shape != real_video.shape:
            raise ValueError(
                f"Video shape mismatch for {fake_path}: fake {fake_video.shape}, real {real_video.shape}"
            )

        reduce_axes = (0, 1, 2)
        fake_sum += fake_video.sum(axis=reduce_axes)
        fake_square_sum += np.square(fake_video).sum(axis=reduce_axes)
        real_sum += real_video.sum(axis=reduce_axes)
        real_square_sum += np.square(real_video).sum(axis=reduce_axes)
        pixel_count += int(np.prod(fake_video.shape[:-1]))

    fake_mean = fake_sum / pixel_count
    real_mean = real_sum / pixel_count
    fake_std = np.sqrt(np.maximum(fake_square_sum / pixel_count - np.square(fake_mean), 0.0))
    real_std = np.sqrt(np.maximum(real_square_sum / pixel_count - np.square(real_mean), 0.0))
    return fake_mean, fake_std, real_mean, real_std


def normalize_fake_video(
    video: np.ndarray,
    fake_mean: np.ndarray,
    fake_std: np.ndarray,
    real_mean: np.ndarray,
    real_std: np.ndarray,
) -> np.ndarray:
    scale = real_std / np.maximum(fake_std, 1e-6)
    normalized = (video.astype(np.float32) - fake_mean) * scale + real_mean
    return np.clip(normalized, 0.0, 255.0).astype(np.float32)


def discover_pairs(fake_data_path: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for fake_path in sorted(fake_data_path.glob("*.mp4")):
        metadata_path = fake_path.with_suffix(".json")
        if metadata_path.is_file():
            pairs.append((fake_path, metadata_path))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute DMLab metrics using real windows with the exact action "
            "sequence used to generate each fake video."
        )
    )
    parser.add_argument("--fake_data_path", type=Path, required=True, help="Directory of generated MP4s and JSON sidecars.")
    parser.add_argument("--max_size", type=int, default=2048, help="Maximum number of fake/real condition pairs to evaluate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Video batch size before internal metric splitting.")
    parser.add_argument("--seed", type=int, default=21, help="Seed used when subsampling pairs.")
    parser.add_argument("--result_file", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["fvd", "fid"],
        choices=["fvd", "fid"],
        help="Metrics to compute. Defaults to action-matched FVD and frame-wise FID.",
    )
    parser.add_argument(
        "--normalize_fake_to_real",
        action="store_true",
        help=(
            "Match generated RGB means and standard deviations to the selected real "
            "windows before computing metrics. This changes metric comparability."
        ),
    )
    args = parser.parse_args()

    pairs = discover_pairs(args.fake_data_path)
    if len(pairs) == 0:
        raise SystemExit(
            f"No MP4/JSON sidecar pairs found in {args.fake_data_path}. "
            "Regenerate samples after the sample/sample_ddp.py metadata change."
        )

    rng = random.Random(args.seed)
    if args.max_size is not None and len(pairs) > args.max_size:
        pairs = sorted(rng.sample(pairs, args.max_size))

    normalization_stats = None
    if args.normalize_fake_to_real:
        normalization_stats = compute_rgb_stats(pairs)
        fake_mean, fake_std, real_mean, real_std = normalization_stats
        print(f"Fake RGB mean/std: {fake_mean.round(4)} / {fake_std.round(4)}")
        print(f"Real RGB mean/std: {real_mean.round(4)} / {real_std.round(4)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registry = SharedVideoMetricModelRegistry()
    metrics = VideoMetric(
        registry,
        args.metrics,
        split_batch_size=16,
        torchmetrics_kwargs={"sync_on_compute": False},
    ).to(device)

    print(
        f"Evaluating {len(pairs)} action-matched DMLab pairs from {args.fake_data_path} "
        f"with metrics: {args.metrics}"
    )
    fake_batch = []
    real_batch = []

    for fake_path, metadata_path in tqdm(pairs):
        with open(metadata_path) as f:
            metadata = json.load(f)

        fake_video = load_fake_video(fake_path)
        real_video = load_real_condition_window(metadata)

        if fake_video.shape[0] != real_video.shape[0]:
            raise ValueError(
                f"Frame count mismatch for {fake_path}: fake {fake_video.shape[0]}, real {real_video.shape[0]}"
            )

        if normalization_stats is not None:
            fake_video = normalize_fake_video(fake_video, *normalization_stats)

        fake_batch.append(fake_video)
        real_batch.append(real_video)

        if len(fake_batch) == args.batch_size:
            metrics(to_metric_tensor(fake_batch, device), to_metric_tensor(real_batch, device))
            fake_batch.clear()
            real_batch.clear()

    if fake_batch:
        metrics(to_metric_tensor(fake_batch, device), to_metric_tensor(real_batch, device))

    result = metrics.log("final")
    finals = {k: v.cpu().item() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
    for key, value in finals.items():
        print(f"{key:<20} : {value:.6f}")

    if args.result_file is not None:
        with open(args.result_file, "w") as f:
            json.dump(finals, f, indent=4)
        print(f"Saved results to {args.result_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert numpy video archives to MP4 files with imageio."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np


VIDEO_EXTENSIONS = {".npy", ".npz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Root containing .npy/.npz videos.")
    parser.add_argument("--dst", type=Path, required=True, help="Directory where MP4 files are written.")
    parser.add_argument("--key", default="video", help="Array key to read from .npz files.")
    parser.add_argument("--fps", type=int, default=30, help="Output MP4 frame rate.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel conversion workers.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing MP4 files.")
    parser.add_argument("--preserve-dirs", action="store_true", help="Mirror source subdirectories under dst.")
    parser.add_argument("--quality", type=int, default=8, help="imageio ffmpeg quality, 0-10.")
    return parser.parse_args()


def iter_inputs(src: Path, limit: int | None) -> list[Path]:
    paths = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if limit is not None:
        return paths[:limit]
    return paths


def output_path(src_root: Path, dst_root: Path, in_path: Path, preserve_dirs: bool) -> Path:
    rel = in_path.relative_to(src_root).with_suffix(".mp4")
    if preserve_dirs:
        return dst_root / rel
    return dst_root / f"{'_'.join(rel.with_suffix('').parts)}.mp4"


def load_video(path: Path, key: str) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            if key not in archive:
                raise KeyError(f"{path} has keys {archive.files}, not {key!r}")
            video = archive[key]
    else:
        video = np.load(path)

    video = np.asarray(video)
    if video.ndim != 4:
        raise ValueError(f"{path} expected 4D video, got shape {video.shape}")

    if video.shape[-1] in (1, 3, 4):
        pass
    elif video.shape[1] in (1, 3, 4):
        video = np.moveaxis(video, 1, -1)
    else:
        raise ValueError(f"{path} cannot infer channel axis from shape {video.shape}")

    if video.shape[-1] == 1:
        video = np.repeat(video, 3, axis=-1)
    elif video.shape[-1] == 4:
        video = video[..., :3]

    if video.dtype != np.uint8:
        if np.issubdtype(video.dtype, np.floating):
            max_value = float(np.nanmax(video))
            min_value = float(np.nanmin(video))
            if min_value >= 0.0 and max_value <= 1.0:
                video = video * 255.0
            video = np.clip(video, 0, 255)
        video = video.astype(np.uint8)

    return np.ascontiguousarray(video)


def convert_one(task: tuple[str, str, str, int, bool, int]) -> tuple[str, str]:
    in_str, out_str, key, fps, overwrite, quality = task
    in_path = Path(in_str)
    out_path = Path(out_str)

    if out_path.exists() and not overwrite:
        return "skipped", str(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    video = load_video(in_path, key)
    imageio.mimwrite(
        out_path,
        video,
        fps=fps,
        codec="libx264",
        quality=quality,
        pixelformat="yuv420p",
        macro_block_size=16,
    )
    return "written", str(out_path)


def safe_convert_one(task: tuple[str, str, str, int, bool, int]) -> tuple[str, str]:
    try:
        return convert_one(task)
    except Exception as exc:
        return "failed", f"{task[0]}: {exc}"


def build_tasks(args: argparse.Namespace, inputs: Iterable[Path]) -> list[tuple[str, str, str, int, bool, int]]:
    return [
        (
            str(in_path),
            str(output_path(args.src, args.dst, in_path, args.preserve_dirs)),
            args.key,
            args.fps,
            args.overwrite,
            args.quality,
        )
        for in_path in inputs
    ]


def main() -> None:
    args = parse_args()
    args.src = args.src.absolute()
    args.dst = args.dst.absolute()

    inputs = iter_inputs(args.src, args.limit)
    if not inputs:
        raise SystemExit(f"No .npy/.npz files found under {args.src}")

    tasks = build_tasks(args, inputs)
    counts = {"written": 0, "skipped": 0, "failed": 0}

    def update(result: tuple[str, str]) -> None:
        status, path = result
        counts[status] = counts.get(status, 0) + 1
        done = sum(counts.values())
        if done == 1 or done % 100 == 0 or done == len(tasks):
            print(f"{done}/{len(tasks)} {status}: {path}", flush=True)

    if args.workers == 1:
        for task in tasks:
            update(safe_convert_one(task))
    else:
        with mp.Pool(args.workers) as pool:
            for result in pool.imap_unordered(safe_convert_one, tasks):
                update(result)

    print(
        f"Finished: {counts['written']} written, {counts['skipped']} skipped, {counts['failed']} failed",
        flush=True,
    )
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

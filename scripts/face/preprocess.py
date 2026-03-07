import os
import cv2
import argparse
import numpy as np
import imageio as iio
from pathlib import Path
from decord import VideoReader

def resize_and_center_crop_frames(frames: np.ndarray, short_side: int, crop_size: int) -> np.ndarray:
    """
    frames: (T, H, W, 3) uint8
    1) scale so min(H,W) == short_side (keep aspect)
    2) center crop to (crop_size, crop_size)
    return: (T, crop_size, crop_size, 3) uint8
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames (T,H,W,3), got {frames.shape}")

    T, H, W, _ = frames.shape
    scale = short_side / min(H, W)
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))

    # Resize each frame (OpenCV is fast)
    resized = np.empty((T, new_h, new_w, 3), dtype=np.uint8)
    for t in range(T):
        resized[t] = cv2.resize(frames[t], (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center crop
    top = (new_h - crop_size) // 2
    left = (new_w - crop_size) // 2
    if top < 0 or left < 0:
        raise ValueError(
            f"crop_size={crop_size} is larger than resized frame {(new_h, new_w)}. "
            f"Increase --short_side or decrease --crop_size."
        )

    cropped = resized[:, top : top + crop_size, left : left + crop_size]
    return cropped



def process_video(input_path, output_path, short_side=512, crop_size=512):
    # --- read entire video fast ---
    print(input_path)
    vr = VideoReader(str(input_path))
    frames = vr.get_batch(range(len(vr))).asnumpy()
    fps = float(vr.get_avg_fps())
    print(frames.shape)

    # if frames.size == 0:
    #     print(f"❌ Empty: {input_path}")
    #     return

    frames = resize_and_center_crop_frames(frames, short_side, crop_size)

    # --- write video ---
    iio.mimwrite(
        output_path,
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
    )

    print(f"✅ Saved: {output_path}")



def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(list(input_dir.glob("*.mp4")))

    print(f"Found {len(videos)} videos")

    for vid in videos:
        out_path = output_dir / vid.name
        process_video(
            vid,
            out_path,
            short_side=args.short_side,
            crop_size=args.crop_size,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--short_side", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=256)

    args = parser.parse_args()
    main(args)
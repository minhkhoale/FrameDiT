import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import imageio

import torch
import torch.nn.functional as F
from torchvision.io import read_video
from torchvision import transforms as T
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights, raft_large, Raft_Large_Weights

# ----------------------------
# Helpers
# ----------------------------
@torch.no_grad()
def _pad8(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad CHW tensor so H, W are multiples of 8 (RAFT-friendly)."""
    h, w = x.shape[-2:]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, (ph, pw)

def _unpad8(x: torch.Tensor, pad: Tuple[int, int]) -> torch.Tensor:
    ph, pw = pad
    return x[..., : x.shape[-2] - ph if ph else None, : x.shape[-1] - pw if pw else None]

def flow_to_bgr(flow_hw2: np.ndarray, clip_mag: Optional[float] = None) -> np.ndarray:
    """
    Colorize flow (H,W,2) -> BGR image via HSV.
    Hue = direction, Value = normalized magnitude.
    """
    u, v = flow_hw2[..., 0], flow_hw2[..., 1]
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)
    if clip_mag is not None:
        mag = np.clip(mag, 0, clip_mag)
    hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)  # OpenCV H in [0,179]
    hsv[..., 1] = 255
    hsv[..., 2] = (255 * (mag / (mag.max() + 1e-6))).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def draw_flow_arrows(
    img_bgr: np.ndarray, flow_hw2: np.ndarray, step: int = 16, scale: float = 1.0
) -> np.ndarray:
    """Optional: draw sparse arrows for readability."""
    h, w = img_bgr.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            fx, fy = flow_hw2[y, x]
            tip = (int(x + scale * fx), int(y + scale * fy))
            cv2.arrowedLine(img_bgr, (x, y), tip, (255, 255, 255), 1, tipLength=0.3)
    return img_bgr

def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
        ]
    )
    batch = transforms(batch)
    return batch
# ----------------------------
# Main
# ----------------------------
@torch.no_grad()
def flow_overlay_video(
    video_path: str,
    out_video_path: str = "flow_overlay.mp4",
    alpha: float = 0.5,              # overlay strength (0..1)
    save_flows_dir: Optional[str] = None,
    draw_arrows_flag: bool = False,
    arrow_step: int = 16,
    arrow_scale: float = 4.0,
    clip_mag: Optional[float] = None # clip magnitude for consistent visualization
) -> None:
    """
    Compute pairwise RAFT flow for all consecutive frames and save an overlay video.
    - Overlay for frame t uses flow(t -> t+1) color map blended over frame t.
    - Optionally saves raw flow tensors (.pt) into save_flows_dir.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = Raft_Small_Weights.DEFAULT
    raft = raft_small(weights=weights).to(device).eval()
    tfm = weights.transforms()

    # Read video as TCHW (uint8 RGB) and metadata
    vid, _, info = read_video(video_path, output_format="TCHW")
    print(f"Read video {video_path}: {vid.shape}, {info}")
    T, C, H, W = vid.shape
    if T < 2:
        raise ValueError("Need at least 2 frames.")
    fps = info.get("video_fps", 24)

    # Prepare writer (OpenCV expects BGR)
    Path(out_video_path).parent.mkdir(parents=True, exist_ok=True)

    result_videos: List[np.ndarray] = []

    if save_flows_dir:
        Path(save_flows_dir).mkdir(parents=True, exist_ok=True)

    for t in range(T - 1):
        a = vid[t]     # CHW, uint8 RGB
        b = vid[t + 1]
        # transforms() expects list(s) of tensors in CHW
        a_t = preprocess(a.unsqueeze(0)).squeeze(0)
        b_t = preprocess(b.unsqueeze(0)).squeeze(0)
        a_t, pad = _pad8(a_t)
        b_t, _   = _pad8(b_t)
        a_t = a_t.unsqueeze(0).to(device)  # NCHW
        b_t = b_t.unsqueeze(0).to(device)

        out = raft(a_t, b_t)
        flow = out[-1] if isinstance(out, (list, tuple)) else out  # [1,2,H',W']
        flow = _unpad8(flow[0].cpu(), pad)                         # [2,H,W]

        # Save raw flow tensor if requested
        if save_flows_dir:
            stem = f"{Path(video_path).stem}_t{t:06d}_to_t{t+1:06d}"
            torch.save(flow, Path(save_flows_dir, f"{stem}.pt"))

        # Colorize & overlay on frame t
        flow_np = flow.permute(1, 2, 0).numpy().astype(np.float32)  # HxWx2
        flow_bgr = flow_to_bgr(flow_np, clip_mag=clip_mag)          # HxWx3 BGR

        frame_rgb = vid[t].permute(1, 2, 0).numpy()                 # HxWx3 RGB
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(frame_bgr, 1.0 - alpha, flow_bgr, alpha, 0)

        if draw_arrows_flag:
            draw_flow_arrows(overlay, flow_np, step=arrow_step, scale=arrow_scale)

        result_videos.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))  # for imageio

    # (Optional) write the last frame without overlay so lengths match original:
    last_rgb = vid[-1].permute(1, 2, 0).numpy()
    result_videos.append(last_rgb)
    # post process for imageio
    result_videos = [np.clip(frame, 0, 255).astype(np.uint8) for frame in result_videos]

    imageio.mimsave(out_video_path, result_videos, fps=fps)

if __name__ == "__main__":
    # Example for a 16-frame clip:
    flow_overlay_video(
        video_path="/scratch/s224075134/temporal_diffusion/datasets/video/taichi/train/irQNFGmGRQQ#002662#002790.mp4",
        out_video_path="test_data/flow_overlay.mp4",
        alpha=0.5,                 # stronger/weaker overlay
        save_flows_dir=None,  # set None to skip saving .pt tensors
        draw_arrows_flag=True,     # set False to hide arrows
        arrow_step=48,
        arrow_scale=4.0,
        clip_mag=None              # e.g., set to 15.0 for consistent scaling across clips
    )

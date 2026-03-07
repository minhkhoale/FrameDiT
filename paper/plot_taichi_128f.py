import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
import os
import imageio

methods = [
    'Local Factorized',
    'Full 3D',
    'FrameDiT-G',
    'FrameDiT-H',
]


def load_video_frames(video_path):
    """Load all frames from a video file using imageio."""
    reader = imageio.get_reader(video_path)
    frames = []
    for frame in reader:
        frames.append(frame)   # frame is HWC uint8
    reader.close()
    return frames

def plot_videos_from_paths(video_paths, frame_indices, figsize=(16, 4), two_rows=False):
    """
    Args:
        video_paths: list of paths to video files.
        frame_indices: list of frame indices to extract.
        figsize: size of the figure.
        two_rows: if True, split frames into two rows per video.
    """

    # Load all videos first
    videos = []
    for path in video_paths:
        assert os.path.exists(path), f"Video file not found: {path}"
        frames = load_video_frames(path)
        videos.append(frames)

    num_videos = len(videos)
    num_frames = len(frame_indices)

    # ----------------------------
    # Option 1: One row per video
    # ----------------------------
    fig, axes = plt.subplots(num_videos, num_frames,
                                figsize=(13.7, num_videos * 1.5 + 2.25))

    # Make axes 2D for consistency
    if num_videos == 1:
        axes = axes[None, :]

    for v_idx, frames in enumerate(videos):
        for f_idx, frame_index in enumerate(frame_indices):
            if frame_index >= len(frames):
                raise ValueError(f"Frame index {frame_index} out of range for {video_paths[v_idx]}")
            axes[v_idx, f_idx].imshow(frames[frame_index])
            axes[v_idx, f_idx].axis("off")
        
        # if v_idx % 3 == 1:
        #     print('v_idx', v_idx)
        mid_row = v_idx 
        fig.text(
            0.01,  # x-position on figure
            1 - (mid_row + 0.5) / len(methods),  # convert row index to figure y
            methods[v_idx % len(methods)],
            va='center', ha='left', fontsize=14,rotation=90
                )



    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03, hspace=0.05, left=0.03, right=1.0)
    plt.savefig("videos_taichi128_128f_grid.png", dpi=300)
    plt.savefig("videos_taichi128_128f_grid.pdf", dpi=300)


videos = [
    # '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_128f/latte/0011.mp4',
    # '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_128f/fusedmatlatte/0000.mp4',
    # '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_128f/fusedmatlatte/0186.mp4',
    '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_128f/latte/0012.mp4',
    '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_128f/0000.mp4',
    '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_128f/fusedmatlatte/0013.mp4',
    # '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128/fusedmatlatte/0161.mp4',
    # '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128/fusedmatlatte/0179.mp4',
    '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_128f/fusedmatlatte/0256.mp4'
]   # each video: numpy array [T, H, W, C]
frame_indices = [i for i in range(0, 128, 16)]  # 16 frames

plot_videos_from_paths(videos, frame_indices, two_rows=True)   # one row per video
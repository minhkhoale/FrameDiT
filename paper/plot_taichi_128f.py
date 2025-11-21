import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
import os
import imageio

methods = [
    'Local Factorized',
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
    if not two_rows:
        fig, axes = plt.subplots(num_videos, num_frames,
                                 figsize=(8,1))

        # Make axes 2D for consistency
        if num_videos == 1:
            axes = axes[None, :]

        for v_idx, frames in enumerate(videos):
            for f_idx, frame_index in enumerate(frame_indices):
                if frame_index >= len(frames):
                    raise ValueError(f"Frame index {frame_index} out of range for {video_paths[v_idx]}")
                axes[v_idx, f_idx].imshow(frames[frame_index])
                axes[v_idx, f_idx].axis("off")



        plt.tight_layout()
        plt.subplots_adjust(wspace=0.00001, hspace=0.00001)
        plt.savefig("videos_taichi128_128f_grid.png", dpi=300)
        plt.savefig("videos_taichi128_128f_grid.pdf", dpi=300)
        return

    # -------------------------------------------
    # Option 2: Two rows per video (split frames)
    # -------------------------------------------
    split = num_frames // 2
    top_frames = frame_indices[:split]
    bot_frames = frame_indices[split:]

    fig, axes = plt.subplots(num_videos * 2, split,
                             figsize=(12.7, num_videos * 3 + 0.5))

    for v_idx, frames in enumerate(videos):

        # Top row
        for col, frame_index in enumerate(top_frames):
            axes[v_idx*2, col].imshow(frames[frame_index])
            axes[v_idx*2, col].axis("off")

        # Bottom row
        for col, frame_index in enumerate(bot_frames):
            axes[v_idx*2 + 1, col].imshow(frames[frame_index])
            axes[v_idx*2 + 1, col].axis("off")

        if v_idx % 2 == 1:
            mid_row = v_idx - 1
            fig.text(
                0.01,  # x-position on figure
                1 - (mid_row + 1) / (3 * 2),  # convert row index to figure y
                methods[v_idx // 2],
                va='center', ha='left', fontsize=14,rotation=90
                    )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03, hspace=0.05, left=0.03, right=1.0)
    # fig.subplots_adjust(
    #     left=side_padding_left,
    #     right=1 - side_padding_right,
    #     top=0.98,
    #     bottom=0.02,
    #     wspace=inner_padding,
    #     hspace=inner_padding
    # )

    # plt.show()
        
    # save png, pdf
    plt.savefig("videos_taichi128_128f_grid.png", dpi=300)
    plt.savefig("videos_taichi128_128f_grid.pdf", dpi=300)


videos = [
    '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_128f/latte/0011.mp4',
    '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_128f/latte/0012.mp4',
    '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_128f/fusedmatlatte/0000.mp4',
    '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_128f/fusedmatlatte/0013.mp4',
    # '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128/fusedmatlatte/0161.mp4',
    # '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128/fusedmatlatte/0179.mp4',
    '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_128f/fusedmatlatte/0186.mp4',
    '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_128f/fusedmatlatte/0256.mp4'
]   # each video: numpy array [T, H, W, C]
frame_indices = [0, 7, 15, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127]

plot_videos_from_paths(videos, frame_indices, two_rows=True)   # one row per video
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
    print('num_videos', num_videos)

    # ----------------------------
    # Option 1: One row per video
    # ----------------------------
    # -------------------------------------------
    # Option 2: Two rows per video (split frames)
    # -------------------------------------------
    split = num_frames // 2
    top_frames = frame_indices[:split]
    bot_frames = frame_indices[split:]

    fig, axes = plt.subplots(num_videos * 2, split,
                             figsize=(13.5, num_videos * 3 + 1.5))

    for v_idx, frames in enumerate(videos):
        print('v_idx', v_idx)
        # Top row
        for col, frame_index in enumerate(top_frames):
            print('col', col, 'frame_index', frame_index)
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
    plt.savefig("videos_taichi128_64f_grid.png", dpi=300)
    plt.savefig("videos_taichi128_64f_grid.pdf", dpi=300)

    


videos = [
    '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_64f/latte/0002.mp4',
    '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_64f/latte/0037.mp4',
    '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_64f/framedit-g/2036.mp4',
    '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_64f/framedit-g/2040.mp4',
    '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_64f/framedit-h/0043.mp4',
    '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/qualitative_video/taichi128_64f/framedit-h/0052.mp4'
]   # each video: numpy array [T, H, W, C]
frame_indices = [i for i in range(0, 64, 4)]  # 16 frames

plot_videos_from_paths(videos, frame_indices, two_rows=True)   # one row per video
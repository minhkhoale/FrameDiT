import os
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# ============ CONFIG ============
# Directories containing frame folders
methods = {
    "GT": "/path/to/ground_truth_frames",
    "Full-DiT": "/path/to/full_dit_outputs",
    "Factorized-DiT": "/path/to/factorized_dit_outputs",
    "MatrixDiT (Ours)": "/path/to/matrixdit_outputs"
}

# Output folder
save_dir = "./qualitative_results"
os.makedirs(save_dir, exist_ok=True)

# Frames to visualize
frame_indices = [0, 5, 10, 15]  # sample temporal positions
num_frames_show = len(frame_indices)

# Target resolution (resize for uniformity)
target_size = (256, 256)

# ============ FUNCTIONS ============

def load_frame(path, resize_to):
    img = Image.open(path).convert("RGB")
    if resize_to is not None:
        img = img.resize(resize_to, Image.Resampling.LANCZOS)
    return np.array(img)

def load_video_frames(folder, frame_idxs):
    files = sorted([f for f in os.listdir(folder) if f.endswith(('.png', '.jpg'))])
    frames = [load_frame(os.path.join(folder, files[i]), target_size) for i in frame_idxs]
    return frames

def create_comparison_grid(video_dict, save_path):
    """
    video_dict: {method_name: list of np.ndarray frames}
    """
    methods_list = list(video_dict.keys())
    num_methods = len(methods_list)
    fig, axes = plt.subplots(num_methods, num_frames_show, figsize=(num_frames_show * 2.2, num_methods * 2.0))

    for row, method in enumerate(methods_list):
        frames = video_dict[method]
        for col, frame in enumerate(frames):
            ax = axes[row, col] if num_methods > 1 else axes[col]
            ax.imshow(frame)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"t={frame_indices[col]}", fontsize=10)
        axes[row, 0].set_ylabel(method, fontsize=9, rotation=0, labelpad=40)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

# ============ MAIN LOOP ============
video_names = sorted(os.listdir(methods["GT"]))

for vid_name in video_names:
    # assume folder names are consistent across methods
    print(f"Processing {vid_name}...")
    video_dict = {}
    for method, root_dir in methods.items():
        video_path = os.path.join(root_dir, vid_name)
        video_dict[method] = load_video_frames(video_path, frame_indices)
    
    save_path = os.path.join(save_dir, f"{vid_name}_comparison.png")
    create_comparison_grid(video_dict, save_path)

print("✅ Done. Qualitative grids saved in:", save_dir)

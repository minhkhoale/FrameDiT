import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import matplotlib.pyplot as plt

# ============ CONFIG ============
# methods = {
#     "AR-Diffusion": "paper/qualitative_video/sky/ardiffusion",
#     "Latte": "paper/qualitative_video/sky/latte",
#     "MatrixDiT (Ours)": "paper/qualitative_video/sky/matrixdit",
# }
# save_path = "./qualitative_results/sky_comparison.png"

import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import matplotlib.pyplot as plt

# ==================================
# CONFIG
# ==================================
root_folder = "qualitative_video"    # <--- root folder

datasets = ["FaceForensics", "SkyTimelapse", "Taichi-HD"]           # <--- each dataset becomes a COLUMN group

methods = ["AR-Diffusion", "Latte", "MatrixDiT-H (Ours)"]

# number of videos per method
video_counts = {
    "AR-Diffusion": 2,
    "Latte": 2,
    "MatrixDiT-H (Ours)": 2,
}

frame_indices = [0, 4, 8, 12, 15]
target_size = (128, 128)
num_frames = len(frame_indices)

save_path = "./qualitative_results/multi_dataset_grid.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)


# ==================================
# HELPERS
# ==================================
def load_frame(frame, resize_to):
    img = Image.fromarray(frame).convert("RGB")
    if resize_to is not None:
        img = img.resize(resize_to, Image.Resampling.LANCZOS)
    return np.array(img)

def load_video_frames(video_path, frame_idxs, resize_to):
    reader = imageio.get_reader(video_path)
    n = reader.count_frames()
    idxs = [min(i, n - 1) for i in frame_idxs]
    frames = [load_frame(reader.get_data(i), resize_to) for i in idxs]
    reader.close()
    return frames

def list_videos(folder):
    if not os.path.exists(folder):
        return []
    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".mp4") and not f.startswith(".")
    ])


# ==================================
# GRID DIMENSIONS
# ==================================
rows_per_method = [video_counts[m] for m in methods]
total_rows = sum(rows_per_method)     # rows per dataset block
total_cols = len(datasets) * num_frames

fig, axes = plt.subplots(
    total_rows, total_cols,
    figsize=(total_cols * 1.5, total_rows * 1.5),
    squeeze=False
)
method_row_start = 0 
# For each dataset (column block)
for d_idx, dataset in enumerate(datasets):
    col_offset = d_idx * num_frames

    row_idx = 0
    for method in methods:
        method_videos = video_counts[method]
        method_path = os.path.join(root_folder, dataset, method)

        video_files = list_videos(method_path)[:method_videos]

        for vid_file in video_files:
            full_path = os.path.join(method_path, vid_file)
            frames = load_video_frames(full_path, frame_indices, target_size)

            for f_idx, frame in enumerate(frames):
                ax = axes[row_idx, col_offset + f_idx]
                ax.imshow(frame)
                ax.axis("off")
                ax.set_aspect("equal")

            # Label each video on the leftmost dataset
            if d_idx == 0:
                axes[row_idx, 0].set_ylabel(
                    f"{method}\n({vid_file})",
                    fontsize=7, rotation=0, labelpad=15, va="center"
                )

            row_idx += 1

        
        if d_idx == 0:
            row_start = method_row_start
            row_end = row_start + 2

            # ---- Add method label on far left ----
            y_center = (row_start + 2 / 2) / total_rows

            fig.text(
                0.049,                # x-position (left of plot)
                1 - y_center,         # y-position (centered vertically)
                method,
                fontsize=16,
                fontweight="bold" if "Ours" in method else "normal",
                rotation=90,
                va="center",
                ha="center"
            )
            method_row_start += 2
        
        

    # Add dataset header across its column group
    mid_col = col_offset + num_frames / 2
    fig.text(
        mid_col / total_cols,
        0.98,
        dataset,
        ha="center",
        va="bottom",
        fontsize=16,
        #weight="bold"
    )


plt.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.02, wspace=0.02, hspace=0.01)
plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.00)
plt.close(fig)

print(f"✅ Saved multi-dataset comparison grid to: {save_path}")


# methods = {
#     "AR-Diffusion": "qualitative_video/face/ardiffusion",
#     "Latte": "qualitative_video/face/latte",
#     "MatrixDiT (Ours)": "qualitative_video/face/matrixdit",
# }
# save_path = "./qualitative_results/face_comparison.png"

# os.makedirs(os.path.dirname(save_path), exist_ok=True)

# frame_indices = [0, 4, 8, 12, 15]
# target_size = (128,128)
# num_frames = len(frame_indices)

# # number of videos per method (baseline: 1, ours: 2)
# method_video_counts = {
#     "AR-Diffusion": 2,
#     "Latte": 2,
#     "MatrixDiT (Ours)": 2,
# }

# # ============ FUNCTIONS ============
# def load_frame_array(frame, resize_to):
#     img = Image.fromarray(frame).convert("RGB")
#     if resize_to is not None:
#         img = img.resize(resize_to, Image.Resampling.LANCZOS)
#     return np.array(img)

# def load_video_frames(video_path, frame_idxs, resize_to):
#     reader = imageio.get_reader(video_path)
#     n_frames = reader.count_frames()
#     valid_idxs = [min(i, n_frames - 1) for i in frame_idxs]
#     frames = [load_frame_array(reader.get_data(i), resize_to) for i in valid_idxs]
#     reader.close()
#     return frames

# def list_videos(folder):
#     return sorted([
#         f for f in os.listdir(folder)
#         if f.lower().endswith(".mp4") and not f.startswith(".")
#     ])

# # ============ MAIN ============
# num_methods = len(methods)
# rows_per_method = [method_video_counts[m] for m in methods]
# total_rows = sum(rows_per_method)

# # force all subplots equal size
# fig, axes = plt.subplots(
#     total_rows, num_frames,
#     figsize=(num_frames * 2.0, total_rows * 2.0),
#     squeeze=False
# )

# row_idx = 0
# for m_idx, (method, root_dir) in enumerate(methods.items()):
#     num_videos = method_video_counts[method]
#     video_files = list_videos(root_dir)[:num_videos]
#     for v_idx, vid_file in enumerate(video_files):
#         frames = load_video_frames(os.path.join(root_dir, vid_file), frame_indices, target_size)
#         for f_idx, frame in enumerate(frames):
#             ax = axes[row_idx, f_idx]
#             ax.imshow(frame)
#             ax.axis("off")
#             ax.set_aspect("equal")       # enforce square cells

#             # if row_idx == 0:
#             #     ax.set_title(f"t={frame_indices[f_idx]}", fontsize=9, pad=2)

#         # left label (video name)
#         axes[row_idx, 0].set_ylabel(
#             f"({vid_file})",
#             fontsize=7.5,
#             rotation=0,
#             labelpad=15,
#             va="center"
#         )
#         row_idx += 1

# # add method labels beside corresponding blocks
# row_offset = 0
# for m_idx, (method, _) in enumerate(methods.items()):
#     block_size = method_video_counts[method]
#     y_center = (row_offset + block_size / 2) / total_rows
#     plt.figtext(
#         -0.015,
#         1 - y_center,
#         method,
#         fontsize=11,
#         fontweight="bold",
#         rotation=90,
#         va="center",
#         ha="center"
#     )
#     row_offset += block_size

# plt.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.02, wspace=0.02, hspace=0.08)
# plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
# plt.close(fig)

# print(f"✅ Saved perfectly uniform comparison grid to: {save_path}")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import StringIO

# -------------------------------------------------------
# CVPR Style
# -------------------------------------------------------
def set_cvpr_style():
    palette = {
        "Local Factorized": "#B75C00",
        "Full 3D": "#D282D2",
        "FrameDiT-G": "#3A5BCD",
        "FrameDiT-H": "#2D8A4E",
    }
    sns.set_theme(context="paper", style="whitegrid")
    sns.set_palette(list(palette.values()))
    plt.rcParams.update({
        "figure.figsize": (8.6, 2.8),
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.linewidth": 0.7,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "grid.alpha": 0.3,
    })
    return palette

def beautify_ax(ax, x_label, y_label, logy=False):
    ax.set_xlabel(x_label)
    ax.xaxis.label.set_size(9)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="both", which="major", length=3.2, width=0.8, pad=1.5)
    for s in ax.spines.values():
        s.set_color("black")
        s.set_linewidth(0.5)
    if logy:
        ax.set_yscale("log")


# -------------------------------------------------------
# Plot FVD vs Steps
# -------------------------------------------------------
def plot_fvd_vs_steps(df, save_path):
    palette = set_cvpr_style()
    df["model"] = df["model"].replace({
        "MatrixDiT-G": "FrameDiT-G",
        "MatrixDiT-H": "FrameDiT-H"
    })

    n_frames_list = sorted(df["n_frames"].unique())
    model_order = ["Local Factorized", "Full 3D", "FrameDiT-G", "FrameDiT-H"]

    fig, axes = plt.subplots(
        1, len(n_frames_list),
        figsize=(10.6, 2.4),
        sharex=True
    )

    if len(n_frames_list) == 1:
        axes = [axes]

    for ax, nf in zip(axes, n_frames_list):
        sub = df[df["n_frames"] == nf]

        for model in model_order:
            mdf = sub[sub["model"] == model]
            if len(mdf) == 0:
                continue

            ax.plot(
                mdf["steps"]/1000, mdf["fvd"],
                label=model,
                color=palette[model],
                linewidth=1.7,
            )

        ax.set_title(f"{nf} Frames", fontsize=12, pad=4)
        beautify_ax(ax, "Training Steps (x1000)", "FVD↓", logy=False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=4,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#D6D6D6",
        fontsize=10,
        handlelength=1.5,
        columnspacing=1.0
    )

    fig.tight_layout(pad=0.5, w_pad=0.5)
    plt.subplots_adjust(bottom=0.22)
    for ext in [".pdf", ".png"]:
        fig.savefig(save_path + ext, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved figure to {save_path}.pdf and .png")


# -------------------------------------------------------
# Load Data & Run
# -------------------------------------------------------

df = pd.read_csv('/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/paper/fvd_convergence.csv')
df = df[df['steps'] >= 240000]
df['steps'] = df['steps'] / 4  # in thousands

plot_fvd_vs_steps(df, "./fvd_vs_steps")

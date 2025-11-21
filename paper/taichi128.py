import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------------------------
# CVPR-style setup
# --------------------------------------------------------------------
def set_cvpr_style():
    palette = {
        "Local Factorized": "#B75C00",
        "Full 3D": "#D282D2",
        "MatrixDiT-G": "#3A5BCD",
        "MatrixDiT-H": "#2D8A4E",
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
        #"grid.linestyle": "--",
    })
    return palette

def beautify_ax(ax, x_label, y_label, logy=False):
    ax.set_xlabel(x_label)
    # xlabel fontsize
    ax.xaxis.label.set_size(9)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="both", which="major", length=3.2, width=0.8, pad=1.5)
    for s in ax.spines.values():
        s.set_color("black")
        s.set_linewidth(0.5)
    if logy:
        ax.set_yscale("log")
    # ax.xaxis.set_minor_locator(None)
    # ax.yaxis.set_minor_locator(None)

# --------------------------------------------------------------------
# Plot with OOM highlight tied to Full DiT
# --------------------------------------------------------------------
def plot_metrics_vs_frames(save_path: Path):
    model_order = ["MatrixDiT-G", "MatrixDiT-H", "Local Factorized", "Full 3D"]
    data = {
        "Model": ["Local Factorized", "Full 3D", "MatrixDiT-G", "MatrixDiT-H"],
        "FVD_16": [75.7, 71.7, 70.3, 67.2],
        "FVMD_16": [1039, 999.0, 990, 982],
        "FID_16": [16.18, 15.01, 14.44, 13.91],
        "PSNR_16": [9.15, 9.10, 9.14, 9.16],
        "FVD_32": [156.0, 148.3, 138.8, 126.9],
        "FVMD_32": [539.3, 497.7, 488, 478],
        "FID_32": [16.29, 16.46, 16.61, 15.48],
        "PSNR_32": [9.12, 9.14, 9.06, 9.13],
        "FVD_64": [302.4, np.nan, 228.0, 225.3],
        "FVMD_64": [146.7, np.nan, 134.397, 125.811],
        "FID_64": [21.40, np.nan, 16.15, 17.54],
        "PSNR_64": [9.20, np.nan, 9.16, 9.15],
        "FVD_128": [354.4, np.nan, 265.6, 256.4],
        "FVMD_128": [80.47, np.nan,  74.2,  70.1],
        "FID_128": [30.46, np.nan, 23.13, 22.29],
        "PSNR_128": [9.27, np.nan, 9.26, 9.26],
    }

    df = pd.DataFrame(data)

    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
    df = df.set_index("Model").reindex(model_order).reset_index()
    #order: MatrixDiT-G, MatrixDiT-H, Local Factorized, Full 3D

    #df = df.sort_values("Model").reset_index(drop=True)


    frames = [16, 32, 64, 128]
    metrics = [("FVD", "FVD↓"), ("FVMD", "FVMD↓"), ("FID", "FID↓"), ("PSNR", "PSNR↑")]
    palette = set_cvpr_style()

    #fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.6), dpi=400, sharex=True)
    subplot_size = 2.4  # inches per subplot (adjust as needed)
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 2.4), dpi=400, sharex=True)

    for i, (metric, ylabel) in enumerate(metrics):
        ax = axes[i]
        if metric == "FVMD":
            # --- compute normalized FVMD relative to Local Factorized ---
            local_vals = np.array([df[f"FVMD_{f}"].loc[df["Model"] == "Local Factorized"].values[0] for f in frames])
            for model in df["Model"]:
                # if model == "Local Factorized":
                #     continue  # skip baseline itself in normalization
                y = np.array([df[f"FVMD_{f}"].loc[df["Model"] == model].values[0] for f in frames])
                # ratio = y / local_vals
                improvement = (local_vals - y) / local_vals * 100.0
                print('improvement:', improvement)
                ax.plot(frames, improvement, marker="o", linewidth=1.7, markersize=0,
                        label=model if i == 1 else None,
                        color=palette[model], zorder=2)
            # add baseline line at 1.0
            #ax.axhline(100, color="gray", lw=0.8, ls="--", zorder=1)
            beautify_ax(ax, "Video length", "% FVMD Gain over Local↑")
            # ax.set_ylim(0.85, 1.15)
            # ax.set_yticks([0.9, 1.0, 1.1])
            ax.yaxis.grid(True, linestyle="--", alpha=0.3)
            continue

        for model in df["Model"]:
            y = [df[f"{metric}_{f}"].loc[df["Model"] == model].values[0] for f in frames]
            ax.plot(frames, y, marker="o", linewidth=1.7, markersize=0,
                    label=model if i == 1 else None,
                    color=palette[model], zorder=2)

            # ---- improved OOM indicator ----
            # if model == "Full DiT":
            #     color = palette[model]
            #     # locate last valid
            #     valid_idx = np.where(~pd.isna(y))[0]
            #     if len(valid_idx) > 0:
            #         last_idx = valid_idx[-1]
            #         x_last, y_last = frames[last_idx], y[last_idx]
            #         print('y_last:', y_last)
            #         for f, val in zip(frames, y):
            #             if pd.isna(val):
            #                 # dashed extension
            #                 ax.plot([x_last, f], [y_last, y_last],
            #                         linestyle="--", color=color, lw=1.0, alpha=0.8, zorder=1)
            #                 # colored “×”
            #                 ax.scatter(f, y_last, marker="x", s=38, color=color, edgecolor="black", lw=1.0, zorder=5)
            #                 # label near marker
            #                 ax.text(f + 1, y_last * 1.03 if y_last > 15.0 else y_last * 1.0005, "OOM",
            #                         color=color, fontsize=8.5,
            #                         fontweight="medium", ha="left", va="bottom",
            #                         bbox=dict(facecolor="white", edgecolor=color,
            #                                   boxstyle="round,pad=0.15", lw=0.6, alpha=0.8),
            #                         zorder=6)
            if model == "Full":
                color = palette[model]
                valid_idx = np.where(~pd.isna(y))[0]
                if len(valid_idx) > 0:
                    last_idx = valid_idx[-1]
                    x_last, y_last = frames[last_idx], y[last_idx]
                    # for f, val in zip(frames, y):
                    #     if pd.isna(val):
                    #         ax.plot([x_last, f], [y_last, y_last], linestyle="--", color=color, lw=1.0, alpha=0.8)
                    #         ax.scatter(f, y_last, marker="x", s=35, color=color, lw=1.4, zorder=5)
        beautify_ax(ax, "Video length", ylabel, logy=(metric in ["FVMD"]))
        # ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(frames)
        ax.margins(x=0)
        ax.set_xlim(frames[0], frames[-1])
        

        # axis stubs
        # x0, x1 = ax.get_xlim()
        # y0, y1 = ax.get_ylim()
        # bx, by = 0.02*(x1-x0), 0.02*(y1-y0)
        # ax.plot([x0, x0+bx], [y0, y0], color="black", lw=0.9, clip_on=False)
        # ax.plot([x0, x0], [y0, y0+by], color="black", lw=0.9, clip_on=False)

    # shared legend
    handles, labels = axes[1].get_legend_handles_labels()
    # legend = fig.legend(
    #     handles, labels,
    #     loc='upper center',          # top center below title
    #     bbox_to_anchor=(0.5, 0.05), # position outside bottom
    #     ncol=5,                      # number of legend columns
    #     frameon=True,                # draw a box around legend
    #     framealpha=1.0,              # solid white background
    #     facecolor='white',
    #     edgecolor='#D6D6D6',
    #     fontsize=8.5,
    #     columnspacing=1.2,
    #     handlelength=1.5,
    #     handletextpad=0.5,
    #     borderpad=0.3
    # )
    
    # replace MatrixDiT to FrameDiT in legend
    new_labels = []
    for label in labels:
        if label == "MatrixDiT-G":
            new_labels.append("FrameDiT-G")
        elif label == "MatrixDiT-H":
            new_labels.append("FrameDiT-H")
        else:
            new_labels.append(label)
    labels = new_labels
    legend = fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.05),
        ncol=5,
        frameon=True,
        framealpha=1.0,
        facecolor='white',
        edgecolor='#D6D6D6',
        fontsize=10,
        columnspacing=1.2,
        handlelength=1.5,
        handletextpad=0.5,
        borderpad=0.3
    )
    legend.get_frame().set_linewidth(0.4)
    # legend.get_frame().set_linewidth(0.5)

    fig.tight_layout(pad=0.5, w_pad=0.5)
    plt.subplots_adjust(bottom=0.22)
    for ext in [".pdf", ".png"]:
        fig.savefig(save_path.with_suffix(ext), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {save_path}.pdf / .png with colored OOM markers")


# --------------------------------------------------------------------
# Run
# --------------------------------------------------------------------
if __name__ == "__main__":
    plot_metrics_vs_frames(Path("./metrics_vs_frames"))

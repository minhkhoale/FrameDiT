# benchmark_dit_speed.py
import sys
import os
print('os.getcwd()', os.getcwd())
sys.path.append('..')  # to import from parent dir
import time
from statistics import mean, median
import numpy as np
from typing import Tuple, Optional
from omegaconf import OmegaConf
import torch
from torch import nn
from models import get_models
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
from pathlib import Path
import seaborn as sns
import matplotlib.patches as patches

def set_cvpr_style():
    """
    Set consistent publication-quality (CVPR-style) plotting parameters.
    Custom palette matching the reference figure.
    """
    custom_palette = {
        "Local Factorized": "#B75C00",
        "MatrixDiT-G": "#3A5BCD",
        "MatrixDiT-H": "#2D8A4E",
        "Full 3D": "#D282D2",
    }

    sns.set_theme(context="paper", style="whitegrid")
    sns.set_palette(list(custom_palette.values()))

    plt.rcParams.update({
        "figure.figsize": (5.0, 3.3),
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 0.1,
        "lines.markersize": 0.0,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "figure.autolayout": False,
    })

    return custom_palette

# def add_subplot_borders(fig, axes, color="black", linewidth=0.8, pad=0.015):
#     """
#     Add black rectangular borders around each subplot.
#     Works with both single axis and list/array of axes.
#     """
#     if not isinstance(axes, (list, np.ndarray)):
#         axes = [axes]
#     for ax in axes:
#         # Get position in figure coordinates
#         pos = ax.get_position()
#         rect = patches.Rectangle(
#             (pos.x0 - pad, pos.y0 - pad),
#             pos.width + 2*pad,
#             pos.height + 2*pad,
#             transform=fig.transFigure,
#             clip_on=False,
#             lw=linewidth,
#             edgecolor=color,
#             facecolor="none"
#         )
#         fig.patches.append(rect)


def beautify_ax(ax, x_label, y_label, title=None, logy=False):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))

    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)

    ax.tick_params(
        axis='both',
        which='major',
        labelsize=10,      # smaller tick labels
        pad=0,          # closer to the axis
        length=3.5,
        width=0.8,
    )

    # Adjust distance between axis and label text
    ax.yaxis.labelpad = 2     # y label closer
    ax.xaxis.labelpad = 2

    # ax.xaxis.set_minor_locator(None)
    # ax.yaxis.set_minor_locator(None)


def plot_metric_lines(df, y_col, out_basepath: Path, title: str, y_label: str, logy=False):
    palette = set_cvpr_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.8), dpi=400)

    for m in df['model'].unique():
        sdf = df[df['model'] == m].sort_values('num_frames')
        ax.plot(sdf['num_frames'], sdf[y_col], marker='o', label=m, color=palette[m])

    beautify_ax(ax, "Video length", y_label, title, logy)
    ax.legend(loc='upper left', frameon=False)
    fig.tight_layout(pad=0.2)

    for ext in (".pdf", ".png"):
        fig.savefig(out_basepath.with_suffix(ext), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {out_basepath}.pdf and .png")


def plot_combined_three(df: pd.DataFrame, save_path: Path):
    """
    Create one clean 3-panel figure for FLOPs, Latency, Memory vs #Frames.
    Ideal for CVPR figures.
    """
    custom_palette = set_cvpr_style()
    metrics = [
        ("flops_GF", "GFLOPs", "(a)"),
        ("latency_avg_ms", "Latency (ms)", "(b)"),
        ("peak_mem_MB", "Peak Memory (MB)", "(c)"),
    ]

    #fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.6), dpi=400, sharex=True)
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9), dpi=400, sharex=True)

    # palette = sns.color_palette("colorblind")

    for i, (y_col, y_label, tag) in enumerate(metrics):
        ax = axes[i]
        if df[y_col].isna().all():
            ax.set_visible(False)
            continue

        for j, m in enumerate(df['model'].unique()):
            sdf = df[df['model'] == m].sort_values('num_frames')
            ax.plot(
                sdf['num_frames'], sdf[y_col],
                marker='o', linewidth=1.5,
                label=m if i == 1 else None,  # legend once (middle)
                color=custom_palette[m]
            )

        beautify_ax(ax, "Video length" if i == 1 else "", y_label, logy=False)
        #ax.text(0.93, 0.04, tag, transform=ax.transAxes, fontsize=10, va='top', ha='left')

        # if i == 1:
        #     ax.legend(loc='upper left', frameon=False)

        if i == 1:
            # Create legend outside figure in a small framed box
            handles, labels = ax.get_legend_handles_labels()
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
                loc='upper center',          # top center below title
                bbox_to_anchor=(0.5, 0), # position outside bottom
                ncol=5,                      # number of legend columns
                frameon=True,                # draw a box around legend
                framealpha=1.0,              # solid white background
                facecolor='white',
                edgecolor='#D6D6D6',
                fontsize=10,
                columnspacing=1.2,
                handlelength=1.5,
                handletextpad=0.5,
                borderpad=0.3
            )
            legend.get_frame().set_linewidth(0.4)
        
        ax.margins(x=0)
        ax.set_xlim(df['num_frames'].min(), df['num_frames'].max())
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color("black")

    #for ax in plt.  axes:


    # add_subplot_borders(fig, axes, color="black", linewidth=0.8, pad=0.01)

    fig.tight_layout(pad=0.6, w_pad=1.2)
    for ext in (".pdf", ".png"):
        fig.savefig(save_path.with_suffix(ext), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved combined figure: {save_path}.pdf and .png")


def plot_combined_three_with_range(df: pd.DataFrame, df_matlatte: pd.DataFrame, df_fusedmatlatte: pd.DataFrame, save_path: Path):
    """
    Create one clean 3-panel figure for FLOPs, Latency, Memory vs #Frames.
    Shows MatLatte as a shaded area (min-max range) across configurations.
    """
    custom_palette = set_cvpr_style()
    metrics = [
        ("flops_GF", "GFLOPs", "(a)"),
        ("latency_avg_ms", "Latency (ms)", "(b)"),
        ("peak_mem_MB", "Peak Memory (MB)", "(c)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9), dpi=400, sharex=True)

    for i, (y_col, y_label, tag) in enumerate(metrics):
        ax = axes[i]
        if df[y_col].isna().all():
            ax.set_visible(False)
            continue

        # First, plot the MatLatte range as shaded area
        if not df_matlatte.empty and y_col in df_matlatte.columns:
            matlatte_grouped = df_matlatte.groupby('num_frames')[y_col].agg(['min', 'max']).reset_index()
            ax.fill_between(
                matlatte_grouped['num_frames'],
                matlatte_grouped['min'],
                matlatte_grouped['max'],
                alpha=0.6,
                color=custom_palette["MatrixDiT-G"],
                label='MatrixDiT-G' if i == 1 else None,
                linewidth=0
            )
        
        if not df_fusedmatlatte.empty and y_col in df_fusedmatlatte.columns:
            fusedmatlatte_grouped = df_fusedmatlatte.groupby('num_frames')[y_col].agg(['min', 'max']).reset_index()
            ax.fill_between(
                fusedmatlatte_grouped['num_frames'],
                fusedmatlatte_grouped['min'],
                fusedmatlatte_grouped['max'],
                alpha=0.6,
                color=custom_palette["MatrixDiT-H"],
                label='MatrixDiT-H' if i == 1 else None,
                linewidth=0
            ) 

        # Then plot the discrete models
        for j, m in enumerate(df['model'].unique()):
            sdf = df[df['model'] == m].sort_values('num_frames')
            ax.plot(
                sdf['num_frames'], sdf[y_col],
                marker='o', linewidth=2,
                label=m if i == 1 else None,
                color=custom_palette[m]
            )

        beautify_ax(ax, "Video length", y_label, logy=False)
        if y_label.lower().startswith("latency"):
            ax.ticklabel_format(axis='y', style='plain')   # disable scientific notation
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
            ax.yaxis.get_offset_text().set_visible(False)
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=False))

        if i == 1:
            # Create legend outside figure
            handles, labels = ax.get_legend_handles_labels()
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
                bbox_to_anchor=(0.5, 0),
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
        
        ax.margins(x=0)
        ax.set_xlim(df['num_frames'].min(), df['num_frames'].max())
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color("black")

    fig.tight_layout(pad=0.6, w_pad=1.2)
    for ext in (".pdf", ".png"):
        fig.savefig(save_path.with_suffix(ext), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved combined figure: {save_path}.pdf and .png")


def plot_combined_four_with_speedup(
    df: pd.DataFrame,
    df_matlatte: pd.DataFrame,
    df_fusedmatlatte: pd.DataFrame,
    save_path: Path
):
    """
    Create a 4-panel CVPR-style figure:
        (a) FLOPs vs Frames
        (b) Latency vs Frames
        (c) Memory vs Frames
        (d) Speed-Up vs Frames (↑, baseline = Full 3D)
    Includes shaded ranges for MatrixDiT-G/H.
    """
    custom_palette = set_cvpr_style()
    metrics = [
        ("flops_GF", "GFLOPs", "(a)"),
        ("latency_avg_ms", "Latency (ms)", "(b)"),
        ("peak_mem_MB", "Peak Memory (MB)", "(c)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(11.2, 2.9), dpi=400, sharex=True)
    speedup_ax = axes[3]

    # -------------------------
    # (a)–(c): same as before
    # -------------------------
    for i, (y_col, y_label, tag) in enumerate(metrics):
        ax = axes[i]
        if df[y_col].isna().all():
            ax.set_visible(False)
            continue

        # Shaded MatLatte / FusedMatLatte ranges
        for range_df, color_key, label_name in [
            (df_matlatte, "MatrixDiT-G", "MatrixDiT-G Range"),
            (df_fusedmatlatte, "MatrixDiT-H", "MatrixDiT-H Range"),
        ]:
            if not range_df.empty and y_col in range_df.columns:
                grouped = range_df.groupby("num_frames")[y_col].agg(["min", "max"]).reset_index()
                ax.fill_between(
                    grouped["num_frames"], grouped["min"], grouped["max"],
                    alpha=0.35,
                    color=custom_palette[color_key],
                    label=label_name if i == 1 else None,
                    linewidth=0,
                )

        # Plot each model line
        for m in df["model"].unique():
            sdf = df[df["model"] == m].sort_values("num_frames")
            ax.plot(
                sdf["num_frames"], sdf[y_col],
                marker="o", linewidth=1.5,
                label=m if i == 1 else None,
                color=custom_palette[m],
            )

        beautify_ax(ax, "Video length" if i == 1 else "", y_label, logy=False)
        ax.margins(x=0)
        ax.set_xlim(df["num_frames"].min(), df["num_frames"].max())

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color("black")

    # -------------------------
    # (d): Speed-Up vs Frames
    # -------------------------
    base_df = df[df["model"] == "Full 3D"][["num_frames", "latency_avg_ms"]].set_index("num_frames")
    df_speed = df
    df_speed["SpeedUp"] = df_speed.apply(
        lambda r: base_df.loc[r["num_frames"], "latency_avg_ms"] / r["latency_avg_ms"]
        if r["num_frames"] in base_df.index else np.nan,
        axis=1
    )
    df_speed = df_speed.dropna(subset=["SpeedUp"])

    # df_speed.to_csv('df_speedup.csv', index=False)
    for m in df_speed["model"].unique():
        sdf = df_speed[df_speed["model"] == m].sort_values("num_frames")
        print('sdf', sdf)
        speedup_ax.plot(
            sdf["num_frames"], sdf["SpeedUp"],
            marker="o", linewidth=1.5,
            label=m,
            color=custom_palette.get(m, None),
        )
    for range_df, color_key, label_name in [
        (df_matlatte, "MatrixDiT-G", "MatrixDiT-G Range"),
        (df_fusedmatlatte, "MatrixDiT-H", "MatrixDiT-H Range"),
    ]:
        if not range_df.empty:
            # Compute SpeedUp for each configuration
            sdf = range_df.copy()
            sdf["SpeedUp"] = sdf.apply(
                lambda r: base_df.loc[r["num_frames"], "latency_avg_ms"] / r["latency_avg_ms"]
                if r["num_frames"] in base_df.index else np.nan,
                axis=1
            )
            grouped = sdf.groupby("num_frames")["SpeedUp"].agg(["min", "max"]).reset_index()
            speedup_ax.fill_between(
                grouped["num_frames"],
                grouped["min"],
                grouped["max"],
                alpha=0.35,
                color=custom_palette[color_key],
                label=label_name if label_name not in speedup_ax.get_legend_handles_labels()[1] else None,
                linewidth=0
        )

    # speedup_ax.axhline(1.0, color=custom_palette['Full 3D'], linestyle="-", linewidth=1.0)
    beautify_ax(speedup_ax, "Video length", "Speed-Up vs Full 3D (↑)", logy=False)
    speedup_ax.set_xscale("log", base=2)

    # speedup_ax.set_xlim(df["num_frames"].min(), df["num_frames"].max())
    speedup_ax.set_ylim(bottom=0)

    for spine in speedup_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")

    # -------------------------
    # Shared legend (once)
    # -------------------------
    handles, labels = speedup_ax.get_legend_handles_labels()
    legend = fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=5,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#D6D6D6",
        fontsize=10,
        columnspacing=1.2,
        handlelength=1.5,
        handletextpad=0.5,
        borderpad=0.3,
    )
    legend.get_frame().set_linewidth(0.4)

    fig.tight_layout(pad=0.6, w_pad=1.2)
    for ext in (".pdf", ".png"):
        fig.savefig(save_path.with_suffix(ext), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved combined 4-panel figure (with speed-up): {save_path}.pdf and .png")

# -----------------------------
# FLOPs helpers (fvcore -> thop -> ptflops)
# -----------------------------
def try_fvcore_flops(model: nn.Module, inputs: torch.Tensor, timesteps: torch.Tensor) -> Optional[int]:
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        def model_wrap(*args):
            return model(args[0], timesteps)
        model = model_wrap
        with torch.no_grad():
            flops = FlopCountAnalysis(model, inputs).total()
        return int(flops)
    except Exception as e:
        print('fvcore failed:', e)
        return None

def try_thop_flops(model: nn.Module, inputs: torch.Tensor, timesteps: torch.Tensor) -> Optional[int]:
    try:
        from thop import profile
        model.eval()
        with torch.no_grad():
            flops, _ = profile(model, inputs=(inputs, timesteps), verbose=False)
        return int(flops)
    except Exception as e:
        print('thop failed:', e)
        return None

def try_ptflops_flops(model: nn.Module, inputs: torch.Tensor, timesteps: torch.Tensor) -> Optional[int]:
    try:
        from ptflops import get_model_complexity_info
        # ptflops needs a tuple of input sizes (no batch dim)
        inp = inputs
        if inp.dim() == 5:
            # e.g., [B, C, T, H, W] or [B, T, C, H, W] -> strip batch
            s = tuple(inp.shape[1:])
        else:
            s = tuple(inp.shape[1:])
        
        # add timesteps as extra input
        model.eval()
        class ModelWrap(nn.Module):
            def __init__(self, model, timesteps):
                super().__init__()
                self.model = model
                self.timesteps = timesteps
            def forward(self, x):
                return self.model(x, self.timesteps)
        model = ModelWrap(model, timesteps)
        model.eval()

        # Monkey: ptflops creates its own input; to keep close, ignore AMP here.
        macs, params = get_model_complexity_info(model, s, as_strings=False, backend='aten',
                                                 print_per_layer_stat=False, verbose=False)
        # MACs -> FLOPs (1 MAC ~ 2 FLOPs for real-valued ops)
        return int(macs * 2)
    except Exception as e:
        print('ptflops failed:', e)
        return None

def count_flops(model: nn.Module, inputs: torch.Tensor, timesteps: torch.Tensor) -> Optional[int]:
    # Prefer fvcore (robust on aten ops), then thop, then ptflops
    #for fn in (try_ptflops_flops):
    fl = try_ptflops_flops(model, inputs, timesteps)
    if fl is not None:
        return fl
    return None

# -----------------------------
# Timing & memory
# -----------------------------
@torch.inference_mode()
def benchmark_forward(
    model: nn.Module,
    inputs: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    device: str = "cuda",
    amp: bool = True,
    warmup: int = 10,
    iters: int = 50
):
    model = model.to(device).eval()
    inputs = inputs.to(device, non_blocking=True)
    timesteps = timesteps.to(device, non_blocking=True)

    # Warm-up (catches lazy init & cudnn autotune)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    scaler_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (amp and device.startswith("cuda")) else torch.no_grad()
    with scaler_ctx:
        for _ in range(max(1, warmup)):
            _ = model(inputs, timesteps)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    times = []
    with scaler_ctx:
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(inputs, timesteps)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    avg = mean(times)
    p50 = median(times)
    p95 = sorted(times)[int(0.95 * (len(times) - 1))]

    peak_mem = None
    if device.startswith("cuda"):
        peak_mem = torch.cuda.max_memory_allocated()  # bytes

    return {
        "latency_avg_s": avg,
        "latency_p50_s": p50,
        "latency_p95_s": p95,
        "peak_mem_bytes": peak_mem,
        "iters": iters,
    }

def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# -----------------------------
# Convenience runner
# -----------------------------
def run_all(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    *,
    input_layout: str = "BCTHW",   # or "BTCHW"
    device: str = "cuda",
    amp: bool = True,
    warmup: int = 10,
    iters: int = 50
):
    """
    input_shape includes batch dim.
    Example shapes:
      - BCTHW: (B, C, T, H, W)   e.g., (1, 4, 16, 32, 32) latent video
      - BTCHW: (B, T, C, H, W)   e.g., (1, 16, 3, 256, 256) RGB video
    """
    assert input_layout in ("BCTHW", "BTCHW")
    x = torch.randn(*input_shape)
    t = torch.randint(0, 1000, (input_shape[0],), dtype=torch.long)  # dummy timesteps

    # If model expects BCTHW but user provides BTCHW (or vice versa), handle outside.
    # Here we assume your model matches the chosen layout.

    # FLOPs (use FP32 for counting to avoid AMP casting issues)
    model_cpu = model.to("cpu").eval()
    flops = count_flops(model_cpu, x, t)  # stays on CPU; FLOPs are device-agnostic

    #reset peak mem
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    # Timing & memory (on target device)
    results = benchmark_forward(model, x, t, device=device, amp=amp, warmup=warmup, iters=iters)

    params = num_params(model)
    out = {
        "params": params,
        "flops": flops,
        **results
    }
    # Pretty print
    def fmt_bytes(b):
        if b is None: return None
        for unit in ["B","KB","MB","GB","TB"]:
            if b < 1024: return f"{b:.2f} {unit}"
            b /= 1024
        return f"{b:.2f} PB"

    print("==== Benchmark ====")
    print(f"Params: {params/1e6:.3f} M")
    if flops is not None:
        print(f"FLOPs (forward): {flops/1e9:.3f} GFLOPs")
    else:
        print("FLOPs: could not be computed with fvcore/thop/ptflops (custom ops?).")
    print(f"Latency (avg):  {results['latency_avg_s']*1000:.2f} ms")
    print(f"Latency (p50):  {results['latency_p50_s']*1000:.2f} ms")
    print(f"Latency (p95):  {results['latency_p95_s']*1000:.2f} ms")
    if results["peak_mem_bytes"] is not None:
        print(f"Peak GPU mem:   {fmt_bytes(results['peak_mem_bytes'])}")
    print(f"AMP enabled:    {amp}")
    print(f"Device:         {device}")
    print("===================")
    return out

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Replace with your DiT variant
    results = []
    set_cvpr_style()

    # latent_size = 16
    # num_classes = None
    # # all_num_frames = [8, 16, 32, 64, 128, 256]
    # all_num_frames = [256]
    # learn_sigma=False
    # in_channels=4
    # extras=1
    # model_names = [
    #     'MatLatte-M/1-64/2', 'MatLatte-M/1-128/2', 'MatLatte-M/1-256/2', 
    #     'MatLatte-M/2-64/2', 'MatLatte-M/2-128/2', 'MatLatte-M/2-256/2',
    #     'MatLatte-M/4-64/2', 'MatLatte-M/4-128/2', 'MatLatte-M/4-256/2',
    #     'MatLatte-M/8-64/2', 'MatLatte-M/8-128/2', 'MatLatte-M/8-256/2',
    #     'MatLatte-M/16-64/2', 'MatLatte-M/16-128/2', 'MatLatte-M/16-256/2',
    #     'MatLatte-M/32-64/2', 'MatLatte-M/32-128/2', 'MatLatte-M/32-256/2',
    #     'MatLatte-M/64-64/2', 'MatLatte-M/64-128/2', 'MatLatte-M/64-256/2',
    #     'Latte-M/2', 
    #     'FusedMatLatte-M/1-64/2-concat', 'FusedMatLatte-M/1-128/2-concat', 'FusedMatLatte-M/1-256/2-concat',
    #     'FusedMatLatte-M/2-64/2-concat', 'FusedMatLatte-M/2-128/2-concat', 'FusedMatLatte-M/2-256/2-concat',
    #     'FusedMatLatte-M/4-64/2-concat', 'FusedMatLatte-M/4-128/2-concat', 'FusedMatLatte-M/4-256/2-concat',
    #     'FusedMatLatte-M/8-64/2-concat', 'FusedMatLatte-M/8-128/2-concat', 'FusedMatLatte-M/8-256/2-concat',
    #     'FusedMatLatte-M/16-64/2-concat', 'FusedMatLatte-M/16-128/2-concat', 'FusedMatLatte-M/16-256/2-concat',
    #     'FusedMatLatte-M/32-64/2-concat', 'FusedMatLatte-M/32-128/2-concat', 'FusedMatLatte-M/32-256/2-concat',
    #     'FusedMatLatte-M/64-64/2-concat', 'FusedMatLatte-M/64-128/2-concat', 'FusedMatLatte-M/64-256/2-concat',
    # ]
    # for model_name in model_names:
    #     for num_frames in all_num_frames:
    #         print(f"\n=== Benchmarking {model_name} | n_frames={num_frames} ===")
    #         config = {
    #             'model': model_name,
    #             'latent_size': latent_size,
    #             'num_classes': num_classes,
    #             'num_frames': num_frames,
    #             'learn_sigma': learn_sigma,
    #             'in_channels': in_channels,
    #             'extras': extras,
    #         }
    #         args = OmegaConf.create(config)


    #         model = get_models(args)

    #         # Example input: latent video [B,C,T,H,W]
    #         # Adjust to your DiT input shape (e.g., (1, 4, 16, 32, 32) or (1, 3, 16, 256, 256))
    #         input_shape = (1, num_frames, in_channels, latent_size, latent_size)  # BCTHW

    #         bench = run_all(
    #             model,
    #             input_shape,
    #             input_layout="BTCHW",
    #             device="cuda" if torch.cuda.is_available() else "cpu",
    #             amp=True,          # set False if you want pure FP32 timing
    #             warmup=10,
    #             iters=50,
    #         )
    #         results.append({
    #             "model": model_name,
    #             "num_frames": num_frames,
    #             "params_M": bench["params"] / 1e6,
    #             "flops_GF": (bench["flops"] / 1e9) if bench["flops"] is not None else None,
    #             "latency_avg_ms": bench["latency_avg_s"] * 1000,
    #             "latency_p50_ms": bench["latency_p50_s"] * 1000,
    #             "latency_p95_ms": bench["latency_p95_s"] * 1000,
    #             "peak_mem_MB": (bench["peak_mem_bytes"] / (1024**2)) if bench["peak_mem_bytes"] is not None else None,
    #         })

    csv_path = './speed_vs_frames_1.csv'
    #df = pd.DataFrame(results).sort_values(["model", "num_frames"])
    #df.to_csv(csv_path, index=False)
    save_dir = Path(".")
    df = pd.read_csv(csv_path)

    # change model name for better legend
    # df['model'] = df['model'].replace({
    #     'Latte-M/2': 'Local Factorized',
    #     'MatLatte-M/2': 'MatrixDiT-G',
    #     'DiT3D-M/2': 'Full',
    #     'FusedMatLatte-M/2': 'MatrixDiT-U',
    # })

    # remove 256 frames
    df = df[df['num_frames'] != 256]

    df['model'] = df['model'].apply(lambda x: 'MatrixDiT-G' if x.startswith('MatLatte') else x)
    df['model'] = df['model'].apply(lambda x: 'MatrixDiT-H' if x.startswith('FusedMatLatte') else x)

    # Latte-M/2 to Local Factorized
    # DiT3D-M/2 to Full 3D
    df['model'] = df['model'].replace({
        'Latte-M/2': 'Local Factorized',
        'DiT3D-M/2': 'Full 3D',
    }) 

    matlatte_df = df[df['model'] == 'MatrixDiT-G']
    fusedmatlatte_df = df[df['model'] == 'MatrixDiT-H']
    df = df[~df['model'].isin(['MatrixDiT-G', 'MatrixDiT-H'])]

    # df_flops = df.dropna(subset=["flops_GF"])
    # if not df_flops.empty:
    #     plot_metric_lines(
    #         df_flops, "flops_GF",
    #         save_dir / "flops_vs_frames",
    #         title="FLOPs vs Number of Frames (forward)",
    #         y_label="FLOPs (GFLOPs)",
    #         logy=False
    #     )
    #     print(f"Saved plot: {save_dir / 'flops_vs_frames.png'}")
    # else:
    #     print("FLOPs unavailable for all entries; skipping FLOPs plot.")

    # # 2) Latency (avg) vs frames
    # plot_metric_lines(
    #     df, "latency_avg_ms",
    #     save_dir / "latency_vs_frames",
    #     title="Average Forward Latency vs Number of Frames",
    #     y_label="Latency (ms)",
    #     logy=False
    # )
    # print(f"Saved plot: {save_dir / 'latency_vs_frames.png'}")

    # # 3) Peak memory vs frames
    # df_mem = df.dropna(subset=["peak_mem_MB"])
    # if not df_mem.empty:
    #     plot_metric_lines(
    #         df_mem, "peak_mem_MB",
    #         save_dir / "memory_vs_frames",
    #         title="Peak GPU Memory vs Number of Frames",
    #         y_label="Peak Memory (MB)",
    #         logy=False
    #     )
    #     print(f"Saved plot: {save_dir / 'memory_vs_frames.png'}")
    # else:
    #     print("Peak memory unavailable; skipping memory plot.")

    plot_combined_three_with_range(df, matlatte_df, fusedmatlatte_df, save_dir / "speed_vs_frames")

    # plot_combined_four_with_speedup(df, matlatte_df, fusedmatlatte_df, save_dir / "speed_vs_frames")

# benchmark_dit_speed.py
import sys
import os
print('os.getcwd()', os.getcwd())
sys.path.append('..')  # to import from parent dir
import time
from statistics import mean, median
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
from pprint import pprint
import seaborn as sns

def set_pub_style():
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

def plot_metric_lines(df, y_col, out_basepath: Path, title: str, y_label: str, logy=False):
    fig, ax = plt.subplots(figsize=(3.6, 2.8), dpi=400)  # smaller figure
    for m in df['model'].unique():
        sdf = df[df['model'] == m].sort_values('num_frames')
        ax.plot(sdf['num_frames'], sdf[y_col], marker='o', label=m, linewidth=1.8)

    beautify_ax(ax, x_label='Number of tokens per frame', y_label=y_label, title=title, logy=logy)

    # ✅ smaller, tighter legend INSIDE top-right corner
    ax.legend(
        loc='upper left',
        frameon=True,
        framealpha=0.9,
        facecolor='white',
        edgecolor='0.8',
        fontsize=9,
        handlelength=2.5,
        handletextpad=0.4
    )

    fig.tight_layout(pad=0.2)
    png_path = out_basepath.with_suffix(".png")
    pdf_path = out_basepath.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {png_path} and {pdf_path}")


def plot_combined_three(df: pd.DataFrame, save_path: Path):
    """
    Create one figure with 3 aligned subplots:
    (a) FLOPs, (b) Latency, (c) Memory vs #Frames
    """
    custom_palette = set_pub_style()
    metrics = [
        ("FVD", "FVD↓", "(a)"),
        ("flops_GF", "FLOPs", "(b)"),
        ("latency_avg_ms", "Latency (s)", "(c)"),
        ("peak_mem_MB", "Peak Memory (GB)", "(d)")
    ]
    df['latency_avg_ms'] = df['latency_avg_ms']/1000
    df['peak_mem_MB'] = df['peak_mem_MB']/1024

    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.4), dpi=400, sharex=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (y_col, y_label, sublabel) in enumerate(metrics):
        ax = axes[i]
        # skip metrics missing data
        if df[y_col].isna().all():
            ax.set_visible(False)
            continue

        for j, m in enumerate(df['model'].unique()):
            sdf = df[df['model'] == m].sort_values('num_frames')
            ax.plot(sdf['num_frames'], sdf[y_col],
                    marker=None, linewidth=1.5,
                    label=m if i == 1 else None,  # show legend only once (middle)
                    color=custom_palette[m]
        )
        # beautify_ax(ax, x_label='Number of tokens per frame' if i == 1 else '',
        #             y_label=y_label, title='', logy=False)
        beautify_ax(ax, "Video length", y_label, logy=False)
        # ax.text(0.02, 0.95, sublabel, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')
        if i != 0:
            ax.yaxis.label.set_visible(True)
        if i > 0:
            ax.yaxis.set_tick_params(labelleft=True)
        #ax.set_xticks(frames)
        ax.margins(x=0)
        #ax.set_xlim(frames[0], frames[-1])
        # if i == 1:
        #     ax.legend(loc='upper left', fontsize=9, frameon=True,
        #               framealpha=0.9, handlelength=2.4, handletextpad=0.4)
        ax.set_xticks([16, 32, 64, 128])

    # global formatting
    # fig.suptitle("Efficiency of Transformer Variants with Factorized, Matrix, and Full Attention", fontsize=13)
    #fig.tight_layout(pad=0.6, w_pad=1.6)
    handles, labels = axes[1].get_legend_handles_labels()
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

    png_path = save_path.with_suffix(".png")
    pdf_path = save_path.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"Saved combined figure: {png_path} and {pdf_path}")


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
        def model_wrap(*args):
            return model(args[0], timesteps)
        model = model_wrap
        # Monkey: ptflops creates its own input; to keep close, ignore AMP here.
        macs, params = get_model_complexity_info(model, s, as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        # MACs -> FLOPs (1 MAC ~ 2 FLOPs for real-valued ops)
        return int(macs * 2)
    except Exception as e:
        print('ptflops failed:', e)
        return None

def count_flops(model: nn.Module, inputs: torch.Tensor, timesteps: torch.Tensor) -> Optional[int]:
    # Prefer fvcore (robust on aten ops), then thop, then ptflops
    for fn in (try_fvcore_flops, try_thop_flops, try_ptflops_flops):
        fl = fn(model, inputs, timesteps)
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
    import math
    results = []
    # set_pub_style()

    all_token_size = [64]
    all_latent_size = [int(math.sqrt(ts*4)) for ts in all_token_size]  # assuming 4x compression
    num_classes = None
    num_frames = 16
    learn_sigma=False
    in_channels=4
    extras=1
    model_names = ['Latte-S/2', 'Latte-B/2', 'Latte-L/2', 'Latte-XL/2', 
                    'DiT3D-S/2', 'DiT3D-B/2', 'DiT3D-L/2', 'DiT3D-XL/2',
                   'MatLatte-S/64-256/2', 'MatLatte-B/64-256/2', 'MatLatte-L/64-256/2', 'MatLatte-XL/64-256/2',
                   'FusedMatLatte-S/64-64/2-concat', 'FusedMatLatte-B/64-64/2-concat', 'FusedMatLatte-L/64-64/2-concat', 'FusedMatLatte-XL/64-64/2-concat'
                ]
    # model_names = ['']
    for model_name in model_names:
        for token_size, latent_size in zip(all_token_size, all_latent_size):
            print(f"\n=== Benchmarking {model_name} | token_size={token_size} ===")
            config = {
                'model': model_name,
                'latent_size': latent_size,
                'num_classes': num_classes,
                'num_frames': num_frames,
                'learn_sigma': learn_sigma,
                'in_channels': in_channels,
                'extras': extras,
            }
            args = OmegaConf.create(config)


            model = get_models(args)

            # Example input: latent video [B,C,T,H,W]
            # Adjust to your DiT input shape (e.g., (1, 4, 16, 32, 32) or (1, 3, 16, 256, 256))
            input_shape = (1, num_frames, in_channels, latent_size, latent_size)  # BCTHW

            bench = run_all(
                model,
                input_shape,
                input_layout="BTCHW",
                device="cuda" if torch.cuda.is_available() else "cpu",
                amp=True,          # set False if you want pure FP32 timing
                warmup=10,
                iters=50,
            )
            results.append({
                "model": model_name,
                "token_size": token_size,
                "params_M": bench["params"] / 1e6,
                "flops_GF": (bench["flops"] / 1e9) if bench["flops"] is not None else None,
                "latency_avg_ms": bench["latency_avg_s"] * 1000,
                "latency_p50_ms": bench["latency_p50_s"] * 1000,
                "latency_p95_ms": bench["latency_p95_s"] * 1000,
                "peak_mem_MB": (bench["peak_mem_bytes"] / (1024**2)) if bench["peak_mem_bytes"] is not None else None,
            })

    pprint(results)
    exit(0)

    csv_path = './speed_fvd_vs_frames.csv'
    # df = pd.DataFrame(results)
    # print(df)
    # df = df.sort_values(["model", "token_size"])
    # df.to_csv(csv_path, index=False)
    save_dir = Path(".")
    df = pd.read_csv(csv_path)

    # change model name for better legend
    df['model'] = df['model'].replace({
        'Latte-M/2': 'Local Factorized',
        'MatLatte-M/64-256/2': 'MatrixDiT-G',
        'DiT3D-M/2': 'Full 3D',
        'FusedMatLatte-M/64-256/2-concat': 'MatrixDiT-H',
    })

    df_flops = df.dropna(subset=["flops_GF"])
    # if not df_flops.empty:
    #     plot_metric_lines(
    #         df_flops, "flops_GF",
    #         save_dir / "flops_vs_tokens",
    #         title="FLOPs vs Number of Tokens (forward)",
    #         y_label="FLOPs (GFLOPs)",
    #         logy=False
    #     )
    #     print(f"Saved plot: {save_dir / 'flops_vs_tokens.png'}")
    # else:
    #     print("FLOPs unavailable for all entries; skipping FLOPs plot.")

    # # 2) Latency (avg) vs tokens
    # plot_metric_lines(
    #     df, "latency_avg_ms",
    #     save_dir / "latency_vs_tokens",
    #     title="Average Forward Latency vs Number of Tokens",
    #     y_label="Latency (ms)",
    #     logy=False
    # )
    # print(f"Saved plot: {save_dir / 'latency_vs_tokens.png'}")

    # # 3) Peak memory vs tokens
    # df_mem = df.dropna(subset=["peak_mem_MB"])
    # if not df_mem.empty:
    #     plot_metric_lines(
    #         df_mem, "peak_mem_MB",
    #         save_dir / "memory_vs_tokens",
    #         title="Peak GPU Memory vs Number of Tokens",
    #         y_label="Peak Memory (MB)",
    #         logy=False
    #     )
    #     print(f"Saved plot: {save_dir / 'memory_vs_tokens.png'}")
    # else:
    #     print("Peak memory unavailable; skipping memory plot.")

    plot_combined_three(df, save_dir / "combined_metrics_vs_frames")
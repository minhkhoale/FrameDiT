import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fvd_600k = {
    '0.25': [4018, 1063, 234, 135, 113, 102, 96, 92, 90, 86, 85, 82, 80, 78, 77, 76, 76, 75, 74, 73],
    '0.5':  [4252, 1056, 279, 147, 108,  94, 89, 85, 83, 80, 79, 78, 76, 72, 73, 72, 71, 70, 71, 70],
    '0.75': [4080, 1133, 280, 146, 110,  94, 91, 84, 82, 79, 78, 76, 74, 73, 75, 75, 74, 72, 70, 69],
    '1.0':  [3799, 1134, 306, 146, 116,  90, 88, 83, 80, 77, 73, 72, 71, 70, 67, 69, 69, 68, 68, 66],
}

# -----------------------------
# X axis: steps (10k, 20k, ..., 150k)
# -----------------------------
skip_n = 5
steps = np.arange(10, 10 * len(fvd_600k['0.25']) + 1, 10)[skip_n:]  # [60, 70, ..., 150]

# -----------------------------
# Style setup (CVPR-style palette)
# -----------------------------
custom_palette = {
    "0.25": "#B75C00",   # brown / DFoT
    "0.5":  "#3A5BCD",   # blue / HG-v
    "0.75": "#2D8A4E",   # green / HG-f
    "1.0":  "#D282D2",  # purple / SD
}

sns.set_theme(context="paper", style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.linewidth": 0.7,     # thin border
    "lines.linewidth": 1.8,
    "lines.markersize": 4.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6.2, 3.8))

for label, values in fvd_600k.items():
    plt.plot(
        steps, values[skip_n:],
        marker='o',
        linewidth=1.8,
        markersize=0,
        color=custom_palette[label],
        label=label.upper()
    )

# -----------------------------
# Axes and grid
# -----------------------------
plt.xlabel("Training Steps", fontsize=13, labelpad=5)
plt.ylabel("FVD↓", fontsize=13, labelpad=5)

plt.xticks(steps[::2], [f"{s}k" for s in steps[::2]], fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.grid(axis='x', visible=False)

# Thin black spines around the full box
for spine in ['bottom', 'left', 'top', 'right']:
    plt.gca().spines[spine].set_visible(True)
    plt.gca().spines[spine].set_color("black")
    plt.gca().spines[spine].set_linewidth(0.5)

    

# Compact tick label spacing
# plt.tick_params(axis='both', which='major', pad=1.5, length=3, width=0.7)
plt.tick_params(
    axis='x', which='major',
    direction='out',
    length=3,        # small vertical tick (like '|')
    width=1.0,
    color='black',
    pad=2
)

# -----------------------------
# Legend (keep original inside style)
# -----------------------------
plt.legend(
    title="Data Size",
    fontsize=13,
    title_fontsize=13,
    loc="upper right",
    frameon=False  # keep legend clean, no box
)

# set x limits
plt.xlim(steps[0], steps[-1])

# -----------------------------
# Layout & Save
# -----------------------------
plt.tight_layout(pad=0.4)
plt.savefig("ablation_data_size.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
plt.savefig("ablation_data_size.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close()
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# -----------------------------
# New Data
# -----------------------------
# This data appears to be from Table 4:
# X-axis: GFLOPs (inferred from the values)
# Y-axis: FVD (from Table 4)
data = {
    341.601718272: 72.95, # N_qk = 1
    342.032683008: 72.91, # N_qk = 2
    342.89461248:  72.36, # N_qk = 4
    344.618471424: 72.41, # N_qk = 8
    348.066189312: 71.28, # N_qk = 16
    354.961625088: 71.91, # N_qk = 32
    368.75249664:  70.31  # N_qk = 64
}

# Extract X (GFLOPs) and Y (FVD) values
x_values = list(data.keys())
y_values = list(data.values())

# -----------------------------
# Style setup (CVPR-style palette)
# -----------------------------
# Using a single color from the original palette for the single line
line_color = "#3A5BCD" # Blue

sns.set_theme(context="paper", style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.linewidth": 0.7,      # thin border
    "lines.linewidth": 1.8,
    "lines.markersize": 4.5,    # Make markers visible for ablation plot
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

# Plot the single line of data
plt.scatter(
    x_values, y_values,
    marker='o',             # Use markers to show the discrete ablation points
    # linewidth=1.8,
    # markersize=4.5,         # Use the default markersize
    color=line_color,
    label="MatrixDiT-G"     # Label for the legend
)

# plot individual data points for Latte: 317.662138368, 75.7
plt.scatter(
    [317.662138368], [75.7],
    marker='o',
    #linewidth=1.8,
    #markersize=4.5,
    color="#FF5733",  # Different color for Latte
    label="Latte"
)
# plot individual data points for DiT3D: 361.2309504, 71.7
plt.scatter(
    [361.2309504], [71.7],
    marker='o',
    #linewidth=1.8,
    #markersize=4.5,
    color="#33FF57",  # Different color for DiT3D
    label="DiT3D"
)

# -----------------------------
# Axes and grid
# -----------------------------
plt.xlabel("GFLOPs", fontsize=13, labelpad=5) # Changed label
plt.ylabel("FVD↓", fontsize=13, labelpad=5)

# Use default tick formatting for these values, but style them
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.grid(axis='x', visible=False)

# Thin black spines around the full box
for spine in ['bottom', 'left', 'top', 'right']:
    plt.gca().spines[spine].set_visible(True)
    plt.gca().spines[spine].set_color("black")
    plt.gca().spines[spine].set_linewidth(0.5)

plt.tick_params(
    axis='x', which='major',
    direction='out',
    length=3,
    width=1.0,
    color='black',
    pad=2
)
plt.tick_params(
    axis='y', which='major',
    direction='out',
    length=3,
    width=1.0,
    color='black',
    pad=2
)

# -----------------------------
# Legend
# -----------------------------
# Simplified legend for a single line
plt.legend(
    loc='upper center',          # top center below title
    #bbox_to_anchor=(0.5, 1), # position outside bottom
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

# set x limits with a bit of padding
# plt.xlim(min(x_values) - 5, max(x_values) + 5)

# -----------------------------
# Layout & Save
# -----------------------------
plt.tight_layout(pad=0.4)
plt.savefig("ablation_gflops_fvd.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
plt.savefig("ablation_gflops_fvd.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close()

print("Plot 'ablation_gflops_fvd.png' saved successfully.")
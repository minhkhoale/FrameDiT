import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fvd_600k = {
    's': [8798, 1824, 1766, 734, 386, 289, 229, 200, 178, 158, 147, 143, 132, 122, 123],
    'b': [7932, 2095, 1008, 355, 174, 129, 105, 96, 93, 85, 89, 82, 80, 77, 79],
    'l': [7878, 1749, 562, 158, 108, 82, 80, 78, 73, 66, 67, 67, 61, 62, 63],
    'xl': [7459, 1796, 589, 146, 96, 82, 80, 74, 68, 64, 63, 62, 60, 59, 59]
}

# -----------------------------
# X axis: steps (10k, 20k, ..., 150k)
# -----------------------------
skip_n = 5
steps = np.arange(10, 10 * len(fvd_600k['s']) + 1, 10)[skip_n:]  # [60, 70, ..., 150]

# -----------------------------
# Plot
# -----------------------------
sns.set_context("paper")
sns.set_style("whitegrid")
sns.set_palette("colorblind")  # color-blind safe palette

plt.figure(figsize=(6.4, 4.3))  # typical CVPR figure aspect ratio

for label, values in fvd_600k.items():
    plt.plot(
        steps, values[skip_n:], 
        marker='o', linewidth=2.2, markersize=5.5, 
        label=label.upper()
    )

# Axis formatting
plt.xticks(steps, [f"{s}k" for s in steps], fontsize=11)
plt.yticks(fontsize=11)
plt.xlabel("Training Steps", fontsize=13)
plt.ylabel("FVD ↓", fontsize=13)

# Remove top and right spines for a clean academic look
sns.despine()

# Legend formatting
plt.legend(
    title="Model Size",
    fontsize=10.5,
    title_fontsize=11,
    loc="upper right",
    frameon=False
)

# Add light gridlines only on y-axis for readability
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("ablation_model_size.png", dpi=300)
#SAVEPDF
plt.savefig("ablation_model_size.pdf")
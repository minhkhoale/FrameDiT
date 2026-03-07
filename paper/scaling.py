import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Your data
# -------------------------
def draw_diameter_legend(ax, g_values, scale_func,
                        x0=0.03, x1=0.25, y=0.07):
    """
    Draw horizontal diameter legend above the plot.
    
    ax: matplotlib axis
    g_values: list of GFLOPs ticks (e.g., [5,20,80,320])
    scale_func: function mapping GFLOPs -> bubble area
    x0,x1,y: position in axis fraction coordinates
    """

    # Convert axis fraction to data coords
    trans = ax.transAxes

    # Draw horizontal line
    ax.plot([x0, x1], [y, y], transform=trans,
            color='0.5', lw=1.5, clip_on=False)

    gmin, gmax = -20, 750

    # Place ticks
    for g in g_values:
        t = (g - gmin) / (gmax - gmin)
        x = x0 + t * (x1 - x0)

        # vertical tick
        ax.plot([x, x], [y, y+0.015], transform=trans,
                color='0.5', lw=1.5, clip_on=False)

        # label
        ax.text(x, y-0.02, f"{g}",
                transform=trans,
                ha="center", va="top",
                fontsize=9, color='0.45')

    # Title text
    ax.text(x0, y+0.03, "Diameter",
            transform=trans,
            fontsize=10, color='0.45')

    ax.text(x1, y+0.03, "GFLOPs",
            transform=trans,
            fontsize=10, color='0.45', ha='right')

# -------------------------
# Data
# -------------------------
csv_text = """model,s,b,l,xl,gflop_s,gflop_b,gflop_l,gflop_xl,param_s,param_b,param_l,param_xl
Local Factorized,143,89,75,70,32.505232,88.755929088,315.448623104,465.743167488,32.505,129.596,315.449,465.743
Full 3D,113,77,68,64,32.949387264,131.732865024,466.141577216,687.8048256,32.949,131.733,466.142,687.805
FrameDiT-G,122,79,70,66,29.325508608,117.2373504,414.601969664,611.703373824,29.326,117.237,414.602,611.703
FrameDiT-H,110,76,60,53,34.78020096,139.018371072,492.012044288,725.987672064,34.780,139.018,492.012,725.988
"""
df = pd.read_csv(pd.io.common.StringIO(csv_text))
palette = {
    "Local Factorized": "#F78F27",
    "Full 3D": "#FD88FD",
    "FrameDiT-G": "#567BFF",
    "FrameDiT-H": "#43C26F",
}
sizes = ["s","b","l","xl"]

rows = []
for _, r in df.iterrows():
    for s in sizes:
        rows.append({
            "model": r["model"],
            "fvd": r[s],
            "param": r[f"param_{s}"]
        })
data = pd.DataFrame(rows)

# -------------------------
# Bubble scaling
# -------------------------
# gmin, gmax = data["param"].min(), data["param"].max()
# area_min, area_max = 80, 12000

# def scale_area(g):
#     return area_min + (g - gmin) / (gmax - gmin) * (area_max - area_min)

# data["area"] = data["param"].apply(scale_area)

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(8,5))
ax = plt.gca()

models = data["model"].unique()

for m in models:
    sub = data[data["model"] == m]
    ax.scatter(
        sub["param"],
        sub["fvd"],
        s=120,
        #s=sub["area"],
        #alpha=0.55,
        #edgecolors="none",
        label=m,
        c=palette[m]
    )

# axis formatting
ax.set_xlabel("Model size (Million)", fontsize=18)
ax.set_ylabel("FVD ↓", fontsize=18)
ax.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
ax.set_ylim(30, 150)
ax.set_xlim(0,800)
ax.tick_params(axis='x', labelsize=17)
#ax.set_xticks([])
ax.tick_params(axis='y', labelsize=17)
# -------------------------
# Diameter legend
# -------------------------
# legend_g = [500, 500, 500, 500]
# legend_g = [g for g in legend_g if gmin <= g <= gmax]

# legend_handles = [
#     plt.scatter([], [], s=scale_area(g), alpha=0.45)
#     for g in legend_g
# ]
# legend_labels = [f"{g:g}" for g in legend_g]

# leg = ax.legend(
#     legend_handles,
#     legend_labels,
#     title="Diameter\nGFLOPs",
#     loc="lower left",
#     frameon=True
# )
# ax.add_artist(leg)
#g_ticks = [30, 300, 700]
#draw_diameter_legend(ax, g_ticks, scale_area)

# model legend
ax.legend(title="Model", loc="upper right", fontsize=17, title_fontsize=18, frameon=True)
# from matplotlib.lines import Line2D

# models = data["model"].unique()

# legend_handles = [
#     Line2D(
#         [0], [0],
#         marker='o',
#         color='none',
#         markerfacecolor=palette[m],
#         markersize=12,   # ← fixed size here
#         label=m,
#         #alpha=0.55,
#         markeredgecolor="none",
#     )
#     for i, m in enumerate(models)
# ]

# ax.legend(
#     handles=legend_handles,
#     title="Model",
#     loc="upper right",
#     fontsize=11,
#     title_fontsize=13,
#     frameon=True
# )

plt.tight_layout()

plt.savefig('scaling.png')
plt.savefig('scaling.pdf')
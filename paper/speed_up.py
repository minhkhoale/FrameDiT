import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======== Load data ========
df = pd.read_csv("speed_vs_frames_1.csv")

# Extract models of interest (optional: simplify labels)
selected_models = {
    "DiT3D-M/2": "Full DiT",
    "Latte-M/2": "Factorized DiT",
    "MatLatte-M/1-128/2": "MatrixDiT (Ours)",
    "FusedMatLatte-M/1-128/2-concat": "Fused-MatrixDiT (Ours)"
}

df = df[df["model"].isin(selected_models.keys())]
df["Model"] = df["model"].map(selected_models)

# ======== Compute speedup ========
# baseline (DiT3D-M/2)
baseline = df[df["Model"] == "Full DiT"][["num_frames", "latency_avg_ms"]].set_index("num_frames")

def compute_speedup(row):
    base_lat = baseline.loc[row["num_frames"], "latency_avg_ms"]
    return base_lat / row["latency_avg_ms"]

df["SpeedUp"] = df.apply(compute_speedup, axis=1)

# ======== Plot ========
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.figsize": (5.0, 3.3),
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "savefig.dpi": 400,
    "axes.linewidth": 1.0
})

palette = {
    "Full DiT": "#1f77b4",
    "Factorized DiT": "#ff7f0e",
    "MatrixDiT (Ours)": "#2ca02c",
    "Fused-MatrixDiT (Ours)": "#d62728"
}

plt.figure()
sns.lineplot(
    data=df,
    x="num_frames",
    y="SpeedUp",
    hue="Model",
    style="Model",
    markers=True,
    dashes=False,
    palette=palette
)

plt.xscale("log", base=2)
plt.xlabel("Number of Frames")
plt.ylabel("Speed-Up vs Full DiT (↑)")
plt.title("Inference Speed-Up vs Frame Length")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("figure_speedup_vs_frames.png")
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# =====================
# Table Data
# =====================
data = {
    "Property": [
        "Handle Large Motion",
        "Linear Complexity",
        "Temporal Coherence",
        "Cross-Frame Reasoning",
        "Design Flexibility",
        "Memory Footprint"
    ],
    "Full Attention": ["✓", "✗", "✓", "✓", "✗", "✗"],
    "Factorized Attention": ["✗", "✓", "✗", "✗", "✗", "✓"],
    "Matrix Attention (Ours)": ["✓", "✓", "✓", "✓", "✓", "✓"]
}

df = pd.DataFrame(data)

# =====================
# Create Figure
# =====================
fig, ax = plt.subplots(figsize=(5.5, 2.4))
ax.axis("off")

# Column colors — highlight our method
col_colors = ["#F5F5F5", "#EFEFEF", "#EFEFEF", "#E0E5FF"]

# Build the table
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc="center",
    colColours=col_colors,
    loc="center"
)

# =====================
# Style Configuration
# =====================
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.4)

# Adjust header style
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold", color="black")
        cell.set_facecolor("#DDE1FF")
    # Shade "Ours" column
    if col == 3:
        cell.set_facecolor("#E7E7FF")

# =====================
# Optional caption
# =====================
plt.figtext(
    0.5, -0.05,
    "Comparison of attention mechanisms. "
    "Matrix Attention combines motion robustness and efficiency with flexible design.",
    wrap=True, ha="center", fontsize=9
)

plt.tight_layout()
plt.savefig('comparison_table.png', dpi=300)

# =====================
# Export Options
# =====================
# fig.savefig("matrix_attention_comparison.pdf", bbox_inches="tight", dpi=600)
# fig.savefig("matrix_attention_comparison.svg", bbox_inches="tight")

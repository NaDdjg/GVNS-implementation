import os
import matplotlib.pyplot as plt

# ==========================================================
# Output folder
# ==========================================================

output_dir = "result_summaries/plots/scalability_analysis"
os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# Data
# ==========================================================

models = [
    "GVNS+",
    "BF",
    "DQN",
    "DQN+BF",
    "PPO",
    "Q-Learning",
    "QL+BF"
]

avg_gap = [
    1.51,
    2.24,
    1.38,
    1.67,
    0.99,
    3.57,
    6.33
]

avg_time = [
    9540,
    9892,
    8858,
    11117,
    9338,
    8412,
    9464
]

# ==========================================================
# Colors
# ==========================================================

colors = {
    "GVNS+": "#1f77b4",
    "BF": "#ff7f0e",
    "DQN": "#2ca02c",
    "DQN+BF": "#9467bd",
    "PPO": "#d62728",
    "Q-Learning": "#8c564b",
    "QL+BF": "#7f7f7f"
}

# ==========================================================
# Plot
# ==========================================================

plt.figure(figsize=(10, 7))

for model, x, y in zip(models, avg_time, avg_gap):

    size = 150 if model == "PPO" else 90

    plt.scatter(
        x,
        y,
        s=size,
        color=colors[model],
        edgecolors="black",
        linewidths=1.2,
        alpha=0.9,
        zorder=3
    )

    plt.annotate(
        model,
        (x, y),
        xytext=(8, 6),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold" if model == "PPO" else "normal"
    )

# ==========================================================
# Highlight PPO
# ==========================================================

ppo_x = avg_time[models.index("PPO")]
ppo_y = avg_gap[models.index("PPO")]

plt.scatter(
    ppo_x,
    ppo_y,
    s=500,
    facecolors='none',
    edgecolors='red',
    linewidths=2.5,
    zorder=2
)

# ==========================================================
# Ideal region
# ==========================================================

plt.axhspan(
    0,
    1.5,
    alpha=0.10,
    color='green',
    label="High-quality region"
)

# ==========================================================
# Formatting
# ==========================================================

plt.title(
    "Solution Quality vs Computational Time",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Average Execution Time (s)",
    fontsize=12
)

plt.ylabel(
    "Average Gap (%)",
    fontsize=12
)

plt.grid(
    linestyle="--",
    alpha=0.5
)

plt.legend(loc="upper left")

plt.tight_layout()

# ==========================================================
# Save
# ==========================================================

save_path = os.path.join(
    output_dir,
    "quality_time_tradeoff.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved to: {save_path}")
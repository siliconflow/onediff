import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Plot optimization scores")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file"
    )
    return parser.parse_args()


# Load settings from the YAML file
args = parse_args()
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

categories = config["categories"]
optimization_scores = config["optimization_scores"]

# Use default scores on bars if configured to do so
if config.get("use_default_scores_on_bars", False):
    optimization_scores_on_bars = config.get("optimization_scores", [0.3, 0.3, 0.35])
else:
    optimization_scores_on_bars = config.get("custom_scores_on_bars", [])


# Set up a colormap for gradient color
colormap_range = config.get("colormap_range", [0.2, 0.5, 0.8])
# colors = plt.cm.get_cmap(config['colormap'])(colormap_range)
colors = mpl.colormaps.get_cmap(config["colormap"])(colormap_range)


# Set figure size and resolution
fig, ax = plt.subplots(
    figsize=(
        config["figure"]["figsize"]["width"],
        config["figure"]["figsize"]["height"],
    ),
    dpi=config["figure"]["dpi"],
)

# Customize bar width and use gradient colors
bars = ax.bar(
    categories,
    optimization_scores,
    width=config["bar_width"],
    color=colors,
    edgecolor="None",
)

# Add Optimization Scores on top of the bars
if config.get("display_scores_on_bars", False):
    for bar, score in zip(bars, optimization_scores_on_bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            config.get("scores_format_on_bars", "{score:.2f}").format(score=score),
            ha="center",
            va="bottom",
            fontsize=config["label_fontsize"],
        )
        # ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{score:.2f}', ha='center', va='bottom', fontsize=config['label_fontsize'])

# Adjust font size for ticks
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(
    categories,
    fontsize=config["label_fontsize"],
    rotation=config["rotation_angle"],
    ha="right",
)
ax.tick_params(
    axis="x", which="both", direction="in"
)  # Set x-axis ticks direction to 'in'

ax.set_yticks(ax.get_yticks())
ax.set_yticklabels(
    [f"{tick:.2f}" for tick in ax.get_yticks()], fontsize=config["label_fontsize"]
)
ax.tick_params(
    axis="y", which="both", direction="in"
)  # Set y-axis ticks direction to 'in'

# Remove right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Set the color of dashed lines for right and top frame
ax.spines["right"].set_linestyle(config["line_style"])
ax.spines["top"].set_linestyle(config["line_style"])
ax.spines["right"].set_edgecolor(config["line_color"])
ax.spines["top"].set_edgecolor(config["line_color"])

# Add dashed grid lines for both x-axis and y-axis ticks
ax.grid(
    axis="both",
    linestyle=config["line_style"],
    linewidth=config["line_width"],
    color=config["line_color"],
    alpha=config["line_alpha"],
)

# Set labels and title
ax.set_xlabel(config["x_label"], fontsize=config["label_fontsize"])
ax.set_ylabel(config["y_label"], fontsize=config["label_fontsize"])
ax.set_title(config["plot_title"], fontsize=config["label_fontsize"])

# Save the plot with higher quality
plt.savefig(config["output_filename"], dpi=config["figure"]["dpi"], bbox_inches="tight")

# Show the plot
plt.show()

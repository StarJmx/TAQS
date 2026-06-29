import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# ==============================
# Path configuration
# ==============================
EXCEL_PATH = "../../results/noise_impact_experiment.xlsx"
SHEET_NAME = "plot"
OUTPUT_DIR = "../../results/insights_into_performance_improvements_figures"

# Color palette (consistent style with original code)
COLOR_PALETTE = [
    "#d62728",  # Red
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#1f77b4",  # Blue
]

# Marker styles (refer to utils convention)
MARKERS = ["s", "D", "^", "o"]
LS = ["--", ":", "-.", "-"]
RATIO = [0, 0.5, 0.7, 0.9]

def set_paper_style():
    """Simplified version of utils.set_paper_style"""
    plt.style.use("default")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def plot_noise_impact():
    if not os.path.exists(EXCEL_PATH):
        print(f"[Error] File not found: {EXCEL_PATH}")
        return

    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=None)

    # ==============================
    # Data parsing
    # ==============================

    # First row (except first column) as x-axis
    raw_x_values = df.iloc[0, 1:-1].values

    def parse_x_value(v):
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        s = str(v).strip()
        match = re.search(r"-?\d+(\.\d+)?", s)
        if match:
            return float(match.group())
        raise ValueError(f"Cannot parse x value from: {v}")

    x_values = np.array([parse_x_value(v) for v in raw_x_values], dtype=float)

    # Curve names (first column, starting from second row)
    curve_names = df.iloc[1:, 0].values

    # Curve data (excluding first row, first column, and last column)
    y_values = df.iloc[1:, 1:-1].values

    # Last column (baseline values)
    baseline_values = df.iloc[1:, -1].values

    # ==============================
    # Plotting
    # ==============================
    set_paper_style()

    figsize = (3, 3)
    dpi = 600
    fontsize = 12

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.minorticks_on()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    # Main curves
    for i, name in enumerate(curve_names):
        r = RATIO[i]
        ax.plot(
            x_values,
            y_values[i],
            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            marker=MARKERS[i % len(MARKERS)],
            markersize=3,
            linewidth=1.2,
            label=name,
            zorder=2,
        )

    # Horizontal dashed lines (baseline)
    for i, name in enumerate(curve_names):
        ax.axhline(
            y=baseline_values[i],
            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            linestyle=LS[i % len(LS)],
            linewidth=1.5,
            alpha=0.8,
            label=f"{name} (full)",
            zorder=1,
        )

    # ==============================
    # Style details (aligned with top conference style)
    # ==============================
    ax.set_xlabel("$k$", fontsize=fontsize)
    ax.set_ylabel(r"$R^2$", fontsize=fontsize)
    # ax.set_xlim(0, 200)
    ax.set_xticks([0, 50, 100, 150, 200])

    ax.tick_params(axis='both', labelsize=fontsize)

    ax.legend(
        title="Noise level",
        loc="best",
        fontsize=9,
        frameon=True,
        framealpha=0.3,
    )

    fig.tight_layout()

    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "_noise_impact_plot.pdf")
    fig.savefig(save_path)
    plt.close(fig)

    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_noise_impact()

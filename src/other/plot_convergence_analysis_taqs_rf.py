import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# Path configuration
# ==============================
EXCEL_PATH = "../../results/convergence_analysis_taqs_rf.xlsx"
OUTPUT_DIR = "../../results/insights_into_performance_improvements_figures"

# Color palette
COLOR_PALETTE = [
    "#d62728",  # Red
    "#1f77b4",  # Blue
]

# Marker styles
MARKERS = ["s", "o"]

def set_paper_style():
    """Simplified version of utils.set_paper_style"""
    plt.style.use("default")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def plot_convergence_r2():
    if not os.path.exists(EXCEL_PATH):
        print(f"[Error] File not found: {EXCEL_PATH}")
        return

    # Read Excel, first row as column names
    df = pd.read_excel(EXCEL_PATH, sheet_name=0, header=0)

    # Filter rows with R2 metric
    df_r2 = df[df["Metric"] == "R2"].copy()

    if df_r2.empty:
        print("[Error] No R2 data found in the Excel file.")
        return

    # Extract x-axis values: parse n values from column names (e.g., "n=10" -> 10)
    x_cols = []
    x_values = []
    for col in df.columns[2:]:  # Skip Method and Metric columns
        if "_std" not in col:  # Only take mean columns
            n_val = int(col.split("=")[1])
            x_values.append(n_val)
            x_cols.append(col)

    x_values = np.array(x_values, dtype=float)

    # Extract mean and standard deviation for each method
    methods = df_r2["Method"].unique()
    means_list = []
    stds_list = []
    method_names = []

    for method in methods:
        row = df_r2[df_r2["Method"] == method]
        if row.empty:
            continue

        mean_vals = []
        std_vals = []
        for col in x_cols:
            mean_col = col
            std_col = f"{col}_std"
            mean_vals.append(row[mean_col].values[0])
            std_vals.append(row[std_col].values[0])

        means_list.append(np.array(mean_vals, dtype=float))
        stds_list.append(np.array(std_vals, dtype=float))
        method_names.append(method)

    # ==============================
    # Plotting
    # ==============================
    set_paper_style()

    figsize = (3, 2.2)
    dpi = 600
    fontsize = 12

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.minorticks_on()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    # Draw curves and error bands
    for i, name in enumerate(method_names):
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        marker = MARKERS[i % len(MARKERS)]

        # Error band (standard deviation)
        ax.fill_between(
            x_values,
            means_list[i] - stds_list[i],
            means_list[i] + stds_list[i],
            color=color,
            alpha=0.2,
            zorder=1,
        )

        # Mean curve
        ax.plot(
            x_values,
            means_list[i],
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.2,
            label=name,
            zorder=2,
        )

    # ==============================
    # Style details
    # ==============================
    ax.set_xlabel("n_estimators (RF)", fontsize=fontsize)
    ax.set_ylabel(r"$R^2$", fontsize=fontsize)

    ax.set_xticks([0, 100, 200, 300, 400, 500])
    # ax.set_xlim(0, 520)

    ax.tick_params(axis='both', labelsize=fontsize)

    ax.legend(
        loc="best",
        fontsize=10,
        frameon=False,
        framealpha=0.3,
    )

    fig.tight_layout()

    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "convergence_r2_rf_plot.jpg")
    fig.savefig(save_path)
    plt.close(fig)

    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    plot_convergence_r2()
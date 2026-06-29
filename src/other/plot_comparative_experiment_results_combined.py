import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import utils

# ==============================
# configuration
# ==============================
EXCEL_PATH = "../../results/comparative_experiment.xlsx"
OUTPUT_DIR = "../../results/comparative_experiment_figures"
METRIC_TO_PLOT = "R2"

COLOR_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#8c564b",
    "#9467bd",
    "#d62728",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


def extract_k_columns(df):
    k_cols = [c for c in df.columns if c.startswith("k=") and not c.endswith("_std")]
    k_cols = sorted(k_cols, key=lambda c: int(c.split("=")[1]))
    k_values = [int(c.split("=")[1]) for c in k_cols]
    return k_cols[:10], k_values[:10]


def set_paper_style():
    """Simplified version of utils.set_paper_style"""
    plt.style.use("default")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


# ==============================
# Subplot drawing
# ==============================
def plot_all_algorithms_subplot(ax, df, show_ylabel=False):
    k_cols, k_values = extract_k_columns(df)
    if not k_cols:
        return

    df_r2 = df[df["Metric"] == METRIC_TO_PLOT].copy()
    if df_r2.empty:
        return

    cfg = utils.load_plot_config()
    curve_cfg = cfg.get("plot_train_test_curve", {})

    markers_cfg = curve_cfg.get("markers")
    markers = markers_cfg if isinstance(markers_cfg, list) else [markers_cfg]

    markersize = curve_cfg.get("scatter_size")
    line_width = curve_cfg.get("line_width")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if show_ylabel:
        ax.set_ylabel(r"$R^2$")
    else:
        ax.set_yticklabels([])

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    models = sorted(df_r2["Model"].dropna().unique())

    for idx, model_name in enumerate(models):
        model_row = df_r2[df_r2["Model"] == model_name]
        if model_row.empty:
            continue

        model_row = model_row.iloc[0]
        r2_values = [model_row[col] for col in k_cols]

        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        marker = markers[idx % len(markers)]

        ax.plot(
            k_values,
            r2_values,
            color=color,
            marker=marker,
            markersize=markersize,
            linewidth=line_width,
            label=model_name.replace("_", " "),
            zorder=2,
        )

    # ===== xtick labels =====
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_xticklabels(["2", "4", "6", "8", "10"])


# ==============================
# Main logic: Multi-dataset merging
# ==============================
def plot_all_datasets_combined(xls):
    set_paper_style()

    datasets = xls.sheet_names
    n_datasets = len(datasets)

    fig, axes = plt.subplots(
        1,
        n_datasets,
        figsize=(2.3 * n_datasets, 2.4),
        dpi=600,
    )

    if n_datasets == 1:
        axes = [axes]

    for i, sheet_name in enumerate(datasets):
        print(f"[Processing] {sheet_name}")
        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name)

        plot_all_algorithms_subplot(
            ax=axes[i],
            df=df,
            show_ylabel=(i == 0),
        )

        axes[i].set_title(sheet_name.replace("_", " "), fontsize=9)

    # ===== Global legend (horizontal at the top) =====
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1),
    )

    # ===== xlabel =====
    fig.supxlabel(
        "$k$",
        fontsize=9,
        y=-0.05,
    )

    # ===== subplot spacing =====
    plt.subplots_adjust(wspace=0.3, top=0.80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_path = os.path.join(
        OUTPUT_DIR,
        "all_datasets_all_algorithms_combined.pdf"
    )

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] Combined figure -> {save_path}")


# ==============================
# main
# ==============================
def main():
    if not os.path.exists(EXCEL_PATH):
        print(f"[Error] Excel file not found: {EXCEL_PATH}")
        return

    xls = pd.ExcelFile(EXCEL_PATH)
    print(f"[Load] {EXCEL_PATH}")
    print(f"[Datasets] {xls.sheet_names}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_all_datasets_combined(xls)

    print(f"[Done] All figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
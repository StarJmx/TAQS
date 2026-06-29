import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import utils

# Input file paths
TAQS_EXCEL_PATH = "../../results/validation_taqs_paradigm.xlsx"
RQAS_EXCEL_PATH = "../../results/validation_rqas_paradigm.xlsx"
OUTPUT_DIR = "../../results/replot_comparison_taqs_rqas_figures"

METRICS_TO_PLOT = ["R2"]
SHOW_TRAIN_CURVE = False

# Fixed tick positions on the x-axis
FIXED_XTICKS = [0, 50, 100, 150, 200]


def extract_n_columns(df):
    """Extract n=... column names and corresponding n value list from dataframe"""
    n_cols = [c for c in df.columns if c.startswith("n=") and not c.endswith("_std")]
    n_cols = sorted(n_cols, key=lambda c: int(c.split("=")[1]))
    n_values = [int(c.split("=")[1]) for c in n_cols]
    return n_cols, n_values


def get_sample_size_mapping(df, n_cols):
    """
    Extract sample size for each n_neighbors from df.
    Return dict: {n_value: sample_size} and baseline_full_train sample size.
    """
    size_row = df[(df["Model"] == "TAQS") & (df["Split"] == "subset") & (df["Metric"] == "SampleSize")]
    if size_row.empty:
        return {}, None

    size_row = size_row.iloc[0]
    mapping = {}
    for col in n_cols:
        n_val = int(col.split("=")[1])
        val = size_row.get(col, np.nan)
        if pd.notna(val):
            mapping[n_val] = float(val)

    baseline_size = size_row.get("baseline_full_train", np.nan)
    if pd.notna(baseline_size):
        baseline_size = float(baseline_size)

    return mapping, baseline_size


def plot_comparison(
    n_values,
    taqs_train,
    taqs_test,
    rqas_train,
    rqas_test,
    baseline_value,
    baseline_size,
    sample_size_map,
    xlabel,
    ylabel,
    save_path,
    show_train_curve=False,
):
    """
    Plot comparison curves between TAQS and RQAS.
    X-axis displays fixed FIXED_XTICKS, tick 0 does not show sample size, other ticks include sample size.
    """
    cfg = utils.load_plot_config()
    curve_cfg = cfg.get("plot_train_test_curve", {})

    utils.set_paper_style()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    figsize = tuple(curve_cfg.get("figsize", [3.2, 3.0]))
    dpi = curve_cfg.get("dpi", 300)
    marker = curve_cfg.get("marker", "o")
    scatter_size = curve_cfg.get("scatter_size")
    line_width = curve_cfg.get("line_width", 1)
    marker_cfg = curve_cfg.get("markers")
    if isinstance(marker_cfg, list):
        markers = marker_cfg
    else:
        markers = [marker_cfg]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.minorticks_on()
    ax.grid(
        which="minor",
        linestyle=curve_cfg.get("minor_grid_linestyle"),
        linewidth=curve_cfg.get("minor_grid_linewidth"),
        alpha=curve_cfg.get("minor_grid_alpha"),
    )
    ax.set_xlabel(xlabel, labelpad=curve_cfg.get("xlabel_pad"))
    ax.set_ylabel(ylabel, labelpad=curve_cfg.get("xlabel_pad"))

    # Plot TAQS test curve (main curve)
    if taqs_test is not None:
        ax.plot(
            n_values,
            taqs_test,
            color=curve_cfg.get("test_line_color"),
            marker=markers[5],
            markersize=scatter_size,
            linewidth=line_width,
            label="TAQS (task_set)",
            zorder=2,
        )

    # Plot TAQS training curve (optional)
    if show_train_curve and taqs_train is not None:
        ax.plot(
            n_values,
            taqs_train,
            color=curve_cfg.get("train_line_color"),
            marker=markers[5],
            markersize=scatter_size,
            linewidth=line_width,
            label="TAQS (train_set)",
        )

    # Plot RQAS test curve
    if rqas_test is not None:
        ax.plot(
            n_values,
            rqas_test,
            color="#1f77b4",
            marker=markers[1],
            markersize=scatter_size,
            linewidth=line_width,
            label="RQAS (task_set)",
            zorder=1,
        )

    # Plot RQAS training curve (optional)
    if show_train_curve and rqas_train is not None:
        ax.plot(
            n_values,
            rqas_train,
            color="darkviolet",
            marker=markers[1],
            markersize=scatter_size,
            linewidth=line_width,
            label="RQAS (train_set)",
        )

    # Plot baseline horizontal line
    if pd.notna(baseline_value):
        label = f"Baseline (N={baseline_size:.0f})" if baseline_size is not None else "baseline_full_train"
        ax.axhline(
            y=float(baseline_value),
            color="#7f7f7f",
            linestyle="--",
            linewidth=1.5,
            label=label,
            zorder=0,
        )

    # Set x-axis ticks to fixed FIXED_XTICKS, and append sample size labels (except tick 0)
    ax.set_xticks(FIXED_XTICKS)
    labels = []
    for n in FIXED_XTICKS:
        if n == 0:
            # Tick 0 does not show sample size
            label = "0"
        else:
            size = sample_size_map.get(n, "")
            if size != "":
                label = f"{n} ({size:.0f})"
            else:
                label = str(n)
        labels.append(label)
    ax.set_xticklabels(labels)

    ax.legend(
        loc="best",
        frameon=False,
        fontsize=8,
        markerscale=1,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Saved] Figure -> {save_path}")


def process_dataset(sheet_name, df_taqs, df_rqas):
    """Process a single dataset and generate comparison plots"""
    # Extract column information from TAQS
    n_cols, n_values = extract_n_columns(df_taqs)
    if not n_values:
        print(f"[Warning] No n columns found in {sheet_name}, skip.")
        return

    # Get TAQS sample size mapping (for baseline size)
    _, baseline_size_taqs = get_sample_size_mapping(df_taqs, n_cols)

    # Get RQAS sample size mapping (for tick labels), fallback to TAQS if RQAS is missing
    sample_size_map = {}
    if df_rqas is not None:
        rqas_size_map, _ = get_sample_size_mapping(df_rqas, n_cols)
        sample_size_map.update(rqas_size_map)
    taqs_size_map, _ = get_sample_size_mapping(df_taqs, n_cols)
    for n_val, size in taqs_size_map.items():
        if n_val not in sample_size_map:
            sample_size_map[n_val] = size

    # List of models to plot
    models = sorted(m for m in df_taqs["Model"].dropna().unique() if m != "TAQS")

    for model_name in models:
        for metric_name in METRICS_TO_PLOT:
            # Extract data from TAQS
            taqs_train = df_taqs[
                (df_taqs["Model"] == model_name) &
                (df_taqs["Split"] == "train") &
                (df_taqs["Metric"] == metric_name)
            ]
            taqs_test = df_taqs[
                (df_taqs["Model"] == model_name) &
                (df_taqs["Split"] == "test") &
                (df_taqs["Metric"] == metric_name)
            ]
            if taqs_train.empty or taqs_test.empty:
                continue

            taqs_train_row = taqs_train.iloc[0]
            taqs_test_row = taqs_test.iloc[0]

            taqs_train_vals = [taqs_train_row[col] for col in n_cols]
            taqs_test_vals = [taqs_test_row[col] for col in n_cols]
            baseline_value = taqs_test_row.get("baseline_full_train", np.nan)

            # Extract data from RQAS (if exists)
            rqas_train_vals = None
            rqas_test_vals = None
            if df_rqas is not None:
                rqas_train = df_rqas[
                    (df_rqas["Model"] == model_name) &
                    (df_rqas["Split"] == "train") &
                    (df_rqas["Metric"] == metric_name)
                ]
                rqas_test = df_rqas[
                    (df_rqas["Model"] == model_name) &
                    (df_rqas["Split"] == "test") &
                    (df_rqas["Metric"] == metric_name)
                ]
                if not rqas_train.empty:
                    rqas_train_row = rqas_train.iloc[0]
                    rqas_train_vals = [rqas_train_row[col] for col in n_cols]
                if not rqas_test.empty:
                    rqas_test_row = rqas_test.iloc[0]
                    rqas_test_vals = [rqas_test_row[col] for col in n_cols]

            save_path = os.path.join(
                OUTPUT_DIR,
                f"{sheet_name}_{model_name}_{metric_name}.png".replace(" ", "_"),
            )

            plot_comparison(
                n_values=n_values,
                taqs_train=taqs_train_vals,
                taqs_test=taqs_test_vals,
                rqas_train=rqas_train_vals,
                rqas_test=rqas_test_vals,
                baseline_value=baseline_value,
                baseline_size=baseline_size_taqs,
                sample_size_map=sample_size_map,
                xlabel="Number of nearest neighbors (Sample size)",
                ylabel=r"Performance (R$^2$)" if metric_name == "R2" else metric_name,
                save_path=save_path,
                show_train_curve=SHOW_TRAIN_CURVE,
            )


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read TAQS result file
    if not os.path.exists(TAQS_EXCEL_PATH):
        print(f"[Error] TAQS file not found: {TAQS_EXCEL_PATH}")
        sys.exit(1)
    xls_taqs = pd.ExcelFile(TAQS_EXCEL_PATH)
    print(f"[Load] {TAQS_EXCEL_PATH}")

    # Try to read RQAS result file
    rqas_available = False
    if os.path.exists(RQAS_EXCEL_PATH):
        xls_rqas = pd.ExcelFile(RQAS_EXCEL_PATH)
        print(f"[Load] {RQAS_EXCEL_PATH}")
        rqas_available = True
    else:
        print(f"[Warning] RQAS file not found: {RQAS_EXCEL_PATH}, only TAQS will be plotted.")
        xls_rqas = None

    # Get all datasets (based on sheets in TAQS)
    datasets = xls_taqs.sheet_names
    print(f"[Datasets] {datasets}")

    for sheet_name in datasets:
        print(f"[Processing] {sheet_name}")
        df_taqs = pd.read_excel(TAQS_EXCEL_PATH, sheet_name=sheet_name)

        df_rqas = None
        if rqas_available and sheet_name in xls_rqas.sheet_names:
            df_rqas = pd.read_excel(RQAS_EXCEL_PATH, sheet_name=sheet_name)

        process_dataset(sheet_name, df_taqs, df_rqas)

    print(f"[Saved] All figures -> {OUTPUT_DIR}")
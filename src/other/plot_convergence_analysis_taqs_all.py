"""
Implement RR / RF / MLP convergence R² plot logic in a single file, corresponding to the following scripts respectively,
keeping parameters and data paths consistent (only centralized for generating three plots at once):
  - plot_convergence_analysis_taqs_rr.py
  - plot_convergence_analysis_taqs_rf.py
  - plot_convergence_analysis_taqs_mlp.py

Run as before with each individual script: working directory is recommended to be ``src/other`` (or
already temporarily switched to this file's directory within ``plot_all_convergence_r2``).
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================
# Path configuration (consistent with each individual script)
# ==============================
EXCEL_PATH_RR = "../../results/convergence_analysis_taqs_rr.xlsx"
EXCEL_PATH_RF = "../../results/convergence_analysis_taqs_rf.xlsx"
EXCEL_PATH_MLP = "../../results/convergence_analysis_taqs_mlp.xlsx"
OUTPUT_DIR = "../../results/insights_into_performance_improvements_figures"

# Color (consistent style with original code)
COLOR_PALETTE = [
    "#d62728",  # Red
    "#1f77b4",  # Blue
]

# Marker styles
MARKERS = ["s", "o"]

FIGSIZE = (3, 2.2)


def set_paper_style():
    """Simplified version of utils.set_paper_style"""
    plt.style.use("default")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


# ---------------------------------------------------------------------------
# RR - Consistent with plot_convergence_r2 in plot_convergence_analysis_taqs_rr.py
# ---------------------------------------------------------------------------
def plot_convergence_r2_rr():
    if not os.path.exists(EXCEL_PATH_RR):
        print(f"[Error] File not found: {EXCEL_PATH_RR}")
        return

    # Read Excel, first row as column names
    df = pd.read_excel(EXCEL_PATH_RR, sheet_name=0, header=0)

    # Filter rows with R2 metric
    df_r2 = df[df["Metric"] == "R2"].copy()

    if df_r2.empty:
        print("[Error] No R2 data found in the Excel file.")
        return

    # Extract x-axis values: column names are alpha=... (RR) or iter=... (MLP class); take mean columns, excluding _std
    x_cols = []
    x_values = []
    for col in df.columns[2:]:  # Skip Method and Metric columns
        if "_std" in col or "=" not in col:
            continue
        param_name, _, raw = col.partition("=")
        if param_name not in ("alpha", "iter"):
            continue
        try:
            x_val = float(raw)
        except ValueError:
            continue
        x_values.append(x_val)
        x_cols.append(col)

    order = np.argsort(x_values)
    x_values = np.array(x_values, dtype=float)[order]
    x_cols = [x_cols[i] for i in order]

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

    figsize = FIGSIZE
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
    ax.set_xlabel(r"alpha (RR)", fontsize=fontsize)
    ax.set_ylabel(r"$R^2$", fontsize=fontsize)

    ax.set_xscale("log")
    ax.set_xticks(x_values)

    ax.tick_params(axis="x", rotation=45)

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
    save_path = os.path.join(OUTPUT_DIR, "convergence_r2_rr_plot.svg")
    fig.savefig(save_path)
    plt.close(fig)

    print(f"[Saved] {save_path}")


# ---------------------------------------------------------------------------
# RF - Consistent with plot_convergence_r2 in plot_convergence_analysis_taqs_rf.py
# ---------------------------------------------------------------------------
def plot_convergence_r2_rf():
    if not os.path.exists(EXCEL_PATH_RF):
        print(f"[Error] File not found: {EXCEL_PATH_RF}")
        return

    # Read Excel, first row as column names
    df = pd.read_excel(EXCEL_PATH_RF, sheet_name=0, header=0)

    # Filter rows with R2 metric
    df_r2 = df[df["Metric"] == "R2"].copy()

    if df_r2.empty:
        print("[Error] No R2 data found in the Excel file.")
        return

    # Extract x-axis values: parse n values from column names (e.g., "n=10" -> 10)
    x_cols = []
    x_values = []
    for col in df.columns[2:]:          # Skip Method and Metric columns
        if "_std" not in col:            # Only take mean columns (columns without _std)
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

    figsize = FIGSIZE
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
    save_path = os.path.join(OUTPUT_DIR, "convergence_r2_rf_plot.svg")
    fig.savefig(save_path)
    plt.close(fig)

    print(f"[Saved] {save_path}")


# ---------------------------------------------------------------------------
# MLP - Consistent with plot_convergence_r2 in plot_convergence_analysis_taqs_mlp.py
# ---------------------------------------------------------------------------
def plot_convergence_r2_mlp():
    if not os.path.exists(EXCEL_PATH_MLP):
        print(f"[Error] File not found: {EXCEL_PATH_MLP}")
        return

    # Read Excel, first row as column names
    df = pd.read_excel(EXCEL_PATH_MLP, sheet_name=0, header=0)

    # Filter rows with R2 metric
    df_r2 = df[df["Metric"] == "R2"].copy()

    if df_r2.empty:
        print("[Error] No R2 data found in the Excel file.")
        return

    # Extract x-axis values: parse n values from column names (e.g., "n=10" -> 10)
    x_cols = []
    x_values = []
    for col in df.columns[2:]:          # Skip Method and Metric columns
        if "_std" not in col:            # Only take mean columns (columns without _std)
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

    figsize = FIGSIZE
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
    ax.set_xlabel("max_iter (MLP)", fontsize=fontsize)
    ax.set_ylabel(r"$R^2$", fontsize=fontsize)

    ax.set_xticks([0, 50, 100, 150, 200])
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
    save_path = os.path.join(OUTPUT_DIR, "convergence_r2_mlp_plot.svg")
    fig.savefig(save_path)
    plt.close(fig)

    print(f"[Saved] {save_path}")


def plot_all_convergence_r2() -> None:
    """Plot RR / RF / MLP figures sequentially (logic identical to three separate scripts)."""
    here = Path(__file__).resolve().parent
    prev_cwd = os.getcwd()
    os.chdir(here)
    try:
        plot_convergence_r2_rr()
        plot_convergence_r2_rf()
        plot_convergence_r2_mlp()
    finally:
        os.chdir(prev_cwd)


if __name__ == "__main__":
    plot_all_convergence_r2()

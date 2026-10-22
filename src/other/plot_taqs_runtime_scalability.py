from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

# ==============================
# Path config
# ==============================
EXCEL_PATH = "../../results/taqs_runtime_scalability.xlsx"
OUTPUT_DIR = "../../results/taqs_runtime_scalability"

# Correspondence: (sheet name, x-axis column, figure suffix) in plot order
SCALING_SHEETS = [
    ("n_scaling", "n"),
    ("m_scaling", "m"),
    ("d_scaling", "d"),
]

TIME_COL = "Time_Median"
MEMORY_COL = "PeakMemory_Median_MB"

MARKERS = ["s", "o"]
COLOR_TIME = "#d62728"
COLOR_MEMORY = "#1f77b4"


def set_paper_style():
    """Simplified style config aligned with project figures."""
    plt.style.use("default")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def plot_scaling_sheet(
    excel_path: str,
    sheet_name: str,
    x_col: str,
    output_dir: str,
) -> Optional[str]:
    """Read one scaling sheet and plot the dual-axis median time / median memory curve; return save path on success."""
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    required = [x_col, TIME_COL, MEMORY_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[Error] Sheet {sheet_name!r}: missing columns {missing}.")
        return None

    plot_df = df.dropna(subset=required).sort_values(by=x_col)
    if plot_df.empty:
        print(f"[Error] Sheet {sheet_name!r}: no valid rows to plot.")
        return None

    x_values = plot_df[x_col].to_numpy(dtype=float)
    time_values = plot_df[TIME_COL].to_numpy(dtype=float)
    memory_values = plot_df[MEMORY_COL].to_numpy(dtype=float)

    # Dataset label is constant per sheet; used in the title only.
    dataset_label = ""
    if "Dataset" in df.columns and not df["Dataset"].dropna().empty:
        dataset_label = str(df["Dataset"].dropna().iloc[0])

    set_paper_style()
    figsize = (3, 2.2)
    dpi = 600
    fontsize = 12

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax2 = ax.twinx()

    ax.minorticks_on()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    ax.plot(
        x_values,
        time_values,
        color=COLOR_TIME,
        marker=MARKERS[0],
        markersize=3.5,
        linewidth=1.5,
        zorder=2,
    )
    ax2.plot(
        x_values,
        memory_values,
        color=COLOR_MEMORY,
        marker=MARKERS[1],
        markersize=3.5,
        linewidth=1.5,
        zorder=1,
    )

    ax.set_xlabel(f"{x_col}", fontsize=fontsize)
    ax.set_ylabel("Time (s)", fontsize=fontsize, color=COLOR_TIME)
    ax2.set_ylabel("Peak memory (MB)", fontsize=fontsize, color=COLOR_MEMORY)
    title = f"{sheet_name.replace('_', ' ')} ({dataset_label})" if dataset_label else sheet_name.replace("_", " ")
    ax.set_title(title, fontsize=fontsize)

    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize, colors=COLOR_TIME)
    ax2.tick_params(axis="y", labelsize=fontsize, colors=COLOR_MEMORY)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"taqs_runtime_{sheet_name}.jpg")
    fig.savefig(save_path)
    plt.close(fig)

    print(f"[Saved] {save_path}")
    return save_path


def plot_taqs_runtime_scalability(
    excel_path: str = EXCEL_PATH,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """
    Read the n_scaling / m_scaling / d_scaling sheets sequentially and generate plots; missing sheets will be skipped with a warning.
    """
    if not os.path.exists(excel_path):
        print(f"[Error] File not found: {excel_path}")
        return

    xl = pd.ExcelFile(excel_path)
    available = set(xl.sheet_names)

    for sheet_name, x_col in SCALING_SHEETS:
        if sheet_name not in available:
            print(f"[Skip] Sheet {sheet_name!r} not in workbook (have: {sorted(available)}).")
            continue
        plot_scaling_sheet(excel_path, sheet_name, x_col, output_dir)


if __name__ == "__main__":
    plot_taqs_runtime_scalability()

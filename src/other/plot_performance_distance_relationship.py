from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

# ==============================
# Path config
# ==============================
EXCEL_PATH = "../../results/performance_distance_relationship.xlsx"
OUTPUT_DIR = "../../results/insights_into_performance_improvements_figures"

# Correspondence between sheet names and base learner names in the workbook (order determines plot order)
SHEET_MODEL_PAIRS = [
    ("LAQ", "RR"),
    ("DRGD", "RF"),
    ("ATRIV", "MLP"),
]

MARKERS = ["s", "o", "^"]


def set_paper_style():
    """Simplified style config aligned with project figures."""
    plt.style.use("default")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def _n_columns_from_df(df: pd.DataFrame) -> list[str]:
    """Column names like n=10, n=20; return mean column name list (sorted by n)."""
    n_cols = [
        c
        for c in df.columns
        if str(c).startswith("n=") and not str(c).endswith("_std")
    ]
    n_cols = sorted(n_cols, key=lambda c: int(str(c).split("=")[1]))
    return n_cols


def plot_performance_distance_relationship_for_sheet(
    excel_path: str,
    sheet_name: str,
    model_name: str,
    output_dir: str,
) -> Optional[str]:
    """Read a single sheet, plot Sample size–R² and Sample size–Average distance dual-axis plot; return save path on success."""
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    n_cols = _n_columns_from_df(df)
    if len(n_cols) == 0:
        print(f"[Error] Sheet {sheet_name!r}: no n=* columns found.")
        return None

    r2_row = df[
        (df["Model"] == model_name)
        & (df["Split"] == "test")
        & (df["Metric"] == "R2")
    ]
    dist_row = df[
        (df["Model"] == "TAQS")
        & (df["Split"] == "subset")
        & (df["Metric"] == "AvgTopKDistToTest")
    ]
    size_row = df[
        (df["Model"] == "TAQS")
        & (df["Split"] == "subset")
        & (df["Metric"] == "SampleSize")
    ]

    if r2_row.empty:
        print(
            f"[Error] Sheet {sheet_name!r}: missing row "
            f"Model={model_name}, Split=test, Metric=R2"
        )
        return None
    if dist_row.empty:
        print(
            f"[Error] Sheet {sheet_name!r}: missing row "
            "Model=TAQS, Split=subset, Metric=AvgTopKDistToTest"
        )
        return None
    if size_row.empty:
        print(
            f"[Error] Sheet {sheet_name!r}: missing row "
            "Model=TAQS, Split=subset, Metric=SampleSize"
        )
        return None

    r2_values = r2_row.iloc[0][n_cols].to_numpy(dtype=float)
    dist_values = dist_row.iloc[0][n_cols].to_numpy(dtype=float)
    size_values = size_row.iloc[0][n_cols].to_numpy(dtype=float)

    set_paper_style()
    figsize = (3, 2.2)
    dpi = 600
    fontsize = 12

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax2 = ax.twinx()

    ax.minorticks_on()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    ax.plot(
        size_values,
        r2_values,
        color="#d62728",
        marker=MARKERS[0],
        markersize=3.5,
        linewidth=1.5,
        zorder=2,
    )
    ax2.plot(
        size_values,
        dist_values,
        color="#1f77b4",
        marker=MARKERS[1],
        markersize=3.5,
        linewidth=1.5,
        zorder=1,
    )

    ax.set_xlabel("Sample size", fontsize=fontsize)
    ax.set_ylabel("$R^2$", fontsize=fontsize, color="#d62728")
    ax2.set_ylabel("Average distance", fontsize=fontsize, color="#1f77b4")
    ax.set_title(f"{sheet_name} ({model_name})", fontsize=fontsize)

    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize, colors="#d62728")
    ax2.tick_params(axis="y", labelsize=fontsize, colors="#1f77b4")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(
        output_dir,
        f"performance_distance_relationship_{sheet_name}_{model_name}.svg",
    )
    fig.savefig(save_path)
    plt.close(fig)

    print(f"[Saved] {save_path}")
    return save_path


def plot_performance_distance_relationship(
    excel_path: str = EXCEL_PATH,
    output_dir: str = OUTPUT_DIR,
    sheet_model_pairs: Optional[list[tuple[str, str]]] = None,
) -> None:
    """
    Read each sheet sequentially according to SHEET_MODEL_PAIRS and generate plots; missing sheets will be skipped with a warning.
    """
    if not os.path.exists(excel_path):
        print(f"[Error] File not found: {excel_path}")
        return

    pairs = sheet_model_pairs or list(SHEET_MODEL_PAIRS)
    xl = pd.ExcelFile(excel_path)
    available = set(xl.sheet_names)

    for sheet_name, model_name in pairs:
        if sheet_name not in available:
            print(f"[Skip] Sheet {sheet_name!r} not in workbook (have: {sorted(available)}).")
            continue
        plot_performance_distance_relationship_for_sheet(
            excel_path, sheet_name, model_name, output_dir
        )


if __name__ == "__main__":
    plot_performance_distance_relationship()

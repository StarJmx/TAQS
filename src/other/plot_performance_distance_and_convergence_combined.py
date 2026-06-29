"""
Merge ``plot_performance_distance_relationship.py`` (first row) with
``plot_convergence_analysis_taqs_all.py`` (second row):
Left side is a narrow text placeholder column, right side is three-column data subplots (2×(1+3) grid); plotting parameters are consistent with original scripts.

- First row: no legend displayed.
- Second row: only the first data subplot (RR) retains the legend.
- The leftmost column of each row is a placeholder axis, which can be filled with descriptive text via
  ``left_gutter_row0_text`` / ``left_gutter_row1_text`` parameters of ``plot_combined_performance_and_convergence``.

Recommended working directory: ``src/other`` (same as source scripts, relative paths depend on Excel).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================
# Paths and constants (consistent with original files)
# ==============================
PERF_EXCEL_PATH = "../../results/performance_distance_relationship.xlsx"
EXCEL_PATH_RR = "../../results/convergence_analysis_taqs_rr.xlsx"
EXCEL_PATH_RF = "../../results/convergence_analysis_taqs_rf.xlsx"
EXCEL_PATH_MLP = "../../results/convergence_analysis_taqs_mlp.xlsx"
OUTPUT_DIR = "../../results/insights_into_performance_improvements_figures"

SHEET_MODEL_PAIRS = [
    ("LAQ", "RR"),
    ("DRGD", "RF"),
    ("ATRIV", "MLP"),
]

MARKERS_PERF = ["s", "o", "^"]

COLOR_PALETTE = ["#d62728", "#1f77b4"]
MARKERS_CONV = ["s", "o"]
FIGSIZE_SINGLE = (3, 2.2)
DPI = 600
FONTSIZE = 12

# Combined figure total size: left placeholder narrow column + three single-plot columns; height 2×single plot height
LEFT_GUTTER_RATIO = 0.16
FIGSIZE_COMBINED = (
    FIGSIZE_SINGLE[0] * (3 + LEFT_GUTTER_RATIO),
    FIGSIZE_SINGLE[1] * 2,
)

LEGEND_LOC_CONV = "best"
LEGEND_FONTSIZE_CONV = 10

# Left placeholder column width ratio (relative to three data axes)
WIDTH_RATIOS = [LEFT_GUTTER_RATIO, 1.0, 1.0, 1.0]


def _style_left_gutter_ax(ax: plt.Axes, text: str, rotation: float = 90.0) -> None:
    """Left narrow column: no coordinate ticks, used to display custom characters centered."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        rotation=rotation,
        fontsize=FONTSIZE,
        clip_on=False,
    )


def set_paper_style() -> None:
    plt.style.use("default")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def _n_columns_from_df(df: pd.DataFrame) -> list[str]:
    n_cols = [
        c
        for c in df.columns
        if str(c).startswith("n=") and not str(c).endswith("_std")
    ]
    n_cols = sorted(n_cols, key=lambda c: int(str(c).split("=")[1]))
    return n_cols


def _load_performance_distance_arrays(
    excel_path: str,
    sheet_name: str,
    model_name: str,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """返回 (size_values, r2_values, dist_values)；失败返回 None。"""
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

    if r2_row.empty or dist_row.empty or size_row.empty:
        print(f"[Error] Sheet {sheet_name!r}: required rows missing.")
        return None

    r2_values = r2_row.iloc[0][n_cols].to_numpy(dtype=float)
    dist_values = dist_row.iloc[0][n_cols].to_numpy(dtype=float)
    size_values = size_row.iloc[0][n_cols].to_numpy(dtype=float)
    return size_values, r2_values, dist_values


def _draw_performance_distance_on_ax(
    ax: plt.Axes,
    size_values: np.ndarray,
    r2_values: np.ndarray,
    dist_values: np.ndarray,
    sheet_name: str,
    model_name: str,
) -> None:
    """与 ``plot_performance_distance_relationship_for_sheet`` 中绘图参数一致（无图例）。"""
    ax2 = ax.twinx()
    ax.minorticks_on()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    ax.plot(
        size_values,
        r2_values,
        color="#d62728",
        marker=MARKERS_PERF[0],
        markersize=3.5,
        linewidth=1.5,
        zorder=2,
    )
    ax2.plot(
        size_values,
        dist_values,
        color="#1f77b4",
        marker=MARKERS_PERF[1],
        markersize=3.5,
        linewidth=1.5,
        zorder=1,
    )

    ax.set_xlabel("Sample size", fontsize=FONTSIZE)
    ax.set_ylabel("R²", fontsize=FONTSIZE, color="#d62728")
    ax2.set_ylabel("Average distance", fontsize=FONTSIZE, color="#1f77b4")
    ax.set_title(f"{sheet_name} ({model_name})", fontsize=FONTSIZE)

    ax.tick_params(axis="x", labelsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=FONTSIZE, colors="#d62728")
    ax2.tick_params(axis="y", labelsize=FONTSIZE, colors="#1f77b4")


def _parse_convergence_rr(df: pd.DataFrame) -> Optional[tuple[np.ndarray, list, list, list]]:
    df_r2 = df[df["Metric"] == "R2"].copy()
    if df_r2.empty:
        return None
    x_cols: list[str] = []
    x_values: list[float] = []
    for col in df.columns[2:]:
        if "_std" in col or "=" not in col:
            continue
        param_name, _, raw = str(col).partition("=")
        if param_name not in ("alpha", "iter"):
            continue
        try:
            x_val = float(raw)
        except ValueError:
            continue
        x_values.append(x_val)
        x_cols.append(col)
    order = np.argsort(x_values)
    x_values_arr = np.array(x_values, dtype=float)[order]
    x_cols = [x_cols[i] for i in order]

    methods = df_r2["Method"].unique()
    means_list: list[np.ndarray] = []
    stds_list: list[np.ndarray] = []
    method_names: list[Any] = []
    for method in methods:
        row = df_r2[df_r2["Method"] == method]
        if row.empty:
            continue
        mean_vals = [row[mean_col].values[0] for mean_col in x_cols]
        std_vals = [row[f"{mean_col}_std"].values[0] for mean_col in x_cols]
        means_list.append(np.array(mean_vals, dtype=float))
        stds_list.append(np.array(std_vals, dtype=float))
        method_names.append(method)
    return x_values_arr, means_list, stds_list, method_names


def _parse_convergence_rf_mlp(df: pd.DataFrame) -> Optional[tuple[np.ndarray, list, list, list]]:
    df_r2 = df[df["Metric"] == "R2"].copy()
    if df_r2.empty:
        return None
    x_cols: list[str] = []
    x_values: list[float] = []
    for col in df.columns[2:]:
        if "_std" not in col:
            n_val = int(str(col).split("=")[1])
            x_values.append(n_val)
            x_cols.append(col)
    x_values_arr = np.array(x_values, dtype=float)

    methods = df_r2["Method"].unique()
    means_list: list[np.ndarray] = []
    stds_list: list[np.ndarray] = []
    method_names: list[Any] = []
    for method in methods:
        row = df_r2[df_r2["Method"] == method]
        if row.empty:
            continue
        mean_vals = [row[mean_col].values[0] for mean_col in x_cols]
        std_vals = [row[f"{mean_col}_std"].values[0] for mean_col in x_cols]
        means_list.append(np.array(mean_vals, dtype=float))
        stds_list.append(np.array(std_vals, dtype=float))
        method_names.append(method)
    return x_values_arr, means_list, stds_list, method_names


def _draw_convergence_rr_on_ax(
    ax: plt.Axes,
    parsed: tuple[np.ndarray, list, list, list],
    show_legend: bool,
) -> None:
    x_values, means_list, stds_list, method_names = parsed
    ax.minorticks_on()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    for i, name in enumerate(method_names):
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        marker = MARKERS_CONV[i % len(MARKERS_CONV)]
        ax.fill_between(
            x_values,
            means_list[i] - stds_list[i],
            means_list[i] + stds_list[i],
            color=color,
            alpha=0.2,
            zorder=1,
        )
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
    ax.set_xlabel(r"alpha (RR)", fontsize=FONTSIZE)
    ax.set_ylabel(r"R$^2$", fontsize=FONTSIZE)
    ax.set_xscale("log")
    ax.set_xticks(x_values)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="both", labelsize=FONTSIZE)
    if show_legend:
        ax.legend(
            loc=LEGEND_LOC_CONV,
            fontsize=LEGEND_FONTSIZE_CONV,
            frameon=False,
            framealpha=0.3,
        )


def _draw_convergence_rf_on_ax(
    ax: plt.Axes,
    parsed: tuple[np.ndarray, list, list, list],
    show_legend: bool,
) -> None:
    x_values, means_list, stds_list, method_names = parsed
    ax.minorticks_on()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    for i, name in enumerate(method_names):
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        marker = MARKERS_CONV[i % len(MARKERS_CONV)]
        ax.fill_between(
            x_values,
            means_list[i] - stds_list[i],
            means_list[i] + stds_list[i],
            color=color,
            alpha=0.2,
            zorder=1,
        )
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
    ax.set_xlabel("n_estimators (RF)", fontsize=FONTSIZE)
    ax.set_ylabel(r"R$^2$", fontsize=FONTSIZE)
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.tick_params(axis="both", labelsize=FONTSIZE)
    if show_legend:
        ax.legend(
            loc=LEGEND_LOC_CONV,
            fontsize=LEGEND_FONTSIZE_CONV,
            frameon=False,
            framealpha=0.3,
        )


def _draw_convergence_mlp_on_ax(
    ax: plt.Axes,
    parsed: tuple[np.ndarray, list, list, list],
    show_legend: bool,
) -> None:
    x_values, means_list, stds_list, method_names = parsed
    ax.minorticks_on()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    for i, name in enumerate(method_names):
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        marker = MARKERS_CONV[i % len(MARKERS_CONV)]
        ax.fill_between(
            x_values,
            means_list[i] - stds_list[i],
            means_list[i] + stds_list[i],
            color=color,
            alpha=0.2,
            zorder=1,
        )
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
    ax.set_xlabel("max_iter (MLP)", fontsize=FONTSIZE)
    ax.set_ylabel(r"R$^2$", fontsize=FONTSIZE)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.tick_params(axis="both", labelsize=FONTSIZE)
    if show_legend:
        ax.legend(
            loc=LEGEND_LOC_CONV,
            fontsize=LEGEND_FONTSIZE_CONV,
            frameon=False,
            framealpha=0.3,
        )


def plot_combined_performance_and_convergence(
    perf_excel: str = PERF_EXCEL_PATH,
    path_rr: str = EXCEL_PATH_RR,
    path_rf: str = EXCEL_PATH_RF,
    path_mlp: str = EXCEL_PATH_MLP,
    output_dir: str = OUTPUT_DIR,
    save_name: str = "performance_distance_and_convergence_combined.jpg",
    left_gutter_row0_text: str = "",
    left_gutter_row1_text: str = "",
    left_gutter_text_rotation: float = 90.0,
) -> Optional[str]:
    """绘制左侧占位 + 2×3 数据子图并保存；失败返回 None。"""
    here = Path(__file__).resolve().parent
    prev_cwd = os.getcwd()
    os.chdir(here)
    try:
        if not os.path.exists(perf_excel):
            print(f"[Error] File not found: {perf_excel}")
            return None

        set_paper_style()
        fig, axes = plt.subplots(
            2,
            4,
            figsize=FIGSIZE_COMBINED,
            dpi=DPI,
            gridspec_kw={"width_ratios": WIDTH_RATIOS},
        )
        _style_left_gutter_ax(
            axes[0, 0],
            left_gutter_row0_text,
            rotation=left_gutter_text_rotation,
        )
        _style_left_gutter_ax(
            axes[1, 0],
            left_gutter_row1_text,
            rotation=left_gutter_text_rotation,
        )

        xl = pd.ExcelFile(perf_excel)
        available_sheets = set(xl.sheet_names)

        # ----- First row: performance / distance (columns 1–3, no legend) -----
        for col, (sheet_name, model_name) in enumerate(SHEET_MODEL_PAIRS):
            ax = axes[0, col + 1]
            if sheet_name not in available_sheets:
                print(
                    f"[Skip] Sheet {sheet_name!r} not in performance workbook "
                    f"(have: {sorted(available_sheets)})."
                )
                ax.set_visible(False)
                continue
            loaded = _load_performance_distance_arrays(perf_excel, sheet_name, model_name)
            if loaded is None:
                ax.set_visible(False)
                continue
            size_values, r2_values, dist_values = loaded
            _draw_performance_distance_on_ax(
                ax,
                size_values,
                r2_values,
                dist_values,
                sheet_name,
                model_name,
            )

        # ----- Second row: convergence RR / RF / MLP (columns 1–3; only RR legend) -----
        conv_specs = [
            (path_rr, _parse_convergence_rr, _draw_convergence_rr_on_ax, "RR"),
            (path_rf, _parse_convergence_rf_mlp, _draw_convergence_rf_on_ax, "RF"),
            (path_mlp, _parse_convergence_rf_mlp, _draw_convergence_mlp_on_ax, "MLP"),
        ]
        for col, (path, parse_fn, draw_fn, tag) in enumerate(conv_specs):
            ax = axes[1, col + 1]
            if not os.path.exists(path):
                print(f"[Error] File not found ({tag}): {path}")
                ax.set_visible(False)
                continue
            df = pd.read_excel(path, sheet_name=0, header=0)
            parsed = parse_fn(df)
            if parsed is None:
                print(f"[Error] No R2 data or parse failed: {path}")
                ax.set_visible(False)
                continue
            draw_fn(ax, parsed, show_legend=(col == 0))

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.45, hspace=0.35)

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_name)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {save_path}")
        return save_path
    finally:
        os.chdir(prev_cwd)


if __name__ == "__main__":
    plot_combined_performance_and_convergence()

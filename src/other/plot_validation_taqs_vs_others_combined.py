import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import utils

# Input file paths
TAQS_EXCEL_PATH = "../../results/validation_taqs_paradigm.xlsx"
RQAS_EXCEL_PATH = "../../results/validation_rqas_paradigm.xlsx"
UTAQS_EXCEL_PATH = "../../results/validation_kmedoids_paradigm.xlsx"
KCEN_EXCEL_PATH = ""
OUTPUT_DIR = "../../results/replot_comparison_taqs_rqas_figures"

METRICS_TO_PLOT = ["R2"]
SHOW_TRAIN_CURVE = False

# Fixed x-axis ticks
FIXED_XTICKS = [0, 50, 100, 150, 200]


plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


def extract_n_columns(df):
    n_cols = [c for c in df.columns if c.startswith("n=") and not c.endswith("_std")]
    n_cols = sorted(n_cols, key=lambda c: int(c.split("=")[1]))
    n_values = [int(c.split("=")[1]) for c in n_cols]
    # Filter out n=1
    n_cols = [c for c, n in zip(n_cols, n_values) if n != 1]
    n_values = [n for n in n_values if n != 1]

    return n_cols, n_values


def get_sample_size_mapping(df, n_cols):
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


def set_paper_style():
    """Simplified version of utils.set_paper_style"""
    plt.style.use("default")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


# =========================
# Subplot plotting function (core)
# =========================
def plot_comparison_subplot(
    ax,
    n_values,
    taqs_train,
    taqs_test,
    rqas_train,
    rqas_test,
    utaqs_train,
    utaqs_test,
    kcen_train,
    kcen_test,
    baseline_value,
    baseline_size,
    sample_size_map,
    ylabel,
    show_train_curve=False,
    show_ylabel=False,
    show_legend=False,
):
    cfg = utils.load_plot_config()
    curve_cfg = cfg.get("plot_train_test_curve", {})

    marker = curve_cfg.get("marker", "o")
    scatter_size = curve_cfg.get("scatter_size")
    line_width = curve_cfg.get("line_width", 1)
    marker_cfg = curve_cfg.get("markers")

    markers = marker_cfg if isinstance(marker_cfg, list) else [marker_cfg]

    # Style
    set_paper_style()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if show_ylabel:
        ax.set_ylabel(ylabel)

    # TAQS
    if taqs_test is not None:
        ax.plot(
            n_values,
            taqs_test,
            color=curve_cfg.get("test_line_color"),
            # color="#fd763f",
            marker=markers[5],
            markersize=scatter_size,
            linewidth=line_width,
            label = "TAQS (Ours)" if show_legend else "_nolegend_",
            zorder=2,
        )

    if show_train_curve and taqs_train is not None:
        ax.plot(
            n_values,
            taqs_train,
            color=curve_cfg.get("train_line_color"),
            linestyle="--",
            marker=markers[5],
            markersize=scatter_size,
            linewidth=line_width,
            label="TAQS (train)",
        )

    # RQAS
    if rqas_test is not None:
        ax.plot(
            n_values,
            rqas_test,
            color="#1f77b4",
            marker=markers[1],
            markersize=scatter_size,
            linewidth=line_width,
            label = "Random" if show_legend else "_nolegend_",
            zorder=1,
        )

    if show_train_curve and rqas_train is not None:
        ax.plot(
            n_values,
            rqas_train,
            color="darkviolet",
            linestyle="--",
            marker=markers[1],
            markersize=scatter_size,
            linewidth=line_width,
            label="RQAS (train)",
        )

    # UTAQS
    if utaqs_test is not None:
        ax.plot(
            n_values,
            utaqs_test,
            color="#7ea6d9",
            marker=markers[0],
            markersize=scatter_size,
            linewidth=line_width,
            label = "K-Medoids" if show_legend else "_nolegend_",
            zorder=1,
        )

    if show_train_curve and utaqs_train is not None:
        ax.plot(
            n_values,
            utaqs_train,
            color="darkviolet",
            linestyle="--",
            marker=markers[1],
            markersize=scatter_size,
            linewidth=line_width,
            label="UTAQS (train)",
        )

    # KCEN
    if kcen_test is not None:
        ax.plot(
            n_values,
            kcen_test,
            color="#e377c2",
            marker=markers[3],
            markersize=scatter_size,
            linewidth=line_width,
            label = "K-Center" if show_legend else "_nolegend_",
            zorder=1,
        )

    if show_train_curve and kcen_train is not None:
        ax.plot(
            n_values,
            kcen_train,
            color="darkviolet",
            linestyle="--",
            marker=markers[3],
            markersize=scatter_size,
            linewidth=line_width,
            label="KCEN (train)",
        )

    # Baseline
    baseline_line = None

    if pd.notna(baseline_value):
        if baseline_size is not None:
            label = f"Baseline (N={int(baseline_size)})"
        else:
            label = "Baseline"

        baseline_line = ax.axhline(
            y=float(baseline_value),
            color="#7f7f7f",
            linestyle="--",
            linewidth=2,
            label=label,
            zorder=0,
        )

    # X-ticks (simplified to avoid crowding)
    ax.set_xticks(FIXED_XTICKS)

    labels = []
    for n in FIXED_XTICKS:
        if n == 0:
            labels.append("0\n")  # Keep two-line structure
        else:
            size = sample_size_map.get(n, None)
            if size is not None:
                labels.append(f"{n}\n({int(size)})")  # Two lines
            else:
                labels.append(f"{n}\n")

    ax.set_xticklabels(labels, ha='center', fontsize=8.5)

    """    # 图例处理：baseline 总是显示，TAQS/RQAS 仅在 show_legend=True 时显示
    if show_legend:
        # 显示全部图例（TAQS, RQAS, baseline）
        ax.legend(frameon=False, fontsize=8)
    else:
        # 只显示 baseline 图例
    """
    if baseline_line is not None:
        ax.legend(handles=[baseline_line], frameon=False, fontsize=8, loc="best")

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # %.2f = 2 decimal places, %.1f = 1 decimal place

# =========================
# Plot by model (core)
# =========================
def process_model_wise(datasets, xls_taqs, xls_rqas, xls_utaqs=None, xls_kcen=None):
    set_paper_style()

    # Collect all models
    all_models = set()
    for sheet in datasets:
        df = pd.read_excel(TAQS_EXCEL_PATH, sheet_name=sheet)
        models = [m for m in df["Model"].dropna().unique() if m != "TAQS"]
        all_models.update(models)

    all_models = sorted(all_models)

    for model_name in all_models:
        print(f"[Model] {model_name}")

        for metric_name in METRICS_TO_PLOT:

            n_datasets = len(datasets)

            fig, axes = plt.subplots(
                1,
                n_datasets,
                figsize=(2.3 * n_datasets, 2.2),
                dpi=1000,
            )

            if n_datasets == 1:
                axes = [axes]

            for i, sheet_name in enumerate(datasets):
                df_taqs = pd.read_excel(TAQS_EXCEL_PATH, sheet_name=sheet_name)

                df_rqas = None
                if xls_rqas and sheet_name in xls_rqas.sheet_names:
                    df_rqas = pd.read_excel(RQAS_EXCEL_PATH, sheet_name=sheet_name)

                n_cols, n_values = extract_n_columns(df_taqs)
                if not n_values:
                    continue

                sample_size_map, baseline_size = get_sample_size_mapping(df_taqs, n_cols)

                # TAQS
                taqs_test = df_taqs[
                    (df_taqs["Model"] == model_name) &
                    (df_taqs["Split"] == "test") &
                    (df_taqs["Metric"] == metric_name)
                ]

                if taqs_test.empty:
                    continue

                taqs_test_row = taqs_test.iloc[0]
                taqs_test_vals = [taqs_test_row[col] for col in n_cols]
                baseline_value = taqs_test_row.get("baseline_full_train", np.nan)

                # Train
                taqs_train_vals = None
                if SHOW_TRAIN_CURVE:
                    taqs_train = df_taqs[
                        (df_taqs["Model"] == model_name) &
                        (df_taqs["Split"] == "train") &
                        (df_taqs["Metric"] == metric_name)
                    ]
                    if not taqs_train.empty:
                        taqs_train_vals = [taqs_train.iloc[0][col] for col in n_cols]

                # RQAS
                rqas_test_vals = None
                rqas_train_vals = None

                if df_rqas is not None:
                    rqas_test = df_rqas[
                        (df_rqas["Model"] == model_name) &
                        (df_rqas["Split"] == "test") &
                        (df_rqas["Metric"] == metric_name)
                    ]
                    if not rqas_test.empty:
                        rqas_test_vals = [rqas_test.iloc[0][col] for col in n_cols]

                # UTAQS
                utaqs_test_vals = None
                utaqs_train_vals = None

                if xls_utaqs is not None:
                    df_utaqs = pd.read_excel(UTAQS_EXCEL_PATH,
                                                  sheet_name=sheet_name) if sheet_name in xls_utaqs.sheet_names else None

                    if df_utaqs is not None:
                        utaqs_test = df_utaqs[
                            (df_utaqs["Model"] == model_name) &
                            (df_utaqs["Split"] == "test") &
                            (df_utaqs["Metric"] == metric_name)
                        ]
                        if not utaqs_test.empty:
                            utaqs_test_vals = [utaqs_test.iloc[0][col] for col in n_cols]

                # KCEN
                kcen_test_vals = None
                kcen_train_vals = None

                if xls_kcen is not None:
                    df_kcen = pd.read_excel(KCEN_EXCEL_PATH,
                                                  sheet_name=sheet_name) if sheet_name in xls_kcen.sheet_names else None

                    if df_kcen is not None:
                        kcen_test = df_kcen[
                            (df_kcen["Model"] == model_name) &
                            (df_kcen["Split"] == "test") &
                            (df_kcen["Metric"] == metric_name)
                        ]
                        if not kcen_test.empty:
                            kcen_test_vals = [kcen_test.iloc[0][col] for col in n_cols]

                # Plotting
                plot_comparison_subplot(
                    ax=axes[i],
                    n_values=n_values,
                    taqs_train=taqs_train_vals,
                    taqs_test=taqs_test_vals,
                    rqas_train=rqas_train_vals,
                    rqas_test=rqas_test_vals,
                    utaqs_train=utaqs_train_vals,
                    utaqs_test=utaqs_test_vals,
                    kcen_train=kcen_train_vals,
                    kcen_test=kcen_test_vals,
                    baseline_value=baseline_value,
                    baseline_size=baseline_size,
                    sample_size_map=sample_size_map,
                    ylabel=r"$R^2$",
                    show_train_curve=SHOW_TRAIN_CURVE,
                    show_ylabel=(i == 0),
                    show_legend=(i == 0),
                    # show_legend=(i == 1) if model_name == "MLP" else (i == 0),
                )

                axes[i].set_title(sheet_name, fontsize=9)

                # Set y-axis range for specific subplots
                if model_name == "MLP" and i == 0:
                    axes[i].set_ylim(-0.1, None)
                if model_name == "MLP" and i == 1:
                    axes[i].set_ylim(0.6, None)
                if model_name == "MLP" and i == 2:
                    axes[i].set_ylim(-1.5, None)
                if model_name == "MLP" and i == 3:
                    axes[i].set_ylim(0.2, None)
                if model_name == "MLP" and i == 5:
                    axes[i].set_ylim(0, None)

                if model_name == "RF" and i == 2:
                    axes[i].set_ylim(-0.25, None)
                if model_name == "RF" and i == 3:
                    axes[i].set_ylim(0.4, None)

                if model_name == "RR" and i == 1:
                    axes[i].set_ylim(0.5, None)
                if model_name == "RR" and i == 3:
                    axes[i].set_ylim(-0.5, None)
                if model_name == "RR" and i == 4:
                    axes[i].set_ylim(-0.7, None)
                if model_name == "RR" and i == 5:
                    axes[i].set_ylim(-0.2, None)

            # Global xlabel
            fig.supxlabel("$k$ (Sample size of the training set)", fontsize=10, y=-0.13)

            plt.subplots_adjust(wspace=0.25, top=0.80)

            fig.text(
                0.09, 0.85,
                model_name,
                fontsize=12,
                fontweight='bold',
                transform=fig.transFigure,
            )

            # Global legend (horizontal at the top)
            if model_name == "RR":
                handles, labels = axes[0].get_legend_handles_labels()
                labelscopy=labels
                temp_list = list(labelscopy)
                temp_list[3] = 'Full'
                labelscopy = tuple(temp_list)
                fig.legend(
                    handles,
                    labelscopy,
                    loc="upper center",
                    ncol=len(labels),
                    frameon=False,
                    fontsize=10,
                    bbox_to_anchor=(0.5, 1.05),
                )

            save_path = os.path.join(
                OUTPUT_DIR,
                f"{model_name}_{metric_name}_combined.pdf".replace(" ", "_"),
            )

            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)

            print(f"[Saved] {save_path}")


# =========================
# main
# =========================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(TAQS_EXCEL_PATH):
        print(f"[Error] TAQS file not found: {TAQS_EXCEL_PATH}")
        sys.exit(1)

    xls_taqs = pd.ExcelFile(TAQS_EXCEL_PATH)
    print(f"[Load] {TAQS_EXCEL_PATH}")

    rqas_available = False
    if os.path.exists(RQAS_EXCEL_PATH):
        xls_rqas = pd.ExcelFile(RQAS_EXCEL_PATH)
        rqas_available = True
        print(f"[Load] {RQAS_EXCEL_PATH}")
    else:
        xls_rqas = None
        print(f"[Warning] RQAS file not found")

    datasets = xls_taqs.sheet_names
    print(f"[Datasets] {datasets}")

    utaqs_available = False
    if os.path.exists(UTAQS_EXCEL_PATH):
        xls_utaqs = pd.ExcelFile(UTAQS_EXCEL_PATH)
        utaqs_available = True
        print(f"[Load] {UTAQS_EXCEL_PATH}")
    else:
        xls_utaqs = None
        print(f"[Warning] UTAQS file not found")

    kcen_available = False
    if os.path.exists(KCEN_EXCEL_PATH):
        xls_kcen = pd.ExcelFile(KCEN_EXCEL_PATH)
        kcen_available = True
        print(f"[Load] {KCEN_EXCEL_PATH}")
    else:
        xls_kcen = None
        print(f"[Warning] KCEN file not found")

    process_model_wise(datasets, xls_taqs, xls_rqas, xls_utaqs, xls_kcen)

    print(f"[Done] All figures saved to {OUTPUT_DIR}")
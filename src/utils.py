"""
utils.py

Shared helpers for experiments: Excel export, regression metrics, and plotting.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml


def save_array_to_excel(data, columns, output_path):
    """
    Save a 2D array to an Excel file.

    Parameters
    ----------
    data : array-like (2D)
        Values to write.
    columns : list
        Column names.
    output_path : str
        Destination path (e.g. results/output.xlsx).
    """

    df = pd.DataFrame(data, columns=columns)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_excel(output_path, index=False)

    print(f"[Saved] Data → {output_path}")


def shuffle_and_save_top_n_samples(input_path, output_path, n_samples=5000, random_state=42):
    """
    Load a dataset, shuffle rows, keep top-N samples, and save to local file.

    Supported formats: csv, xls, xlsx.
    """
    input_lower = input_path.lower()
    if input_lower.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_lower.endswith(".xls") or input_lower.endswith(".xlsx"):
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Unsupported input format. Please use CSV or Excel.")

    df_sampled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df_sampled = df_sampled.head(min(n_samples, len(df_sampled)))

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_lower = output_path.lower()
    if output_lower.endswith(".csv"):
        df_sampled.to_csv(output_path, index=False)
    elif output_lower.endswith(".xls") or output_lower.endswith(".xlsx"):
        df_sampled.to_excel(output_path, index=False)
    else:
        raise ValueError("Unsupported output format. Please use CSV or Excel.")

    print(f"[Saved] Shuffled top-{len(df_sampled)} samples -> {output_path}")


def save_feature_importance(feature_names, contributions, output_path):
    """
    Save feature importance scores to Excel (row order preserved).

    Parameters
    ----------
    feature_names : list
        Feature names.
    contributions : array-like
        Importance values aligned with ``feature_names``.
    output_path : str
        Destination Excel path.
    """

    df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contributions
    })

    df.to_excel(output_path, index=False)

    print(f"[Saved] Feature importance → {output_path}")


def save_dict_to_excel(results_dict, output_path):
    """
    Save a metric dictionary to Excel.

    Example
    -------
    {"R2": 0.85, "MAE": 1.23}
    """

    df = pd.DataFrame(list(results_dict.items()), columns=["Metric", "Value"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_excel(output_path, index=False)

    print(f"[Saved] Results → {output_path}")


def print_section(title):
    """Print a titled section divider for logs."""

    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def evaluate_regression(y_true, y_pred, verbose=True):
    """
    Compute regression metrics: MAE, RMSE, and R2.

    Parameters
    ----------
    y_true : array-like
        Ground-truth targets.
    y_pred : array-like
        Predicted targets.
    verbose : bool
        Print metrics when True (default).

    Returns
    -------
    metrics : dict
        Keys: MAE, RMSE, R2.
    """

    import numpy as np
    import sklearn.metrics as sm

    mae = sm.mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(sm.mean_squared_error(y_true, y_pred))
    r2 = sm.r2_score(y_true, y_pred)

    if verbose:
        print(f"MAE: {mae:.8f}")
        print(f"RMSE: {rmse:.8f}")
        print(f"R2: {r2:.8f}")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def summarize_results(results_list):
    """
    Summarize repeated runs as mean ± std for each metric.

    Parameters
    ----------
    results_list : list of dict
        Each dict is the return value of ``evaluate_regression``.

    Returns
    -------
    summary : dict
        Metric name -> (mean, std).
    """
    import numpy as np

    summary = {}
    for key in results_list[0].keys():
        values = [res[key] for res in results_list]
        summary[key] = (np.mean(values), np.std(values))

    for k, (mean, std) in summary.items():
        print(f"{k}: {mean:.8f} ± {std:.8f}")

    return summary


def set_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "text.usetex": False,

        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        "figure.figsize": (3.5, 2.5),

        "lines.linewidth": 1.8,
        "lines.markersize": 5,

        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",

        "legend.frameon": False,

        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def load_plot_config(config_path=None):
    if config_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.normpath(os.path.join(base_dir, "..", "configs", "plot.yaml"))
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_train_test_curve(
    x_values,
    train_values,
    test_values,
    xlabel,
    ylabel,
    train_label="train_set",
    test_label="test_set",
    save_path=None,
    show=True,
    figsize=(3.2, 3.0),
    dpi=300,
    config_path=None,
):
    """
    Reusable plot helper for train/test performance curves.
    """
    cfg = load_plot_config(config_path)
    curve_cfg = cfg.get("plot_train_test_curve", {})

    figsize = tuple(curve_cfg.get("figsize", list(figsize)))
    dpi = curve_cfg.get("dpi", dpi)
    train_scatter_color = curve_cfg.get("train_scatter_color", "green")
    test_scatter_color = curve_cfg.get("test_scatter_color", "blue")
    train_line_color = curve_cfg.get("train_line_color", "darkgreen")
    test_line_color = curve_cfg.get("test_line_color", "dodgerblue")
    marker = curve_cfg.get("marker", "o")
    markers = marker if isinstance(marker, list) else [marker]
    scatter_size = curve_cfg.get("scatter_size", 10)
    line_width = curve_cfg.get("line_width", 1)
    minor_grid_linestyle = curve_cfg.get("minor_grid_linestyle", "--")
    minor_grid_linewidth = curve_cfg.get("minor_grid_linewidth", 0.3)
    minor_grid_alpha = curve_cfg.get("minor_grid_alpha", 0.7)
    xlabel_pad = curve_cfg.get("xlabel_pad", 5)

    set_paper_style()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.minorticks_on()
    ax.grid(
        which="minor",
        linestyle=minor_grid_linestyle,
        linewidth=minor_grid_linewidth,
        alpha=minor_grid_alpha,
    )
    ax.set_xlabel(xlabel, labelpad=xlabel_pad)
    ax.set_ylabel(ylabel)

    for i, (x, train_val, test_val) in enumerate(zip(x_values, train_values, test_values)):
        m = markers[i % len(markers)]
        ax.scatter(x, train_val, color=train_scatter_color, marker=m, s=scatter_size)
        ax.scatter(x, test_val, color=test_scatter_color, marker=m, s=scatter_size)
    
    ax.plot(x_values, train_values, color=train_line_color, linewidth=line_width, label=train_label)
    ax.plot(x_values, test_values, color=test_line_color, linewidth=line_width, label=test_label)
    ax.legend()
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"[Saved] Figure -> {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def load_style_from_yaml(path):
    """
    Load matplotlib rcParams from a yaml file.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    plt.rcParams.update({
        "font.family": cfg["font"]["family"],
        "font.serif": [cfg["font"]["serif"]],
        "font.size": cfg["font"]["size"],
        "axes.labelsize": cfg["axes"]["labelsize"],
        "axes.titlesize": cfg["axes"]["titlesize"],
        "axes.linewidth": cfg["axes"]["linewidth"],
        "axes.grid": cfg["axes"]["grid"],
        "xtick.labelsize": cfg["ticks"]["xtick_labelsize"],
        "ytick.labelsize": cfg["ticks"]["ytick_labelsize"],
        "legend.fontsize": cfg["legend"]["fontsize"],
        "legend.frameon": cfg["legend"]["frameon"],
        "figure.figsize": tuple(cfg["figure"]["figsize"]),
        "lines.linewidth": cfg["lines"]["linewidth"],
        "lines.markersize": cfg["lines"]["markersize"],
        "grid.linestyle": cfg["grid"]["linestyle"],
        "grid.linewidth": cfg["grid"]["linewidth"],
        "savefig.dpi": cfg["savefig"]["dpi"],
        "savefig.bbox": cfg["savefig"]["bbox"],
        "savefig.pad_inches": cfg["savefig"]["pad_inches"],
    })


def plot_feature_importance_and_validation(
    features,
    importance,
    mofs,
    capacities,
    errors,
    colors=None,
    save_path=None,
    show=True,
):
    """
    Reusable two-panel figure:
    A) horizontal feature-importance bars
    B) validation bar chart with error bars
    """
    import numpy as np

    plt.style.use("seaborn-v0_8-paper")
    if colors is None:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    y_pos = np.arange(len(features))
    ax1.barh(y_pos, importance, color=colors[: len(features)], edgecolor="black", alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features)
    ax1.invert_yaxis()
    ax1.set_xlabel("Relative Importance (SHAP value)")
    ax1.set_title("A: Feature Importance Analysis (GBDT Model)", fontsize=12, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.7)

    bar_colors = ["#cccccc", "#1f77b4", "#d62728"]
    bars = ax2.bar(
        mofs,
        capacities,
        yerr=errors,
        color=bar_colors[: len(mofs)],
        edgecolor="black",
        capsize=5,
        width=0.6,
    )
    ax2.set_ylabel("Adsorption Capacity (mg/g)")
    ax2.set_title("B: Screening Results for Cr(VI) Removal", fontsize=12, fontweight="bold")
    ymax = max(capacities) + max(errors) + 30 if len(capacities) > 0 else 1
    ax2.set_ylim(0, ymax)

    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 10,
            f"{height}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"[Saved] Figure -> {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, (ax1, ax2)


def plot_metrics_from_csv(
    csv_path,
    x_col,
    y_cols,
    labels=None,
    xlabel=None,
    ylabel=None,
    save_path=None,
    show=True,
    style_yaml_path=None,
):
    """
    Plot one or more metric curves from a CSV file.
    """
    if style_yaml_path:
        load_style_from_yaml(style_yaml_path)
    else:
        set_paper_style()

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots()

    if labels is None:
        labels = y_cols

    for col, label in zip(y_cols, labels):
        ax.plot(df[x_col], df[col], label=label)

    ax.set_xlabel(xlabel if xlabel is not None else x_col)
    ax.set_ylabel(ylabel if ylabel is not None else "Value")
    ax.legend()
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"[Saved] Figure -> {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax

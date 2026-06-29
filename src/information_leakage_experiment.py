"""
information_leakage_experiment.py

Information Leakage and Weight Stability analysis for SHAP-based feature weights.

Main experiments:
1) Weight Stability Analysis
   - full-data SHAP weights vs subset SHAP weights (repeated)
   - metrics: Spearman correlation, L2 distance, Top-k overlap
2) Leakage Impact Evaluation (A/B/C)
   - Case A (potential leakage): full-data weights fixed for CV
   - Case B (strict no leakage): fold-specific weights computed on each train fold
   - Case C (compromise): one disjoint subset weights fixed for CV
   - metrics: MAE / RMSE / R2
3) Optional randomization test
   - compute weights using shuffled labels, then run CV with fixed random-y weights
"""

import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore", message="ntree_limit is deprecated.*")

import numpy as np

import pandas as pd
import shap
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src import baselines as bl
from src import data_processor as dp
from src import utils


# ==============================
# Config
# ==============================
DATASET_NAMES = [
    "IHDP",
    "MHR",
    "OBES",
]

# Global seed for subsampling, CV splits, and other stochastic steps.
GLOBAL_SEED = 42
# Number of CV folds for leakage-impact evaluation.
N_SPLITS = 10
# Neighbor count k for weighted Gower KNN.
N_NEIGHBORS = 3

# Weight stability: number of repeated subset draws.
STABILITY_REPEATS = 10
# Weight stability: fraction of data used for each subset SHAP fit.
STABILITY_SUBSET_FRAC = 0.8
# Top-k for the feature-overlap stability metric.
TOP_K = 5

# Case C: fraction of data used to compute fixed weights once.
LEAKAGE_SUBSET_FRAC = 0.8
# Whether to run the optional randomized-label weight test.
INCLUDE_RANDOMIZATION_TEST = False

CONFIG_PATH = "../configs/datasets_config.yaml"
OUTPUT_PATH = "../results/information_leakage_experiment.xlsx"


# ==============================
# Common helpers
# ==============================
def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize feature weights to sum to 1.

    Returns uniform weights when the input sum is <= 0.
    """
    total = np.sum(weights)
    if total <= 0:
        return np.ones_like(weights) / len(weights)
    return weights / total


def summarize_metric_dicts(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Summarize repeated metric dicts as mean and std per metric.

    Returns keys in the form ``<metric>_mean`` and ``<metric>_std``.
    """
    summary = {}
    for key in metric_dicts[0].keys():
        values = np.array([d[key] for d in metric_dicts], dtype=float)
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two vectors; np.nan if undefined."""
    s1 = pd.Series(a).rank(method="average")
    s2 = pd.Series(b).rank(method="average")
    corr = s1.corr(s2)
    return float(corr) if corr is not None else np.nan


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Fraction of shared features in the top-k sets of ``a`` and ``b``."""
    k = min(k, len(a), len(b))
    idx_a = set(np.argsort(a)[-k:])
    idx_b = set(np.argsort(b)[-k:])
    return len(idx_a.intersection(idx_b)) / k


def standardize_with_train_stats(
    X_train: np.ndarray, X_test: np.ndarray, num_start_idx: int, num_end_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize quantitative columns using train-fold statistics and apply to test.

    Prevents test-set information leakage into scaling.
    """
    X_train_std = np.array(X_train, dtype=float, copy=True)
    X_test_std = np.array(X_test, dtype=float, copy=True)

    scaler = StandardScaler()
    scaler.fit(X_train_std[:, num_start_idx:num_end_idx + 1])

    X_train_std[:, num_start_idx:num_end_idx + 1] = scaler.transform(
        X_train_std[:, num_start_idx:num_end_idx + 1]
    )
    X_test_std[:, num_start_idx:num_end_idx + 1] = scaler.transform(
        X_test_std[:, num_start_idx:num_end_idx + 1]
    )
    return X_train_std, X_test_std


def standardize_self(
    X: np.ndarray, num_start_idx: int, num_end_idx: int
) -> np.ndarray:
    """Fit StandardScaler on the given matrix and transform quantitative columns."""
    X_std = np.array(X, dtype=float, copy=True)
    scaler = StandardScaler()
    X_std[:, num_start_idx:num_end_idx + 1] = scaler.fit_transform(
        X_std[:, num_start_idx:num_end_idx + 1]
    )
    return X_std


def get_xgb_model(random_state: int) -> XGBRegressor:
    """Build an XGBoost regressor with fixed mid-size hyperparameters for SHAP weights."""
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_estimators=150,
        max_depth=10,
        min_child_weight=3,
        learning_rate=0.1,
        subsample=0.8,
        gamma=0,
        colsample_bytree=1,
        n_jobs=-1,
    )


def compute_shap_weights(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Train XGBoost and return normalized mean-|SHAP| feature weights."""
    model = get_xgb_model(seed)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model, n_jobs=-1)
    shap_values = explainer(X, check_additivity=False).values
    importance = np.mean(np.abs(shap_values), axis=0)
    return normalize_weights(importance)


def build_weighted_gower_model(config: Dict, feature_weights: np.ndarray):
    """Build a TaskAwareWeightedGowerKNNRegressor from dataset config and weights."""
    return bl.TaskAwareWeightedGowerKNNRegressor(
        n_neighbors=N_NEIGHBORS,
        num_start_idx=config["num_start_idx"],
        num_end_idx=config["num_end_idx"],
        cat_end_idx=config["cat_end_idx"],
        feature_weights=feature_weights,
    )


# ==============================
# Experiment 1: Weight Stability
# ==============================
def run_weight_stability(dataset_name: str, config: Dict, X: np.ndarray, y: np.ndarray):
    """
    Compare full-data SHAP weights against repeated random subset weights.

    Returns
    -------
    w_full : np.ndarray
        SHAP weights from the full standardized dataset.
    stability_df : pd.DataFrame
        Per-repeat stability metrics.
    stability_summary_df : pd.DataFrame
        Mean/std summary of stability metrics.
    """
    rng = np.random.default_rng(GLOBAL_SEED)
    n_samples, n_features = X.shape
    top_k = min(TOP_K, n_features)

    X_full_std = standardize_self(X, config["num_start_idx"], config["num_end_idx"])
    w_full = compute_shap_weights(X_full_std, y, seed=GLOBAL_SEED)

    stability_rows = []
    subset_size = max(2, int(n_samples * STABILITY_SUBSET_FRAC))

    for repeat_id in range(STABILITY_REPEATS):
        subset_idx = rng.choice(n_samples, size=subset_size, replace=False)
        X_sub = X[subset_idx]
        y_sub = y[subset_idx]
        X_sub_std = standardize_self(X_sub, config["num_start_idx"], config["num_end_idx"])
        w_sub = compute_shap_weights(X_sub_std, y_sub, seed=GLOBAL_SEED + repeat_id + 1)
        stability_rows.append(
            {
                "Dataset": dataset_name,
                "Repeat": repeat_id + 1,
                "SubsetFrac": STABILITY_SUBSET_FRAC,
                "Spearman": spearman_corr(w_full, w_sub),
                "L2": float(np.linalg.norm(w_full - w_sub)),
                "TopKOverlap": topk_overlap(w_full, w_sub, top_k),
            }
        )

    stability_df = pd.DataFrame(stability_rows)
    stability_summary = {
        "Dataset": dataset_name,
        "SubsetFrac": STABILITY_SUBSET_FRAC,
        "Repeats": STABILITY_REPEATS,
        "Spearman_mean": float(stability_df["Spearman"].mean()),
        "Spearman_std": float(stability_df["Spearman"].std(ddof=0)),
        "L2_mean": float(stability_df["L2"].mean()),
        "L2_std": float(stability_df["L2"].std(ddof=0)),
        "TopKOverlap_mean": float(stability_df["TopKOverlap"].mean()),
        "TopKOverlap_std": float(stability_df["TopKOverlap"].std(ddof=0)),
    }

    return w_full, stability_df, pd.DataFrame([stability_summary])


# ==============================
# Experiment 2: Leakage Impact A/B/C
# ==============================
def evaluate_cv_with_fixed_weights(
    dataset_name: str,
    case_name: str,
    config: Dict,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
):
    """
    Run K-fold CV with fixed feature weights (Case A, Case C, or randomization).

    Returns per-fold metrics and a cross-fold summary table.
    """
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=GLOBAL_SEED)
    fold_rows = []

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, X_test = standardize_with_train_stats(
            X_train_raw, X_test_raw, config["num_start_idx"], config["num_end_idx"]
        )
        model = build_weighted_gower_model(config, weights)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = utils.evaluate_regression(y_test, y_pred, verbose=False)

        fold_rows.append(
            {
                "Dataset": dataset_name,
                "Case": case_name,
                "Fold": fold_id,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    summary = summarize_metric_dicts(fold_df[["MAE", "RMSE", "R2"]].to_dict("records"))
    summary_row = {
        "Dataset": dataset_name,
        "Case": case_name,
        "MAE_mean": summary["MAE_mean"],
        "MAE_std": summary["MAE_std"],
        "RMSE_mean": summary["RMSE_mean"],
        "RMSE_std": summary["RMSE_std"],
        "R2_mean": summary["R2_mean"],
        "R2_std": summary["R2_std"],
    }
    return fold_df, pd.DataFrame([summary_row])


def evaluate_case_b_strict_no_leakage(
    dataset_name: str, config: Dict, X: np.ndarray, y: np.ndarray
):
    """
    Case B: recompute SHAP weights on each train fold before prediction.

    Returns per-fold metrics and a cross-fold summary table.
    """
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=GLOBAL_SEED)
    fold_rows = []

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, X_test = standardize_with_train_stats(
            X_train_raw, X_test_raw, config["num_start_idx"], config["num_end_idx"]
        )
        w_fold = compute_shap_weights(X_train, y_train, seed=GLOBAL_SEED + fold_id)

        model = build_weighted_gower_model(config, w_fold)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = utils.evaluate_regression(y_test, y_pred, verbose=False)

        fold_rows.append(
            {
                "Dataset": dataset_name,
                "Case": "B_foldwise_no_leakage",
                "Fold": fold_id,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    summary = summarize_metric_dicts(fold_df[["MAE", "RMSE", "R2"]].to_dict("records"))
    summary_row = {
        "Dataset": dataset_name,
        "Case": "B_foldwise_no_leakage",
        "MAE_mean": summary["MAE_mean"],
        "MAE_std": summary["MAE_std"],
        "RMSE_mean": summary["RMSE_mean"],
        "RMSE_std": summary["RMSE_std"],
        "R2_mean": summary["R2_mean"],
        "R2_std": summary["R2_std"],
    }
    return fold_df, pd.DataFrame([summary_row])


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    processor = dp.DataProcessor()
    rng = np.random.default_rng(GLOBAL_SEED)

    all_stability_runs = []
    all_stability_summary = []
    all_weight_stats = []

    all_leakage_fold = []
    all_leakage_summary = []

    for dataset_name in DATASET_NAMES:
        utils.print_section(f"Dataset: {dataset_name}")
        config = dp.load_dataset_config(CONFIG_PATH, dataset_name)
        feature_names, data = processor.load_data(config["file_path"], random_state=GLOBAL_SEED)

        X = np.array(data[:, :-1], dtype=float)
        y = np.array(data[:, -1], dtype=float)
        n_samples = X.shape[0]

        print("[1/2] Weight Stability Analysis")
        w_full, stability_df, stability_summary_df = run_weight_stability(
            dataset_name, config, X, y
        )
        all_stability_runs.append(stability_df)
        all_stability_summary.append(stability_summary_df)

        print("[2/2] Leakage Impact Evaluation (A/B/C)")
        # Case A: full-data SHAP weights fixed
        fold_a_df, summary_a_df = evaluate_cv_with_fixed_weights(
            dataset_name,
            "A_full_weights_potential_leakage",
            config,
            X,
            y,
            w_full,
        )

        # Case B: strict no leakage (fold-wise weights)
        fold_b_df, summary_b_df = evaluate_case_b_strict_no_leakage(
            dataset_name, config, X, y
        )

        # Case C: disjoint subset SHAP weights fixed
        subset_size = max(2, int(n_samples * LEAKAGE_SUBSET_FRAC))
        subset_idx = rng.choice(n_samples, size=subset_size, replace=False)
        X_sub = standardize_self(X[subset_idx], config["num_start_idx"], config["num_end_idx"])
        y_sub = y[subset_idx]
        w_sub = compute_shap_weights(X_sub, y_sub, seed=GLOBAL_SEED + 500)
        fold_c_df, summary_c_df = evaluate_cv_with_fixed_weights(
            dataset_name,
            "C_subset_weights_compromise",
            config,
            X,
            y,
            w_sub,
        )

        all_leakage_fold.extend([fold_a_df, fold_b_df, fold_c_df])
        all_leakage_summary.extend([summary_a_df, summary_b_df, summary_c_df])

        # Optional randomization test: shuffled-y weights
        if INCLUDE_RANDOMIZATION_TEST:
            y_shuffled = rng.permutation(y)
            X_full_std = standardize_self(X, config["num_start_idx"], config["num_end_idx"])
            w_rand = compute_shap_weights(X_full_std, y_shuffled, seed=GLOBAL_SEED + 999)
            fold_r_df, summary_r_df = evaluate_cv_with_fixed_weights(
                dataset_name,
                "R_randomized_label_weights",
                config,
                X,
                y,
                w_rand,
            )
            all_leakage_fold.append(fold_r_df)
            all_leakage_summary.append(summary_r_df)

        # Save full weights for traceability
        full_weight_df = pd.DataFrame(
            {
                "Dataset": dataset_name,
                "Feature": feature_names[:-1],
                "FeatureIndex": list(range(len(w_full))),
                "Weight_full": w_full,
            }
        )
        all_weight_stats.append(full_weight_df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH) as writer:
        pd.concat(all_stability_runs, ignore_index=True).to_excel(
            writer, sheet_name="stability_runs", index=False
        )
        pd.concat(all_stability_summary, ignore_index=True).to_excel(
            writer, sheet_name="stability_summary", index=False
        )
        pd.concat(all_weight_stats, ignore_index=True).to_excel(
            writer, sheet_name="weight_stats", index=False
        )
        pd.concat(all_leakage_fold, ignore_index=True).to_excel(
            writer, sheet_name="leakage_fold_metrics", index=False
        )
        pd.concat(all_leakage_summary, ignore_index=True).to_excel(
            writer, sheet_name="leakage_summary", index=False
        )

    print(f"\n[Saved] -> {OUTPUT_PATH}")
    print("=== Done ===")

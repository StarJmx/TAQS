"""
convergence_analysis_taqs_rr.py

Ridge regularization sweep on TAQS subsets vs full training data (blocked CV).
Ridge has no training epochs; curve sheets use epoch=1 as a schema placeholder.
"""

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src import data_processor as dp
from src import taqs_algorithm as taqs
from src import utils

THREADS_PER_PROCESS = int(os.environ.get("TAQS_THREADS_PER_PROCESS", "4"))
os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_PROCESS)

DATASET_NAME = "LAQ"
STEP = 0.02
N_FOLDS = int(1 / STEP)

N_NEIGHBORS = 25

ALPHA_LIST = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

random_seeds = [42, 123, 999]
MAX_WORKERS = min(len(random_seeds), max(1, os.cpu_count() - 1))


def _to_optional_int(value):
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("none", "null", ""):
            return None
        return int(v)
    return int(value)


def run_fold_experiment(data, config, seed):
    """Run blocked CV for one seed; sweep Ridge alpha on TAQS subset and full train."""
    test_size = int(data.shape[0] * STEP)

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    curve_rows = []

    for fold_idx in range(N_FOLDS):
        start = fold_idx * test_size
        end = (fold_idx + 1) * test_size

        if fold_idx % 10 == 0:
            print(f"[Seed {seed}] Fold {fold_idx+1}/{N_FOLDS}")

        data_test = data[start:end]
        data_train = np.concatenate([data[:start], data[end:]], axis=0)

        # Fold-internal scaling: fit on current train split and transform numeric columns only.
        train_x = data_train[:, :-1]
        train_y = data_train[:, -1]
        test_x = data_test[:, :-1]
        test_y = data_test[:, -1]

        train_x_std = train_x.copy()
        test_x_std = test_x.copy()
        num_slice = slice(config["num_start_idx"], config["num_end_idx"] + 1)

        scaler = StandardScaler()
        scaler.fit(train_x[:, num_slice].astype(float))
        train_x_std[:, num_slice] = scaler.transform(train_x[:, num_slice].astype(float))
        test_x_std[:, num_slice] = scaler.transform(test_x[:, num_slice].astype(float))

        data_train = np.hstack([train_x_std, train_y.reshape(-1, 1)])
        data_test = np.hstack([test_x_std, test_y.reshape(-1, 1)])

        if len(data_test) == 0 or len(data_train) == 0:
            continue

        test_x = data_test[:, :-1]
        test_y = data_test[:, -1]

        selector = taqs.TaskAwareQueryAnchoredSelection(
            data_task=data_test,
            data_train=data_train,
            n_neighbors=N_NEIGHBORS,
            num_start_idx=config["num_start_idx"],
            num_end_idx=config["num_end_idx"],
            cat_end_idx=_to_optional_int(config["cat_end_idx"]),
            feature_weights=config["feature_weights"],
        )

        subset = selector.select_subset()
        subset_x = subset[:, :-1]
        subset_y = subset[:, -1]

        full_x = data_train[:, :-1]
        full_y = data_train[:, -1]

        for alpha in ALPHA_LIST:
            model_taqs = Ridge(alpha=alpha, random_state=seed)
            model_taqs.fit(subset_x, subset_y)
            pred = model_taqs.predict(test_x)
            metrics = utils.evaluate_regression(test_y, pred, verbose=False)

            for m in ["RMSE", "R2"]:
                results["TAQS_RR"]["test"][m, alpha].append(metrics[m])

            curve_rows.append(
                {
                    "method": "TAQS_RR",
                    "seed": seed,
                    "fold": fold_idx,
                    "alpha": alpha,
                    "epoch": 1,
                    "train_loss": np.nan,
                    "val_score": np.nan,
                    "stopped_at": 1,
                    "best_val_score": np.nan,
                }
            )

            model_full = Ridge(alpha=alpha, random_state=seed)
            model_full.fit(full_x, full_y)
            pred_full = model_full.predict(test_x)
            metrics_full = utils.evaluate_regression(test_y, pred_full, verbose=False)

            for m in ["RMSE", "R2"]:
                results["Baseline_RR"]["test"][m, alpha].append(metrics_full[m])

            curve_rows.append(
                {
                    "method": "Baseline_RR",
                    "seed": seed,
                    "fold": fold_idx,
                    "alpha": alpha,
                    "epoch": 1,
                    "train_loss": np.nan,
                    "val_score": np.nan,
                    "stopped_at": 1,
                    "best_val_score": np.nan,
                }
            )

    seed_rows = []

    for method in ["TAQS_RR", "Baseline_RR"]:
        for metric in ["RMSE", "R2"]:
            for alpha in ALPHA_LIST:
                vals = results[method]["test"][(metric, alpha)]
                if len(vals) == 0:
                    continue
                seed_rows.append(
                    (method, "test", metric, alpha, float(np.mean(vals)))
                )

    return seed, seed_rows, curve_rows


def run_seed(dataset_name, seed):
    print(f"[Seed {seed}] Start {dataset_name}")

    config = dp.load_dataset_config("../configs/datasets_config.yaml", dataset_name)
    processor = dp.DataProcessor()

    _, data = processor.load_data(config["file_path"])

    return run_fold_experiment(data, config, seed)


if __name__ == "__main__":

    output_path = "../results/_convergence_analysis_taqs_rr.xlsx"

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    all_curve_rows = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_seed, DATASET_NAME, seed): seed
            for seed in random_seeds
        }

        for future in as_completed(futures):
            seed, seed_rows, curve_rows = future.result()
            print(f"[Seed {seed}] done")

            for method, split, metric, alpha, value in seed_rows:
                results[method][split][metric][alpha].append(value)
            all_curve_rows.extend(curve_rows)

    rows = []
    for method in results:
        for metric in results[method]["test"]:
            row = {
                "Method": method,
                "Metric": metric,
            }
            for alpha in ALPHA_LIST:
                vals = results[method]["test"][metric][alpha]
                row[f"alpha={alpha}"] = float(np.mean(vals))
                row[f"alpha={alpha}_std"] = float(np.std(vals))
            rows.append(row)

    df = pd.DataFrame(rows)

    curve_df = pd.DataFrame(all_curve_rows)
    if curve_df.empty:
        curve_summary_df = pd.DataFrame(
            columns=[
                "method",
                "alpha",
                "epoch",
                "train_loss_mean",
                "train_loss_std",
                "val_score_mean",
                "val_score_std",
            ]
        )
    else:
        curve_summary_df = (
            curve_df.groupby(["method", "alpha", "epoch"], as_index=False)
            .agg(
                train_loss_mean=("train_loss", "mean"),
                train_loss_std=("train_loss", "std"),
                val_score_mean=("val_score", "mean"),
                val_score_std=("val_score", "std"),
            )
            .sort_values(["method", "alpha", "epoch"])
        )

    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name="metrics_summary", index=False)
        curve_df.to_excel(writer, sheet_name="raw_curve", index=False)
        curve_summary_df.to_excel(writer, sheet_name="curve_summary", index=False)

    print(f"[Saved] {output_path}")

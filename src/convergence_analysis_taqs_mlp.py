import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from src import data_processor as dp
from src import taqs_algorithm as taqs
from src import utils

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.neural_network")

# ======================
# Thread control (fully reusing original logic)
# ======================
THREADS_PER_PROCESS = int(os.environ.get("TAQS_THREADS_PER_PROCESS", "4"))
os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_PROCESS)

# ======================
# Experiment configuration
# ======================
DATASET_NAME = "LAQ"
STEP = 0.02
N_FOLDS = int(1 / STEP)

N_NEIGHBORS = 50

MAX_ITER_LIST = [20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 600]

random_seeds = [42, 123, 999]
MAX_WORKERS = min(len(random_seeds), max(1, os.cpu_count() - 1))

BASELINE_KEY = "FULL_TRAIN"


def _to_optional_int(value):
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("none", "null", ""):
            return None
        return int(v)
    return int(value)


# ======================
# Core: single seed experiment
# ======================
def run_fold_experiment(data, config, seed):
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

        # ======================
        # TAQS (only compute once!)
        # ======================
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

        # ======================
        # sweep max_iter
        # ======================
        for max_iter in MAX_ITER_LIST:
            # ---- TAQS ----
            model_taqs = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=max_iter,
                random_state=seed,
                early_stopping=False,
            )
            model_taqs.fit(subset_x, subset_y)
            pred = model_taqs.predict(test_x)
            metrics = utils.evaluate_regression(test_y, pred, verbose=False)

            for m in ["RMSE", "R2"]:
                results["TAQS_MLP"]["test"][m, max_iter].append(metrics[m])

            taqs_val_scores = getattr(model_taqs, "validation_scores_", [])
            taqs_best_val = getattr(model_taqs, "best_validation_score_", np.nan)
            for epoch, train_loss in enumerate(model_taqs.loss_curve_, start=1):
                val_score = (
                    float(taqs_val_scores[epoch - 1])
                    if epoch <= len(taqs_val_scores)
                    else np.nan
                )
                curve_rows.append(
                    {
                        "method": "TAQS_MLP",
                        "seed": seed,
                        "fold": fold_idx,
                        "max_iter": max_iter,
                        "epoch": epoch,
                        "train_loss": float(train_loss),
                        "val_score": val_score,
                        "stopped_at": int(model_taqs.n_iter_),
                        "best_val_score": float(taqs_best_val) if not np.isnan(taqs_best_val) else np.nan,
                    }
                )

            # ---- BASELINE (FULL TRAIN) ----
            model_full = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=max_iter,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-4,
            )
            model_full.fit(full_x, full_y)
            pred_full = model_full.predict(test_x)
            metrics_full = utils.evaluate_regression(test_y, pred_full, verbose=False)

            for m in ["RMSE", "R2"]:
                results["Baseline_MLP"]["test"][m, max_iter].append(metrics_full[m])

            full_val_scores = getattr(model_full, "validation_scores_", [])
            full_best_val = getattr(model_full, "best_validation_score_", np.nan)
            for epoch, train_loss in enumerate(model_full.loss_curve_, start=1):
                val_score = (
                    float(full_val_scores[epoch - 1])
                    if epoch <= len(full_val_scores)
                    else np.nan
                )
                curve_rows.append(
                    {
                        "method": "Baseline_MLP",
                        "seed": seed,
                        "fold": fold_idx,
                        "max_iter": max_iter,
                        "epoch": epoch,
                        "train_loss": float(train_loss),
                        "val_score": val_score,
                        "stopped_at": int(model_full.n_iter_),
                        "best_val_score": float(full_best_val) if not np.isnan(full_best_val) else np.nan,
                    }
                )

    # ======================
    # Summary (per seed)
    # ======================
    seed_rows = []

    for method in ["TAQS_MLP", "Baseline_MLP"]:
        for metric in ["RMSE", "R2"]:
            for max_iter in MAX_ITER_LIST:
                vals = results[method]["test"][(metric, max_iter)]
                if len(vals) == 0:
                    continue
                seed_rows.append(
                    (method, "test", metric, max_iter, float(np.mean(vals)))
                )

    return seed, seed_rows, curve_rows


def run_seed(dataset_name, seed):
    print(f"[Seed {seed}] Start {dataset_name}")

    config = dp.load_dataset_config("../configs/datasets_config.yaml", dataset_name)
    processor = dp.DataProcessor()

    _, data = processor.load_data(config["file_path"])

    return run_fold_experiment(data, config, seed)


# ======================
# Main program
# ======================
if __name__ == "__main__":

    output_path = "../results/_convergence_analysis_taqs_mlp.xlsx"

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

            for method, split, metric, max_iter, value in seed_rows:
                results[method][split][metric][max_iter].append(value)
            all_curve_rows.extend(curve_rows)

    # ======================
    # Aggregate mean/std
    # ======================
    rows = []
    for method in results:
        for metric in results[method]["test"]:
            row = {
                "Method": method,
                "Metric": metric,
            }
            for max_iter in MAX_ITER_LIST:
                vals = results[method]["test"][metric][max_iter]
                row[f"iter={max_iter}"] = float(np.mean(vals))
                row[f"iter={max_iter}_std"] = float(np.std(vals))
            rows.append(row)

    df = pd.DataFrame(rows)

    curve_df = pd.DataFrame(all_curve_rows)
    if curve_df.empty:
        curve_summary_df = pd.DataFrame(
            columns=[
                "method",
                "max_iter",
                "epoch",
                "train_loss_mean",
                "train_loss_std",
                "val_score_mean",
                "val_score_std",
            ]
        )
    else:
        curve_summary_df = (
            curve_df.groupby(["method", "max_iter", "epoch"], as_index=False)
            .agg(
                train_loss_mean=("train_loss", "mean"),
                train_loss_std=("train_loss", "std"),
                val_score_mean=("val_score", "mean"),
                val_score_std=("val_score", "std"),
            )
            .sort_values(["method", "max_iter", "epoch"])
        )

    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name="metrics_summary", index=False)
        curve_df.to_excel(writer, sheet_name="raw_curve", index=False)
        curve_summary_df.to_excel(writer, sheet_name="curve_summary", index=False)

    print(f"[Saved] {output_path}")

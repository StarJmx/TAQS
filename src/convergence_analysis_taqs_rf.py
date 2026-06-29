import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import data_processor as dp
from src import taqs_algorithm as taqs
from src import utils

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

N_NEIGHBORS = 40

N_ESTIMATORS_LIST = [10, 20, 50, 100, 150, 200, 300, 500]

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
        # sweep n_estimators
        # ======================
        for n_est in N_ESTIMATORS_LIST:

            # ---- TAQS ----
            model_taqs = RandomForestRegressor(
                n_estimators=n_est,
                random_state=seed,
                n_jobs=THREADS_PER_PROCESS,
            )
            model_taqs.fit(subset_x, subset_y)
            pred = model_taqs.predict(test_x)
            metrics = utils.evaluate_regression(test_y, pred, verbose=False)

            for m in ["RMSE", "R2"]:
                results["TAQS"]["test"][m, n_est].append(metrics[m])

            # ---- BASELINE (FULL TRAIN) ----
            model_full = RandomForestRegressor(
                n_estimators=n_est,
                random_state=seed,
                n_jobs=THREADS_PER_PROCESS,
            )
            model_full.fit(full_x, full_y)
            pred_full = model_full.predict(test_x)
            metrics_full = utils.evaluate_regression(test_y, pred_full, verbose=False)

            for m in ["RMSE", "R2"]:
                results["FULL"]["test"][m, n_est].append(metrics_full[m])

    # ======================
        # Summary (per seed)
    # ======================
    seed_rows = []

    for method in ["TAQS", "FULL"]:
        for metric in ["RMSE", "R2"]:
            for n_est in N_ESTIMATORS_LIST:
                vals = results[method]["test"][(metric, n_est)]
                if len(vals) == 0:
                    continue
                seed_rows.append(
                    (method, "test", metric, n_est, float(np.mean(vals)))
                )

    return seed, seed_rows


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

    output_path = "../results/_convergence_analysis_taqs_rf.xlsx"

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_seed, DATASET_NAME, seed): seed
            for seed in random_seeds
        }

        for future in as_completed(futures):
            seed, seed_rows = future.result()
            print(f"[Seed {seed}] done")

            for method, split, metric, n_est, value in seed_rows:
                results[method][split][metric][n_est].append(value)

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
            for n_est in N_ESTIMATORS_LIST:
                vals = results[method]["test"][metric][n_est]
                row[f"n={n_est}"] = float(np.mean(vals))
                row[f"n={n_est}_std"] = float(np.std(vals))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)

    print(f"[Saved] {output_path}")
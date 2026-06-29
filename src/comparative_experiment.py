import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

THREADS_PER_PROCESS = 4
os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS_PER_PROCESS)

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from src import baselines as bl
from src import data_processor as dp
from src import utils


dataset_names = [
    "EVEE",
    "IHDP",
    "MHR",
    "OBES",
    "ROSS"
]

random_seeds = [42, 123, 999]
k_list = list(range(1, 11))
MAX_WORKERS = min(len(random_seeds), max(1, os.cpu_count() - 1))


def get_models(k, config):
    num_cols = list(range(config["num_start_idx"], config["num_end_idx"] + 1))
    cat_start = config["num_end_idx"] + 1
    cat_end = config["cat_end_idx"]
    cat_cols = list(range(cat_start, cat_end + 1)) if cat_end >= cat_start else []

    l1_l2_preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    return {
        "VG": bl.VanillaGowerKNNRegressor(
            n_neighbors=k,
            num_start_idx=config["num_start_idx"],
            num_end_idx=config["num_end_idx"],
            cat_end_idx=config["cat_end_idx"]
        ),
        "DAFIG": bl.DAFIGowerKNNRegressor(
            n_neighbors=k,
            num_start_idx=config["num_start_idx"],
            num_end_idx=config["num_end_idx"],
            cat_end_idx=config["cat_end_idx"]
        ),
        "EWG": bl.EntropyWeightedGowerKNNRegressor(
            n_neighbors=k,
            num_start_idx=config["num_start_idx"],
            num_end_idx=config["num_end_idx"],
            cat_end_idx=config["cat_end_idx"],
        ),
        "TAWG (Ours)": bl.TaskAwareWeightedGowerKNNRegressor(
            n_neighbors=k,
            num_start_idx=config["num_start_idx"],
            num_end_idx=config["num_end_idx"],
            cat_end_idx=config["cat_end_idx"],
            feature_weights=config["feature_weights"]
        ),
        "STAWG": bl.ShuffleTaskAwareWeightedGowerKNNRegressor(
            n_neighbors=k,
            num_start_idx=config["num_start_idx"],
            num_end_idx=config["num_end_idx"],
            cat_end_idx=config["cat_end_idx"],
            feature_weights=config["feature_weights"]
        ),
        "MD": make_pipeline(
            l1_l2_preprocessor,
            KNeighborsRegressor(
                n_neighbors=k,
                metric="manhattan",
                weights="uniform"
            )
        ),
        "ED": make_pipeline(
            l1_l2_preprocessor,
            KNeighborsRegressor(
                n_neighbors=k,
                metric="euclidean",
                weights="uniform"
            )
        )
    }


def cross_validate(model, X, y, num_slice):
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10, shuffle=False)
    results = []

    for train_idx, test_idx in kf.split(X):

        X_train = X[train_idx].copy()
        X_test = X[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()
        scaler.fit(X_train[:, num_slice].astype(float))

        X_train[:, num_slice] = scaler.transform(X_train[:, num_slice].astype(float))
        X_test[:, num_slice] = scaler.transform(X_test[:, num_slice].astype(float))

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        res = utils.evaluate_regression(y_test, pred, verbose=False)
        results.append(res)

    return {
        key: np.mean([r[key] for r in results])
        for key in results[0]
    }


def run_seed_experiment(dataset_name, seed):
    config = dp.load_dataset_config("../configs/datasets_config.yaml", dataset_name)
    processor = dp.DataProcessor()
    _, data = processor.load_data(config["file_path"], random_state=seed)
    num_slice = slice(config["num_start_idx"], config["num_end_idx"] + 1)

    X_raw = data[:, :-1]
    y = data[:, -1]

    seed_rows = []
    for k in k_list:
        models = get_models(k, config)
        for model_name, model in models.items():
            res = cross_validate(model, X_raw, y, num_slice)
            for metric in ["MAE", "RMSE", "R2"]:
                seed_rows.append((model_name, metric, k, res[metric]))

    return seed, seed_rows


if __name__ == "__main__":

    output_path = "../results/_temporary_comparative_experiment.xlsx"

    with pd.ExcelWriter(output_path) as writer:

        for dataset_name in dataset_names:
            print(f"\n===== {dataset_name} =====")

            # results[model][metric][k] -> list over random seeds
            results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(run_seed_experiment, dataset_name, seed): seed
                    for seed in random_seeds
                }

                for future in as_completed(futures):
                    seed, seed_rows = future.result()
                    print(f"[Seed {seed}] done")
                    for model_name, metric, k, value in seed_rows:
                        results[model_name][metric][k].append(value)

            rows = []

            for model_name in results:
                for metric in ["MAE", "RMSE", "R2"]:

                    row = {
                        "Model": model_name,
                        "Metric": metric
                    }

                    for k in k_list:
                        values = results[model_name][metric][k]
                        mean = np.mean(values)
                        std = np.std(values)

                        row[f"k={k}"] = mean
                        row[f"k={k}_std"] = std

                    rows.append(row)

            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=dataset_name[:30], index=False)

    print(f"\n[Saved] → {output_path}")
    print("=== Done ===")

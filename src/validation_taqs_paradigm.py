import os
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Control threads used inside each worker process.
# You can override by setting env var `TAQS_THREADS_PER_PROCESS`.
THREADS_PER_PROCESS = int(os.environ.get("TAQS_THREADS_PER_PROCESS", "4"))
os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS_PER_PROCESS)
# Disable dynamic thread downscaling in BLAS backends.
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning

from src import data_processor as dp
from src import taqs_algorithm as taqs
from src import baselines as rqas
from src import utils

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.neural_network")


# Cache tuned MLP hyperparameters to avoid repeated file reads.
_MLP_PARAMS_CACHE = {}
_MLP_PARAMS_LOGGED = set()
# Switch selection method: "taqs" or "rqas".
METHOD_NAME = "taqs"


def load_mlp_params_for_dataset(dataset_name):
    """
    Load tuned MLP hyperparameters for a dataset from mlp_tuning.xlsx.

    Parameters
    ----------
    dataset_name : str
        Dataset name used as the lookup key.

    Returns
    -------
    dict
        MLP hyperparameter dict, or defaults when the dataset is missing.
    """
    global _MLP_PARAMS_CACHE, _MLP_PARAMS_LOGGED

    if not _MLP_PARAMS_CACHE:
        params_file = os.path.join(os.path.dirname(__file__), "..", "results", "mlp_tuning.xlsx")
        if os.path.exists(params_file):
            df = pd.read_excel(params_file, sheet_name="best_params_datasets")
            df.set_index("dataset", inplace=True)
            _MLP_PARAMS_CACHE["df"] = df
            print(f"[Loaded] MLP params cache: {list(df.index)}")
        else:
            print(f"[Warning] MLP params file not found: {params_file}")
            _MLP_PARAMS_CACHE["df"] = None

    df = _MLP_PARAMS_CACHE.get("df")
    if df is None or dataset_name not in df.index:
        if dataset_name not in _MLP_PARAMS_LOGGED:
            print(f"[Warning] Using default MLP params for dataset: {dataset_name}")
            _MLP_PARAMS_LOGGED.add(dataset_name)
        return {
            "hidden_layer_sizes": [128, 61, 32],
            "alpha": 0.001,
            "learning_rate_init": 0.01,
            "max_iter": 200,
        }

    row = df.loc[dataset_name]

    hidden_layers = row["hidden_layer_sizes"]
    if isinstance(hidden_layers, str):
        import ast
        hidden_layers = ast.literal_eval(hidden_layers)
    elif hasattr(hidden_layers, 'tolist'):
        hidden_layers = hidden_layers.tolist()

    if isinstance(hidden_layers, (list, tuple)):
        hidden_layers = [int(x) for x in hidden_layers]

    params = {
        "hidden_layer_sizes": tuple(hidden_layers),
        "alpha": float(row["alpha"]),
        "learning_rate_init": float(row["learning_rate_init"]),
        "max_iter": int(row["max_iter"]),
    }

    if dataset_name not in _MLP_PARAMS_LOGGED:
        print(f"[MLP Params] {dataset_name}: {params}")
        _MLP_PARAMS_LOGGED.add(dataset_name)

    return params


dataset_names = [
    "LAQ",
    "SCD",
    "EVEE",
    "MN1B",
    "ATRIV",
    "DRGD"
]

random_seeds = [42, 123, 999]

STEP = 0.02
# Keep CPU pressure controlled when running 3 seeds in parallel.
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


def build_neighbor_counts():
    # Planned TAQS neighbor settings for sensitivity experiments.
    return [1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200]


def build_models(config, seed, dataset_name=None):
    # Column indices are driven by dataset config to keep code dataset-agnostic.
    num_cols = list(range(config["num_start_idx"], config["num_end_idx"] + 1))
    cat_start = config["num_end_idx"] + 1
    cat_end = _to_optional_int(config["cat_end_idx"])
    cat_cols = list(range(cat_start, cat_end + 1)) if cat_end is not None and cat_end >= cat_start else []

    # RR/MLP need one-hot encoding for categorical inputs.
    rr_preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    mlp_preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    mlp_params = load_mlp_params_for_dataset(dataset_name) if dataset_name else {}

    return {
        "RR": make_pipeline(rr_preprocessor, Ridge()),
        "RF": RandomForestRegressor(
            random_state=seed,
            n_jobs=THREADS_PER_PROCESS,
        ),
        "MLP": make_pipeline(
            mlp_preprocessor,
            MLPRegressor(
                hidden_layer_sizes=mlp_params.get("hidden_layer_sizes", [128, 64, 32]),
                alpha=mlp_params.get("alpha", 0.001),
                learning_rate_init=mlp_params.get("learning_rate_init", 0.01),
                max_iter=mlp_params.get("max_iter", 200),
                random_state=seed,
            ),
        )
    }


def run_fold_experiment(data, config, seed, dataset_name=None):
    neighbor_counts = build_neighbor_counts()
    # 50-fold split: each fold uses a contiguous block of STEP fraction as task set.
    n_folds = int(1 / STEP)
    test_size = int(data.shape[0] * STEP)

    if test_size <= 0:
        raise ValueError(f"Invalid fold size with STEP={STEP}. data_size={data.shape[0]}")

    fold_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    fold_baseline_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    fold_subset_sizes = []
    fold_full_train_sizes = []

    for fold_idx in range(n_folds):
        start_idx = fold_idx * test_size
        end_idx = (fold_idx + 1) * test_size
        if fold_idx % 10 == 0:
            print(
                f"[Seed {seed}] Fold {fold_idx + 1}/{n_folds} "
                f"(test range: {start_idx}:{max(start_idx, end_idx - 1)})"
            )

        # Train set is everything except the current test block.
        data_test = data[start_idx:end_idx]
        data_train = np.concatenate([data[:start_idx], data[end_idx:]], axis=0)

        train_x = data_train[:, :-1]
        train_y = data_train[:, -1]

        test_x = data_test[:, :-1]
        test_y = data_test[:, -1]

        # Standardize quantitative columns only; categorical columns stay unchanged.
        train_x_std = train_x.copy()
        test_x_std = test_x.copy()
        num_slice = slice(config["num_start_idx"], config["num_end_idx"] + 1)

        scaler = StandardScaler()
        scaler.fit(train_x[:, num_slice].astype(float))

        train_x_std[:, num_slice] = scaler.transform(train_x[:, num_slice].astype(float))
        test_x_std[:, num_slice] = scaler.transform(test_x[:, num_slice].astype(float))

        # Reattach labels so selectors receive the original [features | y] layout.
        data_train = np.hstack([train_x_std, train_y.reshape(-1, 1)])
        data_test = np.hstack([test_x_std, test_y.reshape(-1, 1)])

        if data_test.shape[0] == 0 or data_train.shape[0] == 0:
            continue
        fold_full_train_sizes.append(data_train.shape[0])

        per_n_subset_sizes = []
        per_n_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        if METHOD_NAME.lower() == "taqs":
            selector = taqs.TaskAwareQueryAnchoredSelection(
                data_task=data_test,
                data_train=data_train,
                n_neighbors=max(neighbor_counts),
                num_start_idx=config["num_start_idx"],
                num_end_idx=config["num_end_idx"],
                cat_end_idx=_to_optional_int(config["cat_end_idx"]),
                feature_weights=config["feature_weights"],
            )

        elif METHOD_NAME.lower() == "rqas":
            selector = rqas.RandomQueryAnchoredSelection(
                data_task=data_test,
                data_train=data_train,
                n_neighbors=max(neighbor_counts),
                num_start_idx=config["num_start_idx"],
                num_end_idx=config["num_end_idx"],
                cat_end_idx=_to_optional_int(config["cat_end_idx"]),
                random_state=seed,
                size_excel_path="../results/validation_taqs_paradigm.xlsx",
                sheet_name=dataset_name[:30],
            )

        else:
            raise ValueError(f"Unsupported METHOD_NAME: {METHOD_NAME}")
        subset_by_n = selector.select_subsets_for_k_list(neighbor_counts)

        for n_neighbors in neighbor_counts:
            # Reuse precomputed top-kmax neighbors and take the prefix-union for current n.
            subset_train = subset_by_n[n_neighbors]

            per_n_subset_sizes.append(subset_train.shape[0])

            # Each model is fitted on the same selected subset for fair comparison.
            subset_x = subset_train[:, :-1]
            subset_y = subset_train[:, -1]
            models = build_models(config, seed, dataset_name)

            for model_name, model in models.items():
                model.fit(subset_x, subset_y)

                train_pred = model.predict(subset_x)
                train_metrics = utils.evaluate_regression(subset_y, train_pred, verbose=False)

                test_pred = model.predict(test_x_std)
                test_metrics = utils.evaluate_regression(test_y, test_pred, verbose=False)

                # Keep the three core regression metrics for later averaging.
                for metric_name in ["MAE", "RMSE", "R2"]:
                    per_n_results[model_name]["train"][metric_name].append(train_metrics[metric_name])
                    per_n_results[model_name]["test"][metric_name].append(test_metrics[metric_name])

        fold_subset_sizes.append(per_n_subset_sizes)

        # Full-train baseline: train on complete data_train (no TAQS subset filtering).
        baseline_x = data_train[:, :-1]
        baseline_y = data_train[:, -1]
        baseline_test_x = data_test[:, :-1]
        baseline_test_y = data_test[:, -1]
        baseline_models = build_models(config, seed, dataset_name)
        for model_name, model in baseline_models.items():
            model.fit(baseline_x, baseline_y)
            baseline_train_pred = model.predict(baseline_x)
            baseline_test_pred = model.predict(baseline_test_x)
            baseline_train_metrics = utils.evaluate_regression(baseline_y, baseline_train_pred, verbose=False)
            baseline_test_metrics = utils.evaluate_regression(baseline_test_y, baseline_test_pred, verbose=False)
            for metric_name in ["MAE", "RMSE", "R2"]:
                fold_baseline_metrics[model_name]["train"][metric_name].append(baseline_train_metrics[metric_name])
                fold_baseline_metrics[model_name]["test"][metric_name].append(baseline_test_metrics[metric_name])

        for model_name in per_n_results:
            for split_name in ["train", "test"]:
                for metric_name in ["MAE", "RMSE", "R2"]:
                    fold_metrics[model_name][split_name][metric_name].append(
                        per_n_results[model_name][split_name][metric_name]
                    )

    seed_rows = []

    if len(fold_subset_sizes) == 0:
        return seed, seed_rows

    # Average subset size across folds for each n (used for TAQS size analysis).
    avg_subset_size = np.mean(np.array(fold_subset_sizes), axis=0)
    for n_neighbors, avg_size in zip(neighbor_counts, avg_subset_size):
        seed_rows.append(("TAQS", "subset", "SampleSize", n_neighbors, float(avg_size)))
    if len(fold_full_train_sizes) > 0:
        seed_rows.append(
            ("TAQS", "subset", "SampleSize", BASELINE_KEY, float(np.mean(fold_full_train_sizes)))
        )

    for model_name in fold_metrics:
        for split_name in ["train", "test"]:
            for metric_name in ["MAE", "RMSE", "R2"]:
                # Shape: [n_folds, n_neighbor_options] -> mean over folds.
                values = np.mean(np.array(fold_metrics[model_name][split_name][metric_name]), axis=0)
                for n_neighbors, value in zip(neighbor_counts, values):
                    seed_rows.append((model_name, split_name, metric_name, n_neighbors, float(value)))

    for model_name in fold_baseline_metrics:
        for split_name in ["train", "test"]:
            for metric_name in ["MAE", "RMSE", "R2"]:
                baseline_value = float(np.mean(fold_baseline_metrics[model_name][split_name][metric_name]))
                seed_rows.append((model_name, split_name, metric_name, BASELINE_KEY, baseline_value))

    return seed, seed_rows


def run_seed_experiment(dataset_name, seed):
    print(f"[Seed {seed}] Start dataset: {dataset_name}")
    config = dp.load_dataset_config("../configs/datasets_config.yaml", dataset_name)
    processor = dp.DataProcessor()
    _, data = processor.load_data(config["file_path"])

    seed, seed_rows = run_fold_experiment(data, config, seed, dataset_name)
    print(f"[Seed {seed}] Finished dataset: {dataset_name} (rows={len(seed_rows)})")
    return seed, seed_rows


if __name__ == "__main__":
    if METHOD_NAME.lower() == "taqs":
        output_path = "../results/_temporary_validation_taqs_paradigm.xlsx"
        figure_dir = "../results/validation_taqs_paradigm_figures"
    elif METHOD_NAME.lower() == "rqas":
        output_path = "../results/_temporary_validation_rqas_paradigm.xlsx"
        figure_dir = "../results/validation_rqas_paradigm_figures"
    else:
        raise ValueError(f"Unsupported METHOD_NAME: {METHOD_NAME}")
    neighbor_counts = build_neighbor_counts()

    with pd.ExcelWriter(output_path) as writer:
        for dataset_name in dataset_names:
            print(f"\n===== {dataset_name} =====")

            # results[model][split][metric][n] -> list over random seeds
            results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(run_seed_experiment, dataset_name, seed): seed
                    for seed in random_seeds
                }

                for future in as_completed(futures):
                    seed, seed_rows = future.result()
                    print(f"[Seed {seed}] done")
                    for model_name, split_name, metric_name, n_neighbors, value in seed_rows:
                        results[model_name][split_name][metric_name][n_neighbors].append(value)

            # Convert aggregated dict to table with mean/std columns.
            rows = []
            for model_name in results:
                for split_name in results[model_name]:
                    for metric_name in results[model_name][split_name]:
                        row = {
                            "Model": model_name,
                            "Split": split_name,
                            "Metric": metric_name,
                        }
                        for n_neighbors in neighbor_counts:
                            values = results[model_name][split_name][metric_name][n_neighbors]
                            row[f"n={n_neighbors}"] = float(np.mean(values)) if values else np.nan
                            row[f"n={n_neighbors}_std"] = float(np.std(values)) if values else np.nan
                        baseline_values = results[model_name][split_name][metric_name][BASELINE_KEY]
                        row["baseline_full_train"] = float(np.mean(baseline_values)) if baseline_values else np.nan
                        row["baseline_full_train_std"] = float(np.std(baseline_values)) if baseline_values else np.nan
                        rows.append(row)

            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=dataset_name[:30], index=False)
            print(
                f"[{dataset_name}] Aggregated rows: {len(rows)} | "
                f"saved sheet: {dataset_name[:30]}"
            )

            # Save R2 train/test curves for each model under this dataset.
            for model_name in ["RR", "RF", "MLP"]:
                if model_name not in results:
                    continue

                train_values = [
                    np.mean(results[model_name]["train"]["R2"][n_neighbors])
                    for n_neighbors in neighbor_counts
                ]
                test_values = [
                    np.mean(results[model_name]["test"]["R2"][n_neighbors])
                    for n_neighbors in neighbor_counts
                ]
                save_path = os.path.join(
                    figure_dir,
                    f"{dataset_name}_{model_name}_R2.png".replace(" ", "_"),
                )
                utils.plot_train_test_curve(
                    x_values=neighbor_counts,
                    train_values=train_values,
                    test_values=test_values,
                    xlabel="Number of nearest neighbors",
                    ylabel=r"Performance (R$^2$)",
                    train_label="train_set",
                    test_label="task_set",
                    save_path=save_path,
                    show=False,
                )

    print(f"\n[Saved] -> {output_path}")
    print("=== Done ===")

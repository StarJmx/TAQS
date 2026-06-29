"""
tune_mlp_params.py

Grid-search MLP hyperparameters across datasets and export a generalizable
best configuration to results/mlp_tuning.xlsx.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

# Parallel controls:
# - number of worker processes
# - number of BLAS/OpenMP threads per process
THREADS_PER_PROCESS = int(os.environ.get("TAQS_TUNE_THREADS_PER_PROCESS", "2"))
os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import data_processor as dp
from src import utils


dataset_names = [
    "LAQ",
    "SCD",
    "EVEE",
    "MN1B",
    "ATRIV",
    "DRGD"
]

SEED = 42
N_SPLITS = 5
MAX_WORKERS = int(
    os.environ.get(
        "TAQS_TUNE_MAX_WORKERS",
        str(max(1, min(len(dataset_names) * 2, 7))),
    )
)


def _to_optional_int(value):
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("none", "null", ""):
            return None
        return int(v)
    return int(value)


def build_param_candidates():
    """Return the Cartesian product of a compact MLP hyperparameter grid."""
    hidden_sizes = [(64, 32), (128, 64), (128, 64, 32)]
    alpha_values = [0.001, 0.01, 0.1]
    learning_rates = [0.0001, 0.001, 0.01]
    max_iters = [200, 600]

    candidates = []
    for hs, alpha, lr, max_iter in product(hidden_sizes, alpha_values, learning_rates, max_iters):
        candidates.append(
            {
                "hidden_layer_sizes": hs,
                "alpha": alpha,
                "learning_rate_init": lr,
                "max_iter": max_iter,
            }
        )
    return candidates


def build_mlp_pipeline(config, params, seed):
    num_cols = list(range(config["num_start_idx"], config["num_end_idx"] + 1))
    cat_start = config["num_end_idx"] + 1
    cat_end = _to_optional_int(config["cat_end_idx"])
    cat_cols = list(range(cat_start, cat_end + 1)) if cat_end is not None and cat_end >= cat_start else []

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    model = MLPRegressor(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation="relu",
        solver="adam",
        alpha=params["alpha"],
        batch_size="auto",
        learning_rate_init=params["learning_rate_init"],
        max_iter=params["max_iter"],
        random_state=seed,
    )
    return make_pipeline(preprocessor, model)


def evaluate_candidate_on_dataset(dataset_name, params, seed=42, n_splits=5):
    """Run shuffled K-fold CV for one hyperparameter candidate on one dataset."""
    config = dp.load_dataset_config("../configs/datasets_config.yaml", dataset_name)
    processor = dp.DataProcessor()
    _, data = processor.load_data(config["file_path"], random_state=seed)

    X = data[:, :-1]
    y = data[:, -1]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []

    for train_idx, test_idx in kf.split(X):
        model = build_mlp_pipeline(config, params, seed=seed)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        fold_metrics.append(utils.evaluate_regression(y[test_idx], pred, verbose=False))

    rmse = float(np.mean([m["RMSE"] for m in fold_metrics]))
    mae = float(np.mean([m["MAE"] for m in fold_metrics]))
    r2 = float(np.mean([m["R2"] for m in fold_metrics]))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def evaluate_task(task):
    candidate_id, params, dataset_name = task
    metrics = evaluate_candidate_on_dataset(dataset_name, params, seed=SEED, n_splits=N_SPLITS)
    return {
        "candidate_id": candidate_id,
        "dataset": dataset_name,
        "hidden_layer_sizes": str(params["hidden_layer_sizes"]),
        "alpha": params["alpha"],
        "learning_rate_init": params["learning_rate_init"],
        "max_iter": params["max_iter"],
        "rmse": metrics["RMSE"],
        "mae": metrics["MAE"],
        "r2": metrics["R2"],
    }


def pick_generalizable_params(all_scores_df):
    """
    Pick the candidate with the best average within-dataset RMSE rank.

    Lower mean rank is better; ties break on lower mean RMSE.
    """
    rank_df = all_scores_df.copy()
    rank_df["rmse_rank"] = rank_df.groupby("dataset")["rmse"].rank(method="average", ascending=True)

    summary = (
        rank_df.groupby("candidate_id", as_index=False)
        .agg(
            mean_rmse_rank=("rmse_rank", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_r2=("r2", "mean"),
        )
        .sort_values(["mean_rmse_rank", "mean_rmse"], ascending=[True, True])
    )
    best_candidate_id = int(summary.iloc[0]["candidate_id"])
    return best_candidate_id, summary


if __name__ == "__main__":
    candidates = build_param_candidates()
    rows = []
    tasks = []
    for candidate_id, params in enumerate(candidates):
        print(f"[Candidate {candidate_id}] {params}")
        for dataset_name in dataset_names:
            tasks.append((candidate_id, params, dataset_name))

    print(
        f"\n[Parallel] workers={MAX_WORKERS}, threads_per_process={THREADS_PER_PROCESS}, "
        f"tasks={len(tasks)}"
    )

    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(evaluate_task, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            if done % 8 == 0 or done == len(tasks):
                print(f"[Progress] {done}/{len(tasks)}")

    all_scores_df = pd.DataFrame(rows)
    best_id, summary_df = pick_generalizable_params(all_scores_df)

    best_row = summary_df[summary_df["candidate_id"] == best_id].iloc[0]
    best_params = candidates[best_id]

    best_params_df = pd.DataFrame(
        [
            {
                "candidate_id": best_id,
                "hidden_layer_sizes": str(best_params["hidden_layer_sizes"]),
                "alpha": best_params["alpha"],
                "learning_rate_init": best_params["learning_rate_init"],
                "max_iter": best_params["max_iter"],
                "mean_rmse_rank": best_row["mean_rmse_rank"],
                "mean_rmse": best_row["mean_rmse"],
                "mean_mae": best_row["mean_mae"],
                "mean_r2": best_row["mean_r2"],
            }
        ]
    )

    output_path = "../results/mlp_tuning.xlsx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        all_scores_df.to_excel(writer, sheet_name="all_scores", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        best_params_df.to_excel(writer, sheet_name="best_params", index=False)

    print("\n=== Generalizable MLP Params (6 datasets) ===")
    print(best_params)
    print(f"[Saved] -> {output_path}")

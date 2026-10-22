"""
Disjoint Task Evaluation Experiment for TAQS

Protocol
--------
D -> 20 clusters by classical Gower on X only
N random samples (N = 5% of |D|) from the largest cluster whose size is at
most 20% of |D| -> Target Pool
that cluster's leftover + all other clusters -> Candidate Pool

Target Pool -> 50% Task + remaining Test

Task and Test are strictly disjoint:
    Task ∩ Test = ∅

The Target Pool is drawn once per dataset from a capped source cluster (the
largest cluster no bigger than 20% of |D|) using a fixed clustering seed and
a fixed target-sampling seed.
Task extraction uses a fixed random seed, so the final Task/Test split is
identical across all random seeds.

For each dataset x downstream model, TAQS k is selected only from the
Candidate Pool by 50-fold CV. The Candidate Pool keeps its original row
order for fold assignment, so every seed shares the same fold partition and
the seed only drives model training; the seed-level pooled OOF R2 values are
averaged per k. The
top-3 k candidates by mean OOF R2 are retained, and the final k is the
candidate with the strongest local neighborhood support.
This yields one global k for each dataset x model (18 global k values total).

Final evaluation:
    Candidate Pool -> {TAQS, Random, K-Medoids} with matched subset size
    -> RR / RF / MLP
    -> untouched Test subset of the Target Pool

Target-domain clustering uses classical Gower on raw mixed-type X only.
Task extraction draws a uniformly random subset of the Target Pool and uses
no distance metric and no target labels.
Downstream standardization follows the original design: fold-specific
standardization during CV, and Candidate-fitted standardization for the final
Candidate/Task/Test evaluation.

This script intentionally reuses the original:
- datasets
- random seeds
- dataset config / data paths
- feature weights
- MLP tuned parameters
- quantitative standardization
- multi-process-by-seed execution
- TAQS neighbor candidate list
"""

import os
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------
# Preserve original worker-thread control
# ---------------------------------------------------------------------
THREADS_PER_PROCESS = int(os.environ.get("TAQS_THREADS_PER_PROCESS", "4"))
os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS_PER_PROCESS)
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

from sklearn_extra.cluster import KMedoids

from src import data_processor as dp
from src import taqs_algorithm as taqs
from src import utils

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    module="sklearn.neural_network",
)
warnings.filterwarnings(
    "ignore",
    message=r"Cluster\s+\d+\s+is empty",
    category=UserWarning,
)

# ---------------------------------------------------------------------
# Original settings retained
# ---------------------------------------------------------------------
dataset_names = [
    "LAQ",
    "SCD",
    "EVEE",
    "MN1B",
    "ATRIV",
    "DRGD",
]

random_seeds = [42, 123, 999]

K_NEIGHBORS = [
    1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50,
    75, 100, 125, 150, 175, 200
]

# Stability control for the global-k selection: keep the top-N candidates
# by mean OOF R2, then prefer the k with stronger local neighborhood
# support (mean OOF R2 of the k itself and its adjacent candidates).
TOP_K_CANDIDATES = 3
LOCAL_NEIGHBOR_RADIUS = 1

N_CV_FOLDS = 50
N_TARGET_CLUSTERS = 20

# Target Pool selection: among the clusters whose size is at most
# TARGET_POOL_MAX_CLUSTER_RATIO * |D|, take the largest one as the source
# cluster; draw TARGET_POOL_RATIO * |D| samples at random (fixed seed
# TARGET_SAMPLE_SEED) from it as the Target Pool. The leftover of that
# cluster and all other clusters form the Candidate Pool.
TARGET_POOL_RATIO = 0.05
TARGET_POOL_MAX_CLUSTER_RATIO = 0.20
TARGET_SAMPLE_SEED = 42

TASK_RATIO_OF_TARGET_POOL = 0.50

# Fixed seed for the unsupervised dataset partition into clusters.
CLUSTER_SEED = 42

# Fixed seed for Task extraction so the final Task/Test split is identical
# across all random seeds.
TASK_EXTRACT_SEED = 42

MAX_WORKERS = min(len(random_seeds), max(1, os.cpu_count() - 1))

OUTPUT_PATH = "../results/_temporary_disjoint_task_evaluation.xlsx"


# ---------------------------------------------------------------------
# MLP parameter cache copied from original experiment
# ---------------------------------------------------------------------
_MLP_PARAMS_CACHE = {}
_MLP_PARAMS_LOGGED = set()


def _to_optional_int(value):
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("none", "null", ""):
            return None
        return int(v)
    return int(value)


def load_mlp_params_for_dataset(dataset_name):
    global _MLP_PARAMS_CACHE, _MLP_PARAMS_LOGGED

    if not _MLP_PARAMS_CACHE:
        params_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "results",
            "mlp_tuning.xlsx",
        )
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
    elif hasattr(hidden_layers, "tolist"):
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


# ---------------------------------------------------------------------
# Models: retained from original Table 4 code
# ---------------------------------------------------------------------
def build_models(config, seed, dataset_name=None):
    num_cols = list(range(config["num_start_idx"], config["num_end_idx"] + 1))
    cat_start = config["num_end_idx"] + 1
    cat_end = _to_optional_int(config["cat_end_idx"])
    cat_cols = (
        list(range(cat_start, cat_end + 1))
        if cat_end is not None and cat_end >= cat_start
        else []
    )

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
        "RR": make_pipeline(
            rr_preprocessor,
            Ridge(),
        ),
        "RF": RandomForestRegressor(
            random_state=seed,
            n_jobs=THREADS_PER_PROCESS,
        ),
        "MLP": make_pipeline(
            mlp_preprocessor,
            MLPRegressor(
                hidden_layer_sizes=mlp_params.get(
                    "hidden_layer_sizes", [128, 64, 32]
                ),
                alpha=mlp_params.get("alpha", 0.001),
                learning_rate_init=mlp_params.get("learning_rate_init", 0.01),
                max_iter=mlp_params.get("max_iter", 200),
                random_state=seed,
            ),
        ),
    }


# ---------------------------------------------------------------------
# Standardization: same rule as original experiments
# Fit only on the current training portion.
# ---------------------------------------------------------------------
def standardize(train_x, other_x_list, num_slice):
    scaler = StandardScaler()
    scaler.fit(train_x[:, num_slice].astype(float))

    train_std = train_x.copy()
    train_std[:, num_slice] = scaler.transform(
        train_x[:, num_slice].astype(float)
    )

    transformed = []
    for x in other_x_list:
        x_std = x.copy()
        x_std[:, num_slice] = scaler.transform(
            x[:, num_slice].astype(float)
        )
        transformed.append(x_std)

    return train_std, transformed


# ---------------------------------------------------------------------
# Classical Gower distance helper
#
# Used only for:
# 1) unsupervised 20-way target-domain construction on the full dataset;
# 2) task-independent K-Medoids training-subset baseline.
#
# Categorical variables remain in their original single-column form.
# ---------------------------------------------------------------------
def compute_gower_distance_matrix(X, num_idx, cat_idx, feature_ranges=None):
    """
    Classical mixed-type Gower distance.

    Numerical features are normalized by supplied feature ranges when
    provided; otherwise, ranges are computed from X itself.
    Categorical features contribute 0 for equality and 1 for mismatch.
    No TAQS/TAWG feature weights are used.

    The target column is excluded before this function is called.
    """
    n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    feature_count = 0

    if num_idx:
        X_num = X[:, num_idx].astype(np.float64)
        for j, idx in enumerate(num_idx):
            col = X_num[:, j]

            if feature_ranges is None:
                r = float(np.nanmax(col) - np.nanmin(col))
            else:
                r = float(feature_ranges[idx])

            if not np.isfinite(r) or r == 0:
                r = 1.0

            D += np.abs(col[:, None] - col[None, :]) / r
            feature_count += 1

    if cat_idx:
        X_cat = X[:, cat_idx]
        for j in range(X_cat.shape[1]):
            col = X_cat[:, j]
            D += (col[:, None] != col[None, :]).astype(np.float64)
            feature_count += 1

    return D / feature_count if feature_count else D


def get_feature_indices(config):
    num_idx = list(range(config["num_start_idx"], config["num_end_idx"] + 1))
    cat_start = config["num_end_idx"] + 1
    cat_end = _to_optional_int(config["cat_end_idx"])
    cat_idx = (
        list(range(cat_start, cat_end + 1))
        if cat_end is not None and cat_end >= cat_start
        else []
    )
    return num_idx, cat_idx


# ---------------------------------------------------------------------
# Controlled target-domain construction
# ---------------------------------------------------------------------
def build_clustered_target_domain(data, config):
    """
    Partition the full dataset into N_TARGET_CLUSTERS clusters using
    classical Gower on X only.

    Among the clusters whose size is at most TARGET_POOL_MAX_CLUSTER_RATIO *
    |D|, the largest one is taken as the source cluster. TARGET_POOL_RATIO *
    |D| samples are then drawn at random (fixed seed) from it to form the
    Target Pool. The leftover of the source cluster and all other clusters
    are merged into the Candidate Pool.

    The clustering itself is unsupervised: y is never used.
    """
    X = data[:, :-1]
    num_idx, cat_idx = get_feature_indices(config)

    print(
        f"[Cluster] N={X.shape[0]}, "
        f"clusters={N_TARGET_CLUSTERS}, "
        f"numeric={len(num_idx)}, categorical={len(cat_idx)}"
    )

    # Full-dataset Gower uses the full dataset's numerical ranges.
    distance_matrix = compute_gower_distance_matrix(
        X,
        num_idx,
        cat_idx,
        feature_ranges=None,
    )

    model = KMedoids(
        n_clusters=N_TARGET_CLUSTERS,
        metric="precomputed",
        random_state=CLUSTER_SEED,
    )
    model.fit(distance_matrix)

    labels = np.asarray(model.labels_, dtype=int)
    cluster_sizes = np.bincount(
        labels,
        minlength=N_TARGET_CLUSTERS,
    ).astype(int)

    n_total = data.shape[0]

    # Source cluster = the largest cluster whose size does not exceed
    # TARGET_POOL_MAX_CLUSTER_RATIO * |D|. Among eligible clusters the
    # largest wins; deterministic tie-break: smaller cluster id.
    max_source_size = TARGET_POOL_MAX_CLUSTER_RATIO * n_total
    eligible = np.flatnonzero(cluster_sizes <= max_source_size)

    if eligible.size == 0:
        # Defensive fallback (cannot happen with 20 clusters): keep the
        # overall largest cluster.
        target_cluster_id = int(np.argmax(cluster_sizes))
    else:
        best = np.lexsort((eligible, -cluster_sizes[eligible]))[0]
        target_cluster_id = int(eligible[best])

    source_indices = np.flatnonzero(labels == target_cluster_id).astype(int)

    # Draw TARGET_POOL_RATIO * |D| samples from that cluster with a fixed
    # seed; the leftover of the cluster stays in the Candidate Pool.
    n_target = int(round(TARGET_POOL_RATIO * n_total))
    n_target = max(1, min(n_target, source_indices.shape[0]))

    rng = np.random.default_rng(TARGET_SAMPLE_SEED)
    sampled = rng.choice(
        source_indices.shape[0],
        size=n_target,
        replace=False,
    )
    target_indices = np.sort(source_indices[sampled]).astype(int)

    print(
        f"[Cluster] source_cluster={target_cluster_id}, "
        f"source_size={source_indices.shape[0]}, "
        f"target_size={len(target_indices)}, "
        f"target_ratio={len(target_indices) / n_total:.4f}, "
        f"ideal_ratio={TARGET_POOL_RATIO:.2f}, "
        f"source_cap={TARGET_POOL_MAX_CLUSTER_RATIO:.2f}"
    )
    print(
        "[Cluster] sizes="
        + ", ".join(str(int(x)) for x in cluster_sizes.tolist())
    )

    return target_indices, target_cluster_id, cluster_sizes


# ---------------------------------------------------------------------
# Task extraction from the target pool
# ---------------------------------------------------------------------
def random_extract_task(data_target_raw, n_task, seed):
    """
    Extract the Task set from the Target Pool by uniform random sampling.

    A fixed seed keeps the final Task/Test split identical across all
    random seeds. No distance metric and no target labels are used.
    """
    n_target = data_target_raw.shape[0]
    n_task = max(1, min(int(n_task), n_target))

    if n_task >= n_target:
        return np.arange(n_target, dtype=int)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_target, size=n_task, replace=False))


def kmedoids_select_training_indices(
    data_candidate_raw,
    subset_size,
    config,
    seed,
):
    """Task-independent K-Medoids baseline on the Candidate Pool using classical Gower."""
    n_candidate = data_candidate_raw.shape[0]
    subset_size = max(1, min(int(subset_size), n_candidate))

    if subset_size >= n_candidate:
        return np.arange(n_candidate, dtype=int)

    X_candidate = data_candidate_raw[:, :-1]
    num_idx, cat_idx = get_feature_indices(config)

    # Candidate-side Gower defines its own numerical ranges.
    distance_matrix = compute_gower_distance_matrix(
        X_candidate,
        num_idx,
        cat_idx,
        feature_ranges=None,
    )

    model = KMedoids(
        n_clusters=subset_size,
        metric="precomputed",
        random_state=seed,
    )
    model.fit(distance_matrix)

    return np.asarray(model.medoid_indices_, dtype=int)


def random_select_training_indices(data_train, subset_size, seed):
    n_train = data_train.shape[0]
    subset_size = max(1, min(int(subset_size), n_train))

    if subset_size >= n_train:
        return np.arange(n_train, dtype=int)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_train, size=subset_size, replace=False))


# ---------------------------------------------------------------------
# CV for TAQS-k on the Candidate Pool
# ---------------------------------------------------------------------
def select_best_k_by_cv(
    data_candidate_raw,
    config,
    seed,
    dataset_name,
):
    """
    Evaluate all candidate k values separately for each downstream learner.

    CV is performed only within the Candidate Pool. The Candidate Pool keeps
    its original row order for fold assignment, so every seed shares the same
    fold partition; the seed only drives stochastic model training.

    In each fold, the fold's validation set serves as the query/task
    representation and also as the evaluation set. The Target Pool is
    never used in k selection.

    OOF predictions are pooled within each seed to obtain one overall
    OOF R2 for every (model, k).
    """
    # Keep the original row order for fold assignment; the seed only
    # drives stochastic model training.
    data_cv_ordered = data_candidate_raw

    fold_indices = np.array_split(
        np.arange(data_cv_ordered.shape[0]),
        N_CV_FOLDS,
    )

    cv_results = []

    # Collect out-of-fold predictions for pooled OOF R2 within this seed.
    oof_y_true = {
        model_name: {k: [] for k in K_NEIGHBORS}
        for model_name in ["RR", "RF", "MLP"]
    }
    oof_y_pred = {
        model_name: {k: [] for k in K_NEIGHBORS}
        for model_name in ["RR", "RF", "MLP"]
    }

    for fold_idx, val_idx in enumerate(fold_indices, start=1):
        train_mask = np.ones(data_cv_ordered.shape[0], dtype=bool)
        train_mask[val_idx] = False

        train_raw = data_cv_ordered[train_mask]
        val_raw = data_cv_ordered[val_idx]

        train_x = train_raw[:, :-1]
        train_y = train_raw[:, -1]

        val_x = val_raw[:, :-1]
        val_y = val_raw[:, -1]

        num_slice = slice(
            config["num_start_idx"],
            config["num_end_idx"] + 1,
        )

        # Fold-specific standardization: fit on fold-train only.
        train_x_std, transformed = standardize(
            train_x,
            [val_x],
            num_slice,
        )
        val_x_std = transformed[0]

        train_std = np.hstack(
            [train_x_std, train_y.reshape(-1, 1)]
        )
        val_std = np.hstack(
            [val_x_std, val_y.reshape(-1, 1)]
        )

        # The current fold's validation set is the TAQS query.
        selector = taqs.TaskAwareQueryAnchoredSelection(
            data_task=val_std,
            data_train=train_std,
            n_neighbors=max(K_NEIGHBORS),
            num_start_idx=config["num_start_idx"],
            num_end_idx=config["num_end_idx"],
            cat_end_idx=_to_optional_int(config["cat_end_idx"]),
            feature_weights=config["feature_weights"],
        )

        subsets = selector.select_subsets_for_k_list(K_NEIGHBORS)

        for k in K_NEIGHBORS:
            subset = subsets[k]
            subset_x = subset[:, :-1]
            subset_y = subset[:, -1]

            models = build_models(
                config,
                seed,
                dataset_name,
            )

            for model_name, model in models.items():
                model.fit(subset_x, subset_y)
                pred = model.predict(val_x_std)

                metrics = utils.evaluate_regression(
                    val_y,
                    pred,
                    verbose=False,
                )

                cv_results.append(
                    {
                        "Fold": fold_idx,
                        "K": k,
                        "Model": model_name,
                        "TrainSize": int(subset.shape[0]),
                        "MAE": float(metrics["MAE"]),
                        "RMSE": float(metrics["RMSE"]),
                        "R2": float(metrics["R2"]),
                    }
                )

                oof_y_true[model_name][k].append(np.asarray(val_y))
                oof_y_pred[model_name][k].append(np.asarray(pred))

    cv_df = pd.DataFrame(cv_results)

    # One pooled OOF R2 per (model, k) for this seed.
    oof_r2_by_model = {}
    for model_name in ["RR", "RF", "MLP"]:
        oof_r2_by_model[model_name] = {}

        for k in K_NEIGHBORS:
            y_true = np.concatenate(oof_y_true[model_name][k])
            y_pred = np.concatenate(oof_y_pred[model_name][k])

            metrics = utils.evaluate_regression(
                y_true,
                y_pred,
                verbose=False,
            )
            oof_r2_by_model[model_name][k] = float(metrics["R2"])

    return cv_df, oof_r2_by_model


def select_global_k_across_seeds(oof_r2_by_seed):
    """
    Select one global k for each downstream learner by averaging the
    seed-level pooled OOF R2 values across the three seeds, then
    applying the retained Top-3 + local-neighborhood stability rule.

    The result is one global k per dataset x downstream model.
    """
    global_k_by_model = {}
    global_k_details = []

    k_order = {k: i for i, k in enumerate(K_NEIGHBORS)}

    for model_name in ["RR", "RF", "MLP"]:
        rows = []

        for seed, oof_r2_by_model in oof_r2_by_seed.items():
            for k, r2 in oof_r2_by_model[model_name].items():
                rows.append(
                    {
                        "Seed": seed,
                        "K": int(k),
                        "OOF_R2": float(r2),
                    }
                )

        detail_df = pd.DataFrame(rows)

        mean_by_k = (
            detail_df
            .groupby("K", as_index=False)["OOF_R2"]
            .mean()
            .rename(columns={"OOF_R2": "Mean_OOF_R2"})
        )

        mean_by_k = mean_by_k.sort_values(
            ["Mean_OOF_R2", "K"],
            ascending=[False, True],
        )

        r2_map = dict(
            zip(mean_by_k["K"], mean_by_k["Mean_OOF_R2"])
        )

        # Local support = mean OOF R2 over the candidate's neighboring
        # k values in the predefined K_NEIGHBORS list, itself included.
        local_support_map = {}

        for k in K_NEIGHBORS:
            idx = k_order[k]
            lo = max(0, idx - LOCAL_NEIGHBOR_RADIUS)
            hi = min(
                len(K_NEIGHBORS),
                idx + LOCAL_NEIGHBOR_RADIUS + 1,
            )
            neighborhood = K_NEIGHBORS[lo:hi]

            local_support_map[k] = float(
                np.mean([r2_map[x] for x in neighborhood])
            )

        # Step 1: retain Top-3 performers by mean OOF R2.
        top_candidates = mean_by_k.head(
            TOP_K_CANDIDATES
        )["K"].tolist()

        # Step 2: among them, prefer stronger local support.
        stable_candidates = []

        for k in top_candidates:
            stable_candidates.append(
                (
                    k,
                    local_support_map[k],
                    r2_map[k],
                )
            )

        # Higher local support first; then higher own OOF R2;
        # then smaller k.
        stable_candidates.sort(
            key=lambda x: (-x[1], -x[2], x[0])
        )

        best_k = int(stable_candidates[0][0])
        global_k_by_model[model_name] = best_k

        for _, row in mean_by_k.iterrows():
            k_val = int(row["K"])
            global_k_details.append(
                {
                    "Model": model_name,
                    "K": k_val,
                    "Mean_OOF_R2": float(row["Mean_OOF_R2"]),
                    "LocalSupport": local_support_map[k_val],
                    "Selected": k_val == best_k,
                }
            )

    return global_k_by_model, pd.DataFrame(global_k_details)


# ---------------------------------------------------------------------
# CV worker for one seed
# ---------------------------------------------------------------------
def run_seed_cv_experiment(
    dataset_name,
    seed,
    target_indices,
):
    print(f"\n[Seed {seed}] CV START {dataset_name}")

    config = dp.load_dataset_config(
        "../configs/datasets_config.yaml",
        dataset_name,
    )

    processor = dp.DataProcessor()
    _, data = processor.load_data(config["file_path"])

    target_mask = np.zeros(data.shape[0], dtype=bool)
    target_mask[target_indices] = True
    data_candidate_raw = data[~target_mask]

    cv_df, oof_r2_by_model = select_best_k_by_cv(
        data_candidate_raw=data_candidate_raw,
        config=config,
        seed=seed,
        dataset_name=dataset_name,
    )

    print(
        f"[Seed {seed}] CV FINISH {dataset_name} | "
        f"D={len(data)}, Candidate={len(data_candidate_raw)}, "
        f"Target={len(target_indices)}"
    )

    return seed, cv_df, oof_r2_by_model


# ---------------------------------------------------------------------
# Final evaluation worker using globally selected k
# ---------------------------------------------------------------------
def run_final_evaluation(
    dataset_name,
    seed,
    target_indices,
    target_cluster_id,
    cluster_sizes,
    global_k_by_model,
    section_label="FinalTest",
):
    print(f"\n[Seed {seed}] {section_label} START {dataset_name}")

    config = dp.load_dataset_config(
        "../configs/datasets_config.yaml",
        dataset_name,
    )

    processor = dp.DataProcessor()
    _, data = processor.load_data(config["file_path"])

    target_mask = np.zeros(data.shape[0], dtype=bool)
    target_mask[target_indices] = True

    data_candidate_raw = data[~target_mask]
    data_target_raw = data[target_mask]

    # ---------------------------------------------------------------
    # Step A: extract Task from the Target Pool by uniform random
    # sampling with a fixed seed (no distance metric, no labels).
    # ---------------------------------------------------------------
    n_task_target = max(
        1,
        int(round(
            data_target_raw.shape[0]
            * TASK_RATIO_OF_TARGET_POOL
        )),
    )
    n_task_target = min(
        n_task_target,
        data_target_raw.shape[0] - 1,
    )

    task_indices_local = random_extract_task(
        data_target_raw,
        n_task_target,
        TASK_EXTRACT_SEED,
    )

    # ---------------------------------------------------------------
    # Step B: Task is the random subset; Test is its complement.
    # Standardization is fit on the Candidate Pool only, then applied
    # unchanged to Candidate, Task, and Test.
    # ---------------------------------------------------------------
    target_all_indices = np.arange(
        data_target_raw.shape[0],
        dtype=int,
    )

    is_task = np.zeros(
        data_target_raw.shape[0],
        dtype=bool,
    )
    is_task[task_indices_local] = True

    test_indices_local = target_all_indices[~is_task]

    num_slice = slice(
        config["num_start_idx"],
        config["num_end_idx"] + 1,
    )

    xcandidate = data_candidate_raw[:, :-1]
    ycandidate = data_candidate_raw[:, -1]

    xtask = data_target_raw[task_indices_local, :-1]
    ytask = data_target_raw[task_indices_local, -1]

    xtest = data_target_raw[test_indices_local, :-1]
    ytest = data_target_raw[test_indices_local, -1]

    # Candidate-fitted standardization avoids using target/test statistics.
    xcandidate_std, transformed = standardize(
        xcandidate,
        [xtask, xtest],
        num_slice,
    )
    xtask_std, xtest_std = transformed

    data_candidate_std = np.hstack(
        [xcandidate_std, ycandidate.reshape(-1, 1)]
    )
    data_task_std = np.hstack(
        [xtask_std, ytask.reshape(-1, 1)]
    )
    data_test_std = np.hstack(
        [xtest_std, ytest.reshape(-1, 1)]
    )

    assert len(
        set(task_indices_local.tolist()).intersection(
            set(test_indices_local.tolist())
        )
    ) == 0, "Task/Test overlap detected."

    # ---------------------------------------------------------------
    # Step C/D/E: Final evaluation using one global k per learner,
    # shared across all random seeds.
    # ---------------------------------------------------------------
    seed_rows = []

    seed_rows.append(
        {
            "Dataset": dataset_name,
            "Seed": seed,
            "Section": "Split",
            "Model": "",
            "Method": "",
            "K": "",
            "D_Total": int(data.shape[0]),
            "CandidateSize": int(data_candidate_std.shape[0]),
            "TargetPoolSize": int(data_target_raw.shape[0]),
            "TargetClusterID": int(target_cluster_id),
            "NTargetClusters": int(N_TARGET_CLUSTERS),
            "TaskSize": int(data_task_std.shape[0]),
            "TestSize": int(data_test_std.shape[0]),
            "CandidateRatioOfD": float(
                data_candidate_std.shape[0] / data.shape[0]
            ),
            "TargetPoolRatioOfD": float(
                data_target_raw.shape[0] / data.shape[0]
            ),
            "TaskRatioOfTargetPool": float(
                data_task_std.shape[0] / data_target_raw.shape[0]
            ),
            "TaskRatioOfTotal": float(
                data_task_std.shape[0] / data.shape[0]
            ),
            "TestRatioOfTotal": float(
                data_test_std.shape[0] / data.shape[0]
            ),
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
        }
    )

    for model_name in ["RR", "RF", "MLP"]:
        best_k = global_k_by_model[model_name]

        selector = taqs.TaskAwareQueryAnchoredSelection(
            data_task=data_task_std,
            data_train=data_candidate_std,
            n_neighbors=best_k,
            num_start_idx=config["num_start_idx"],
            num_end_idx=config["num_end_idx"],
            cat_end_idx=_to_optional_int(config["cat_end_idx"]),
            feature_weights=config["feature_weights"],
        )
        taqs_subset = selector.select_subset()

        target_subset_size = taqs_subset.shape[0]

        random_indices = random_select_training_indices(
            data_candidate_std,
            target_subset_size,
            seed,
        )
        random_subset = data_candidate_std[random_indices]

        kmedoids_indices = kmedoids_select_training_indices(
            data_candidate_raw,
            target_subset_size,
            config,
            seed,
        )
        kmedoids_subset = data_candidate_std[kmedoids_indices]

        full_subset = data_candidate_std

        methods = {
            "TAQS": taqs_subset,
            "Random": random_subset,
            "K-Medoids": kmedoids_subset,
            "Full": full_subset,
        }

        test_x = data_test_std[:, :-1]
        test_y = data_test_std[:, -1]

        for method_name, train_data in methods.items():
            model_seed = seed

            model = build_models(
                config,
                model_seed,
                dataset_name,
            )[model_name]

            train_x = train_data[:, :-1]
            train_y = train_data[:, -1]

            model.fit(train_x, train_y)
            pred = model.predict(test_x)

            metrics = utils.evaluate_regression(
                test_y,
                pred,
                verbose=False,
            )

            seed_rows.append(
                {
                    "Dataset": dataset_name,
                    "Seed": seed,
                    "Section": section_label,
                    "Model": model_name,
                    "Method": method_name,
                    "K": best_k if method_name != "Full" else np.nan,
                    "D_Total": int(data.shape[0]),
                    "CandidateSize": int(data_candidate_std.shape[0]),
                    "TargetPoolSize": int(data_target_raw.shape[0]),
                    "TargetClusterID": int(target_cluster_id),
                    "NTargetClusters": int(N_TARGET_CLUSTERS),
                    "TaskSize": int(data_task_std.shape[0]),
                    "TestSize": int(data_test_std.shape[0]),
                    "CandidateRatioOfD": float(
                        data_candidate_std.shape[0] / data.shape[0]
                    ),
                    "TargetPoolRatioOfD": float(
                        data_target_raw.shape[0] / data.shape[0]
                    ),
                    "TaskRatioOfTargetPool": float(
                        data_task_std.shape[0] / data_target_raw.shape[0]
                    ),
                    "TaskRatioOfTotal": float(
                        data_task_std.shape[0] / data.shape[0]
                    ),
                    "TestRatioOfTotal": float(
                        data_test_std.shape[0] / data.shape[0]
                    ),
                    "TrainSubsetSize": int(train_data.shape[0]),
                    "TrainSubsetRatioOfCandidate": float(
                        train_data.shape[0] / data_candidate_std.shape[0]
                    ),
                    "R2": float(metrics["R2"]),
                    "RMSE": float(metrics["RMSE"]),
                    "MAE": float(metrics["MAE"]),
                }
            )

    print(
        f"[Seed {seed}] {section_label} FINISH {dataset_name} | "
        f"D={len(data)}, Candidate={len(data_candidate_std)}, "
        f"Target={len(data_target_raw)}, "
        f"Task={len(data_task_std)}, Test={len(data_test_std)} | "
        f"TargetCluster={target_cluster_id} | "
        f"Global k={global_k_by_model}"
    )

    return seed, seed_rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # All per-dataset content is accumulated in main process;
    # worker processes only return serializable Python objects.
    final_rows = []
    cv_rows = []
    best_k_rows = []
    global_k_rows = []
    cluster_rows = []

    # Per-dataset agg tables kept for the cross-dataset R2 summary sheet.
    agg_by_dataset = {}

    with pd.ExcelWriter(output_path) as writer:
        for dataset_name in dataset_names:
            print("\n" + "=" * 80)
            print(f"DATASET: {dataset_name}")
            print("=" * 80)

            # ---------------------------------------------------------
            # Phase 0: construct one controlled Target Pool per dataset.
            # Clustering uses X only and is performed once per dataset.
            # ---------------------------------------------------------
            config = dp.load_dataset_config(
                "../configs/datasets_config.yaml",
                dataset_name,
            )
            processor = dp.DataProcessor()
            _, data = processor.load_data(config["file_path"])

            target_indices, target_cluster_id, cluster_sizes = (
                build_clustered_target_domain(data, config)
            )

            for cluster_id, cluster_size in enumerate(cluster_sizes):
                cluster_rows.append(
                    {
                        "Dataset": dataset_name,
                        "ClusterID": int(cluster_id),
                        "ClusterSize": int(cluster_size),
                        "ClusterRatioOfD": float(
                            cluster_size / data.shape[0]
                        ),
                        "IsTarget": bool(
                            cluster_id == target_cluster_id
                        ),
                    }
                )

            # ---------------------------------------------------------
            # Phase 1: run CV for all three seeds and collect seed-level
            # pooled OOF R2 for every (model, k) on Candidate Pool.
            # ---------------------------------------------------------
            oof_r2_by_seed = {}

            with ProcessPoolExecutor(
                max_workers=MAX_WORKERS
            ) as executor:

                futures = {
                    executor.submit(
                        run_seed_cv_experiment,
                        dataset_name,
                        seed,
                        target_indices,
                    ): seed
                    for seed in random_seeds
                }

                for future in as_completed(futures):
                    seed = futures[future]
                    seed_ret, cv_df, oof_r2_by_model = future.result()

                    assert seed_ret == seed

                    cv_df = cv_df.copy()
                    cv_df.insert(0, "Dataset", dataset_name)
                    cv_df.insert(1, "Seed", seed)
                    cv_rows.append(cv_df)
                    oof_r2_by_seed[seed] = oof_r2_by_model

                    print(
                        f"[Seed {seed}] CV collected for {dataset_name} | "
                        f"pooled OOF R2 ready"
                    )

            # ---------------------------------------------------------
            # Phase 2: aggregate the three seed-level OOF R2 values and
            # select one global k for each downstream learner.
            # ---------------------------------------------------------
            global_k_by_model, global_k_detail_df = (
                select_global_k_across_seeds(oof_r2_by_seed)
            )

            print(
                f"[Global k] {dataset_name}: {global_k_by_model}"
            )

            for _, row in global_k_detail_df.iterrows():
                global_k_rows.append(
                    {
                        "Dataset": dataset_name,
                        "Model": row["Model"],
                        "K": int(row["K"]),
                        "Mean_OOF_R2": float(row["Mean_OOF_R2"]),
                        "LocalSupport": float(row["LocalSupport"]),
                        "Selected": bool(row["Selected"]),
                    }
                )

            for model_name in ["RR", "RF", "MLP"]:
                best_k_rows.append(
                    {
                        "Dataset": dataset_name,
                        "Model": model_name,
                        "BestK": global_k_by_model[model_name],
                    }
                )

            # ---------------------------------------------------------
            # Phase 3: final test evaluation. Every seed uses the same
            # global k selected above for each learner.
            # ---------------------------------------------------------
            with ProcessPoolExecutor(
                max_workers=MAX_WORKERS
            ) as executor:

                futures = {
                    executor.submit(
                        run_final_evaluation,
                        dataset_name,
                        seed,
                        target_indices,
                        target_cluster_id,
                        cluster_sizes,
                        global_k_by_model,
                    ): seed
                    for seed in random_seeds
                }

                for future in as_completed(futures):
                    seed = futures[future]
                    seed_ret, seed_rows = future.result()
                    assert seed_ret == seed
                    final_rows.extend(seed_rows)

                    print(
                        f"[Seed {seed}] FINAL collected for {dataset_name}"
                    )

            # ---------------------------------------------------------
            # Write per-dataset sheets
            # ---------------------------------------------------------
            ds_final = pd.DataFrame(
                [
                    r for r in final_rows
                    if r["Dataset"] == dataset_name
                    and r["Section"] == "FinalTest"
                ]
            )

            ds_split = pd.DataFrame(
                [
                    r for r in final_rows
                    if r["Dataset"] == dataset_name
                    and r["Section"] == "Split"
                ]
            )

            if cv_rows:
                ds_cv = pd.concat(
                    [
                        x for x in cv_rows
                        if x["Dataset"].iloc[0] == dataset_name
                    ],
                    ignore_index=True,
                )
            else:
                ds_cv = pd.DataFrame()

            ds_best_k = pd.DataFrame(
                [
                    r for r in best_k_rows
                    if r["Dataset"] == dataset_name
                ]
            )

            ds_global_k = pd.DataFrame(
                [
                    r for r in global_k_rows
                    if r["Dataset"] == dataset_name
                ]
            )

            ds_cluster = pd.DataFrame(
                [
                    r for r in cluster_rows
                    if r["Dataset"] == dataset_name
                ]
            )

            ds_final.to_excel(
                writer,
                sheet_name=f"{dataset_name}_test",
                index=False,
            )

            ds_split.to_excel(
                writer,
                sheet_name=f"{dataset_name}_split",
                index=False,
            )

            ds_cv.to_excel(
                writer,
                sheet_name=f"{dataset_name}_cv",
                index=False,
            )

            ds_best_k.to_excel(
                writer,
                sheet_name=f"{dataset_name}_bestk",
                index=False,
            )

            ds_global_k.to_excel(
                writer,
                sheet_name=f"{dataset_name}_globalk",
                index=False,
            )

            ds_cluster.to_excel(
                writer,
                sheet_name=f"{dataset_name}_clusters",
                index=False,
            )

            # Aggregate over 3 seeds.
            if not ds_final.empty:
                agg = (
                    ds_final
                    .groupby(
                        ["Model", "Method"],
                        as_index=False,
                    )
                    .agg(
                        R2_mean=("R2", "mean"),
                        R2_std=("R2", "std"),
                        RMSE_mean=("RMSE", "mean"),
                        RMSE_std=("RMSE", "std"),
                        MAE_mean=("MAE", "mean"),
                        MAE_std=("MAE", "std"),
                        TrainSubsetSize_mean=(
                            "TrainSubsetSize",
                            "mean",
                        ),
                    )
                )

                # Keep for the cross-dataset summary sheet.
                agg_by_dataset[dataset_name] = agg

                agg.to_excel(
                    writer,
                    sheet_name=f"{dataset_name}_agg",
                    index=False,
                )

    # -------------------------------------------------------------
    # Save flat tables as additional Excel sheets.
    # -------------------------------------------------------------
    with pd.ExcelWriter(
        output_path,
        mode="a",
        engine="openpyxl",
    ) as writer:

        pd.DataFrame(final_rows).to_excel(
            writer,
            sheet_name="ALL_seed_results",
            index=False,
        )

        if cv_rows:
            pd.concat(
                cv_rows,
                ignore_index=True,
            ).to_excel(
                writer,
                sheet_name="ALL_cv",
                index=False,
            )

        pd.DataFrame(best_k_rows).to_excel(
            writer,
            sheet_name="ALL_bestk",
            index=False,
        )

        pd.DataFrame(global_k_rows).to_excel(
            writer,
            sheet_name="ALL_globalk",
            index=False,
        )

        pd.DataFrame(cluster_rows).to_excel(
            writer,
            sheet_name="ALL_clusters",
            index=False,
        )

        # ---------------------------------------------------------
        # Cross-dataset R2 summary: leftmost the agg key columns
        # (Model, Method), then each dataset's R2_mean / R2_std placed
        # side by side -> 2 + 6 * 2 = 14 columns in total.
        # ---------------------------------------------------------
        agg_summary = None

        for dataset_name in dataset_names:
            agg = agg_by_dataset.get(dataset_name)
            if agg is None or agg.empty:
                continue

            part = agg[["Model", "Method", "R2_mean", "R2_std"]].copy()
            part = part.rename(
                columns={
                    "R2_mean": f"{dataset_name}_R2_mean",
                    "R2_std": f"{dataset_name}_R2_std",
                }
            )

            if agg_summary is None:
                agg_summary = part
            else:
                agg_summary = agg_summary.merge(
                    part,
                    on=["Model", "Method"],
                    how="outer",
                )

        if agg_summary is not None and not agg_summary.empty:
            summary_cols = ["Model", "Method"]
            for dataset_name in dataset_names:
                summary_cols += [
                    f"{dataset_name}_R2_mean",
                    f"{dataset_name}_R2_std",
                ]
            summary_cols = [
                c for c in summary_cols if c in agg_summary.columns
            ]
            agg_summary = agg_summary[summary_cols]
            agg_summary.to_excel(
                writer,
                sheet_name="ALL_agg_summary",
                index=False,
            )

    print(f"\n[Saved] -> {output_path}")
    print("=== Clustered Target-Domain Disjoint Task Evaluation complete ===")

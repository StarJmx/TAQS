"""Compute a relative task-domain shift indicator for the TAQS disjoint setting.

This script reproduces the partition used by disjoint_task_evaluation_experiment:
1) Classical Gower distance on the full dataset's X only.
2) K-Medoids with 20 clusters and random_state=42.
3) Select the largest cluster with size <= 20% of D; randomly sample 5% of D
   from it to form Target Pool P (seed=42).
4) Randomly split P into Task Q and Evaluation E (50/50, seed=42).
5) Compute Task->Candidate and Candidate->Candidate nearest-neighbour distances,
   then rho = median(Task->Candidate NN distance) /
              median(Candidate internal NN distance).

The Gower matrix is computed using the same feature-wise normalization and
single-column categorical treatment as the supplied experiment code. Excel
output includes per-sample details for Steps 1 and 2, and the six-dataset rho
summary for Step 3.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# These constants intentionally match the supplied disjoint experiment script.
DATASET_NAMES = ["LAQ", "SCD", "EVEE", "MN1B", "ATRIV", "DRGD"]
N_TARGET_CLUSTERS = 20
TARGET_POOL_RATIO = 0.05
TARGET_POOL_MAX_CLUSTER_RATIO = 0.20
CLUSTER_SEED = 42
TARGET_SAMPLE_SEED = 42
TASK_EXTRACT_SEED = 42
TASK_RATIO_OF_TARGET_POOL = 0.50
CANDIDATE_NN_BLOCK_SIZE = 256

# Optional explicit project root; leave as None for automatic detection.
PROJECT_ROOT_OVERRIDE: Optional[str] = None
DEFAULT_OUTPUT_NAME = "task_domain_shift_analysis.xlsx"


def find_project_root() -> Path:
    """Find project root containing src/data_processor.py and the dataset YAML."""
    candidates = []
    if PROJECT_ROOT_OVERRIDE:
        candidates.append(Path(PROJECT_ROOT_OVERRIDE).expanduser().resolve())

    for base in (Path(__file__).resolve().parent, Path.cwd()):
        candidates.extend([base, *base.parents])

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (
            (candidate / "src" / "data_processor.py").is_file()
            and (candidate / "configs" / "datasets_config.yaml").is_file()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the TAQS project root. Expected both "
        "'src/data_processor.py' and 'configs/datasets_config.yaml'. "
        "Place this script in the project or set PROJECT_ROOT_OVERRIDE at the top."
    )


def to_optional_int(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in ("none", "null", ""):
            return None
        return int(stripped)
    return int(value)


def get_feature_indices(config):
    num_idx = list(range(int(config["num_start_idx"]), int(config["num_end_idx"]) + 1))
    cat_start = int(config["num_end_idx"]) + 1
    cat_end = to_optional_int(config.get("cat_end_idx"))
    cat_idx = list(range(cat_start, cat_end + 1)) if cat_end is not None and cat_end >= cat_start else []
    n_features = int(config.get("num_features", 0))
    # Validate against the actual feature matrix later; this variable is not used
    # to infer a categorical range, since the target column is excluded already.
    _ = n_features
    return num_idx, cat_idx


def resolve_data_path(file_path, project_root: Path) -> Path:
    """Resolve configured dataset path while tolerating common project CWDs."""
    raw = Path(file_path).expanduser()
    if raw.is_absolute():
        if raw.exists():
            return raw.resolve()
        raise FileNotFoundError(f"Configured data file does not exist: {raw}")

    candidates = [Path.cwd() / raw, project_root / raw, project_root / "src" / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = "\n  - ".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(f"Could not resolve data file '{file_path}'. Paths tried:\n  - {tried}")


def compute_gower_distance_matrix(X, num_idx, cat_idx):
    """Classical mixed-type Gower distance, matching the supplied experiment."""
    n = X.shape[0]
    distances = np.zeros((n, n), dtype=np.float64)
    feature_count = 0

    if num_idx:
        X_num = X[:, num_idx].astype(np.float64)
        for j in range(X_num.shape[1]):
            col = X_num[:, j]
            value_range = float(np.nanmax(col) - np.nanmin(col))
            if not np.isfinite(value_range) or value_range == 0:
                value_range = 1.0
            distances += np.abs(col[:, None] - col[None, :]) / value_range
            feature_count += 1

    if cat_idx:
        X_cat = X[:, cat_idx]
        for j in range(X_cat.shape[1]):
            col = X_cat[:, j]
            distances += (col[:, None] != col[None, :]).astype(np.float64)
            feature_count += 1

    if feature_count == 0:
        raise ValueError("No features were configured for the Gower distance.")
    distances /= feature_count

    if not np.isfinite(distances).all():
        raise ValueError(
            "The Gower matrix contains NaN/inf values. Check the datasets for "
            "missing/non-finite values and preprocessing, as K-Medoids requires finite distances."
        )
    return distances


def construct_target_pool(distance_matrix, n_total):
    """Construct target pool indices exactly as in the supplied experiment."""
    from sklearn_extra.cluster import KMedoids

    model = KMedoids(
        n_clusters=N_TARGET_CLUSTERS,
        metric="precomputed",
        random_state=CLUSTER_SEED,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Cluster\s+\d+\s+is empty", category=UserWarning)
        model.fit(distance_matrix)

    labels = np.asarray(model.labels_, dtype=int)
    cluster_sizes = np.bincount(labels, minlength=N_TARGET_CLUSTERS).astype(int)

    max_source_size = TARGET_POOL_MAX_CLUSTER_RATIO * n_total
    eligible = np.flatnonzero(cluster_sizes <= max_source_size)
    if eligible.size == 0:
        target_cluster_id = int(np.argmax(cluster_sizes))
    else:
        # Largest eligible cluster; on ties, select the lower cluster ID.
        best_pos = np.lexsort((eligible, -cluster_sizes[eligible]))[0]
        target_cluster_id = int(eligible[best_pos])

    source_indices = np.flatnonzero(labels == target_cluster_id).astype(int)
    n_target = int(round(TARGET_POOL_RATIO * n_total))
    n_target = max(1, min(n_target, source_indices.size))

    rng = np.random.default_rng(TARGET_SAMPLE_SEED)
    sampled_positions = rng.choice(source_indices.size, size=n_target, replace=False)
    target_indices = np.sort(source_indices[sampled_positions]).astype(int)

    return target_indices, target_cluster_id, cluster_sizes, int(source_indices.size)


def compute_task_to_candidate_details(distance_matrix, task_indices, candidate_indices, dataset_name):
    """Step 1: one nearest-Candidate Gower distance for every Task sample."""
    block = distance_matrix[np.ix_(task_indices, candidate_indices)]
    nearest_positions = np.argmin(block, axis=1)
    nearest_distances = block[np.arange(len(task_indices)), nearest_positions]
    nearest_candidate_indices = candidate_indices[nearest_positions]

    return pd.DataFrame(
        {
            "Dataset": dataset_name,
            "Task_Row_Index_0based": task_indices,
            "Nearest_Candidate_Row_Index_0based": nearest_candidate_indices,
            "Task_to_Candidate_NN_Gower_Distance": nearest_distances,
        }
    )


def compute_candidate_internal_details(distance_matrix, candidate_indices, dataset_name):
    """Step 2: nearest *other* Candidate distance, processed in row blocks."""
    n_candidate = len(candidate_indices)
    nearest_distances = np.empty(n_candidate, dtype=np.float64)
    nearest_indices = np.empty(n_candidate, dtype=int)
    column_positions = np.arange(n_candidate)

    for start in range(0, n_candidate, CANDIDATE_NN_BLOCK_SIZE):
        end = min(start + CANDIDATE_NN_BLOCK_SIZE, n_candidate)
        row_indices = candidate_indices[start:end]
        block = distance_matrix[np.ix_(row_indices, candidate_indices)]

        # Exclude each sample's distance to itself (global diagonal).
        block[np.arange(end - start), column_positions[start:end]] = np.inf
        nearest_pos = np.argmin(block, axis=1)
        nearest_distances[start:end] = block[np.arange(end - start), nearest_pos]
        nearest_indices[start:end] = candidate_indices[nearest_pos]

    return pd.DataFrame(
        {
            "Dataset": dataset_name,
            "Candidate_Row_Index_0based": candidate_indices,
            "Nearest_Other_Candidate_Row_Index_0based": nearest_indices,
            "Candidate_Internal_NN_Gower_Distance": nearest_distances,
        }
    )


def summarise_distances(task_details, candidate_details, dataset_name):
    """Step 3: summarize both distance distributions and compute rho."""
    task_dist = task_details["Task_to_Candidate_NN_Gower_Distance"].to_numpy(dtype=float)
    candidate_dist = candidate_details["Candidate_Internal_NN_Gower_Distance"].to_numpy(dtype=float)

    task_median = float(np.median(task_dist))
    candidate_median = float(np.median(candidate_dist))
    rho = float(task_median / candidate_median) if candidate_median > 0 else np.nan

    return {
        "Dataset": dataset_name,
        "Task_Size_m": int(task_dist.size),
        "Candidate_Size_n": int(candidate_dist.size),
        "Task_to_Candidate_NN_Median": task_median,
        "Task_to_Candidate_NN_Mean": float(np.mean(task_dist)),
        "Task_to_Candidate_NN_Q1": float(np.quantile(task_dist, 0.25)),
        "Task_to_Candidate_NN_Q3": float(np.quantile(task_dist, 0.75)),
        "Candidate_Internal_NN_Median": candidate_median,
        "Candidate_Internal_NN_Mean": float(np.mean(candidate_dist)),
        "Candidate_Internal_NN_Q1": float(np.quantile(candidate_dist, 0.25)),
        "Candidate_Internal_NN_Q3": float(np.quantile(candidate_dist, 0.75)),
        "rho": rho,
        "rho_Interpretation": (
            "Undefined: candidate internal median NN distance is zero"
            if not np.isfinite(rho)
            else "Task farther relative to Candidate internal spacing" if rho > 1
            else "Task approximately at Candidate internal spacing" if np.isclose(rho, 1.0)
            else "Task closer relative to Candidate internal spacing"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output Excel path. Default: <project_root>/results/task_domain_shift_analysis.xlsx",
    )
    args = parser.parse_args()

    project_root = find_project_root()
    # Add root to import path so the script can be run from elsewhere.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src import data_processor as dp

    config_path = project_root / "configs" / "datasets_config.yaml"
    output_path = Path(args.output).expanduser().resolve() if args.output else project_root / "results" / DEFAULT_OUTPUT_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    task_detail_frames = []
    candidate_detail_frames = []
    split_summary_rows = []
    cluster_detail_rows = []

    for dataset_name in DATASET_NAMES:
        print(f"\n{'=' * 72}\nDataset: {dataset_name}\n{'=' * 72}")
        config = dp.load_dataset_config(str(config_path), dataset_name)
        processor = dp.DataProcessor()
        data_path = resolve_data_path(config["file_path"], project_root)
        _, data = processor.load_data(str(data_path))
        data = np.asarray(data)

        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"{dataset_name}: expected a 2D data matrix with features and a target column.")

        X = data[:, :-1]  # label/target column is not used for clustering or distance analysis
        num_idx, cat_idx = get_feature_indices(config)
        if (num_idx and max(num_idx) >= X.shape[1]) or (cat_idx and max(cat_idx) >= X.shape[1]):
            raise ValueError(
                f"{dataset_name}: configured feature indices exceed X's {X.shape[1]} columns. "
                "Check num_end_idx/cat_end_idx in datasets_config.yaml."
            )

        print(f"[Gower] Computing full {len(X)} x {len(X)} matrix (matching clustering protocol)...")
        distance_matrix = compute_gower_distance_matrix(X, num_idx, cat_idx)

        target_indices, target_cluster_id, cluster_sizes, source_cluster_size = construct_target_pool(
            distance_matrix, len(data)
        )
        target_mask = np.zeros(len(data), dtype=bool)
        target_mask[target_indices] = True
        candidate_indices = np.flatnonzero(~target_mask).astype(int)

        n_task = max(1, int(round(len(target_indices) * TASK_RATIO_OF_TARGET_POOL)))
        n_task = min(n_task, len(target_indices) - 1) if len(target_indices) > 1 else len(target_indices)
        rng_task = np.random.default_rng(TASK_EXTRACT_SEED)
        task_local_indices = np.sort(rng_task.choice(len(target_indices), size=n_task, replace=False))
        task_indices = target_indices[task_local_indices]
        evaluation_local_indices = np.setdiff1d(np.arange(len(target_indices)), task_local_indices, assume_unique=True)
        evaluation_indices = target_indices[evaluation_local_indices]

        # Confirm the exact intended partition: Q and E are disjoint; C is the remainder.
        if np.intersect1d(task_indices, evaluation_indices).size:
            raise RuntimeError(f"{dataset_name}: Task and Evaluation sets overlap.")
        if len(candidate_indices) + len(task_indices) + len(evaluation_indices) != len(data):
            raise RuntimeError(f"{dataset_name}: Candidate/Task/Evaluation partition is inconsistent.")

        task_details = compute_task_to_candidate_details(
            distance_matrix, task_indices, candidate_indices, dataset_name
        )
        candidate_details = compute_candidate_internal_details(
            distance_matrix, candidate_indices, dataset_name
        )
        summary = summarise_distances(task_details, candidate_details, dataset_name)
        summary_rows.append(summary)
        task_detail_frames.append(task_details)
        candidate_detail_frames.append(candidate_details)

        split_summary_rows.append(
            {
                "Dataset": dataset_name,
                "Total_N": int(len(data)),
                "Source_Cluster_ID": int(target_cluster_id),
                "Source_Cluster_Size": int(source_cluster_size),
                "Target_Pool_Size": int(len(target_indices)),
                "Candidate_Size": int(len(candidate_indices)),
                "Task_Size": int(len(task_indices)),
                "Evaluation_Size": int(len(evaluation_indices)),
                "Candidate_Ratio": len(candidate_indices) / len(data),
                "Task_Ratio": len(task_indices) / len(data),
                "Evaluation_Ratio": len(evaluation_indices) / len(data),
                "Cluster_Seed": CLUSTER_SEED,
                "Target_Sample_Seed": TARGET_SAMPLE_SEED,
                "Task_Extract_Seed": TASK_EXTRACT_SEED,
                "Num_Clusters": N_TARGET_CLUSTERS,
                "Target_Pool_Ratio_Config": TARGET_POOL_RATIO,
                "Source_Cluster_Max_Ratio": TARGET_POOL_MAX_CLUSTER_RATIO,
            }
        )

        for cluster_id, size in enumerate(cluster_sizes.tolist()):
            cluster_detail_rows.append(
                {
                    "Dataset": dataset_name,
                    "Cluster_ID": int(cluster_id),
                    "Cluster_Size": int(size),
                    "Cluster_Ratio_of_Dataset": float(size / len(data)),
                    "Selected_Source_Cluster": bool(cluster_id == target_cluster_id),
                }
            )

        print(
            f"[Result] N={len(data)}, Candidate={len(candidate_indices)}, "
            f"Task={len(task_indices)}, Evaluation={len(evaluation_indices)}, "
            f"median(Task->Candidate NN)={summary['Task_to_Candidate_NN_Median']:.6f}, "
            f"median(Candidate internal NN)={summary['Candidate_Internal_NN_Median']:.6f}, "
            f"rho={summary['rho']:.6f}" if np.isfinite(summary["rho"]) else
            f"[Result] N={len(data)}, rho undefined because candidate NN median is zero"
        )
        # Release the large pairwise matrix before loading the next dataset.
        del distance_matrix

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="rho_summary", index=False)
        pd.concat(task_detail_frames, ignore_index=True).to_excel(
            writer, sheet_name="task_to_candidate", index=False
        )
        pd.concat(candidate_detail_frames, ignore_index=True).to_excel(
            writer, sheet_name="candidate_internal", index=False
        )
        pd.DataFrame(split_summary_rows).to_excel(writer, sheet_name="split_summary", index=False)
        pd.DataFrame(cluster_detail_rows).to_excel(writer, sheet_name="cluster_sizes", index=False)

    print(f"\n[Saved] {output_path}")
    print("Sheets: rho_summary, task_to_candidate, candidate_internal, split_summary, cluster_sizes")


if __name__ == "__main__":
    main()

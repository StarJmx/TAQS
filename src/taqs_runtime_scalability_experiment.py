"""
TAQS Runtime & Scalability Experiment
======================================

The script is intentionally limited to selection-time / memory benchmarking;
no downstream model training, prediction, evaluation, or CV-based k search is
performed.

Sections
--------
A. Main runtime benchmark on six Stage-II datasets:
   TAQS selection-only vs. Classical-Gower K-Medoids selection-only.

B. Controlled TAQS-only scaling:
   n-scaling on PAIN, m-scaling on PAIN, d-scaling on ATRIV.

Output
------
One Excel workbook with exactly four sheets:
    main_runtime
    n_scaling
    m_scaling
    d_scaling
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
import sys
import time
import traceback
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import psutil
from sklearn_extra.cluster import KMedoids
from sklearn.exceptions import ConvergenceWarning

from src import data_processor as dp
from src import taqs_algorithm as taqs


# -----------------------------------------------------------------------------
# Fixed benchmark environment / implementation settings
# -----------------------------------------------------------------------------
THREADS_PER_PROCESS = int(os.environ.get("TAQS_THREADS_PER_PROCESS", "4"))
os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"

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


# Six Stage-II datasets from the existing experiment.
MAIN_DATASETS = ["LAQ", "SCD", "EVEE", "MN1B", "ATRIV", "DRGD"]

# Runtime benchmark settings.
K_RUNTIME = 50
RUNTIME_SEED = 42
N_REPEATS = 5
WARMUP_RUNS = 1

# Main input split: fixed deterministic split; no random split seed is used.
CANDIDATE_RATIO = 0.98
TASK_RATIO = 0.02

# Controlled scaling settings.
N_SCALING_DATASET = "PAIN"
M_SCALING_DATASET = "PAIN"
D_SCALING_DATASET = "ATRIV"

N_SCALING_VALUES = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000]
N_SCALING_FIXED_M = 100

M_SCALING_VALUES = [10, 25, 50, 100, 200, 500, 1_000, 2_000]
M_SCALING_FIXED_N = 8_000

D_SCALING_VALUES = [16, 32, 64, 128, 256, 512, 1_024, 1_033]
D_SCALING_FIXED_N = 2_000
D_SCALING_FIXED_M = 100

# Small memory-sampling interval. This does not affect the algorithm being
# benchmarked; it is only used to approximate peak RSS during a run.
MEMORY_SAMPLE_INTERVAL_SEC = 0.002

OUTPUT_PATH = "../results/_temporary_taqs_runtime_scalability.xlsx"


# -----------------------------------------------------------------------------
# Generic helpers reused from the existing experiment design
# -----------------------------------------------------------------------------
def _to_optional_int(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("none", "null", ""):
            return None
        return int(value)
    return int(value)


def load_dataset(dataset_name: str):
    """Load one dataset through the existing project DataProcessor interface."""
    config = dp.load_dataset_config(
        "../configs/datasets_config.yaml",
        dataset_name,
    )
    processor = dp.DataProcessor()
    _, data = processor.load_data(config["file_path"])
    return config, data


def get_feature_indices(config) -> Tuple[List[int], List[int]]:
    """Return numerical and categorical feature indices from dataset config."""
    num_idx = list(
        range(
            config["num_start_idx"],
            config["num_end_idx"] + 1,
        )
    )

    cat_start = config["num_end_idx"] + 1
    cat_end = _to_optional_int(config["cat_end_idx"])
    cat_idx = (
        list(range(cat_start, cat_end + 1))
        if cat_end is not None and cat_end >= cat_start
        else []
    )
    return num_idx, cat_idx


def standardize(train_x: np.ndarray, other_x_list: Sequence[np.ndarray], num_slice: slice):
    """Candidate-fitted numerical standardization; categorical columns unchanged."""
    from sklearn.preprocessing import StandardScaler

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


def build_fixed_runtime_split(
    data: np.ndarray,
    candidate_ratio: float = CANDIDATE_RATIO,
):
    """
    Deterministically create the fixed 90% Candidate / 10% Task input split.

    No random seed is used. The original row order is preserved.
    """
    if not (0.0 < candidate_ratio < 1.0):
        raise ValueError("candidate_ratio must be between 0 and 1.")

    split_idx = int(round(data.shape[0] * candidate_ratio))
    split_idx = max(1, min(split_idx, data.shape[0] - 1))

    candidate_raw = data[:split_idx]
    task_raw = data[split_idx:]
    return candidate_raw, task_raw


def prepare_selection_inputs(
    candidate_raw: np.ndarray,
    task_raw: np.ndarray,
    config,
):
    """
    Prepare the exact input representation used for selection.

    Standardization is deliberately performed outside the timed interval and
    is fitted only on the Candidate Pool, then applied unchanged to Task.
    """
    num_slice = slice(
        config["num_start_idx"],
        config["num_end_idx"] + 1,
    )

    x_candidate = candidate_raw[:, :-1]
    y_candidate = candidate_raw[:, -1]
    x_task = task_raw[:, :-1]
    y_task = task_raw[:, -1]

    x_candidate_std, transformed = standardize(
        x_candidate,
        [x_task],
        num_slice,
    )
    x_task_std = transformed[0]

    candidate_std = np.hstack(
        [x_candidate_std, y_candidate.reshape(-1, 1)]
    )
    task_std = np.hstack(
        [x_task_std, y_task.reshape(-1, 1)]
    )

    return candidate_std, task_std


# -----------------------------------------------------------------------------
# Classical Gower: same non-vectorized implementation style as the reference
# experiment. No additional vectorization is introduced for the benchmark.
# -----------------------------------------------------------------------------
def compute_gower_distance_matrix(
    X: np.ndarray,
    num_idx: Sequence[int],
    cat_idx: Sequence[int],
    feature_ranges=None,
):
    """Classical mixed-type Gower distance matrix."""
    n = X.shape[0]
    distance_matrix = np.zeros((n, n), dtype=np.float64)
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

            distance_matrix += (
                np.abs(col[:, None] - col[None, :]) / r
            )
            feature_count += 1

    if cat_idx:
        X_cat = X[:, cat_idx]
        for j in range(X_cat.shape[1]):
            col = X_cat[:, j]
            distance_matrix += (
                col[:, None] != col[None, :]
            ).astype(np.float64)
            feature_count += 1

    return (
        distance_matrix / feature_count
        if feature_count
        else distance_matrix
    )


# -----------------------------------------------------------------------------
# Production selection calls
# -----------------------------------------------------------------------------
def run_taqs_selection(
    candidate_std: np.ndarray,
    task_std: np.ndarray,
    config,
    k: int,
):
    """Run the existing TAQS selector exactly once and return the subset."""
    selector = taqs.TaskAwareQueryAnchoredSelection(
        data_task=task_std,
        data_train=candidate_std,
        n_neighbors=k,
        num_start_idx=config["num_start_idx"],
        num_end_idx=config["num_end_idx"],
        cat_end_idx=_to_optional_int(config["cat_end_idx"]),
        feature_weights=config["feature_weights"],
    )
    return selector.select_subset()


def run_kmedoids_selection(
    candidate_std: np.ndarray,
    config,
    subset_size: int,
    seed: int = RUNTIME_SEED,
):
    """
    Run the existing Classical-Gower + K-Medoids baseline.

    The returned subset size is matched exactly to the TAQS output.
    """
    n_candidate = candidate_std.shape[0]
    subset_size = max(1, min(int(subset_size), n_candidate))

    if subset_size >= n_candidate:
        return candidate_std.copy()

    X_candidate = candidate_std[:, :-1]
    num_idx, cat_idx = get_feature_indices(config)

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

    return candidate_std[np.asarray(model.medoid_indices_, dtype=int)]


# -----------------------------------------------------------------------------
# Peak RSS / isolated-process benchmarking
# -----------------------------------------------------------------------------
def get_peak_memory_bytes() -> int:
    """Return OS-level peak memory for the current benchmark process."""
    process = psutil.Process(os.getpid())
    info = process.memory_info()

    # Windows exposes Peak Working Set through psutil.memory_info().
    peak_wset = getattr(info, "peak_wset", None)
    if peak_wset is not None:
        return int(peak_wset)

    # POSIX fallback: ru_maxrss is KiB on Linux and bytes on macOS.
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(value)
        return int(value * 1024)
    except (ImportError, AttributeError, ValueError):
        # Last-resort fallback for platforms without an OS-level peak metric.
        return int(info.rss)


def _isolated_selection_worker(method_name, payload, conn):
    """Run one selection benchmark inside a fresh child process."""
    try:
        if method_name == "TAQS":
            result = run_taqs_selection(
                payload["candidate_std"],
                payload["task_std"],
                payload["config"],
                payload["k"],
            )
        elif method_name == "K-Medoids":
            result = run_kmedoids_selection(
                payload["candidate_std"],
                payload["config"],
                payload["subset_size"],
                seed=payload["seed"],
            )
        else:
            raise ValueError(f"Unknown benchmark method: {method_name}")

        peak_memory_bytes = get_peak_memory_bytes()
        subset_size = int(result.shape[0])
        conn.send(
            {
                "ok": True,
                "subset_size": subset_size,
                "peak_memory_bytes": peak_memory_bytes,
            }
        )
    except BaseException as exc:
        conn.send(
            {
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        conn.close()


def run_isolated_selection(method_name, payload):
    """Run one selection call in a fresh serial child process.

    Data serialization/process startup occurs outside the timed algorithm
    interval. Peak memory is measured inside the child by the operating system,
    avoiding cross-method allocator/cache contamination and sub-millisecond
    RSS polling limitations.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_isolated_selection_worker,
        args=(method_name, payload, child_conn),
    )
    process.start()
    child_conn.close()

    result = None
    recv_error = None
    try:
        result = parent_conn.recv()
    except (EOFError, BrokenPipeError) as exc:
        recv_error = exc
    finally:
        parent_conn.close()
        process.join()

    if result is None:
        raise RuntimeError(
            f"{method_name} benchmark child exited with code "
            f"{process.exitcode}; no result was received: {recv_error!r}"
        )

    if not result.get("ok", False):
        raise RuntimeError(
            f"{method_name} benchmark failed: {result['error']}\n"
            f"{result['traceback']}"
        )

    return result


# -----------------------------------------------------------------------------
# Benchmark runner: one warm-up + five serial timed repeats
# -----------------------------------------------------------------------------
def measure_selection(
    method_name: str,
    payload: Dict,
    label: str,
    repeats: int = N_REPEATS,
    warmup_runs: int = WARMUP_RUNS,
):
    """
    Benchmark one fixed selection configuration.

    Each warm-up/timed run uses a fresh child process. Process startup and
    input serialization occur outside the algorithm timer. Timed runs are
    strictly serial and use identical input/configuration. Peak memory is the
    OS-level peak memory of the isolated benchmark process.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    print(f"  [Warm-up] {label}")
    for _ in range(warmup_runs):
        _ = run_isolated_selection(method_name, payload)
        gc.collect()

    times = []
    peak_memories = []
    subset_size = None

    for run_id in range(1, repeats + 1):
        gc.collect()
        # The child starts the stopwatch immediately before the selection
        # call, so spawn/IPC overhead is excluded from algorithm runtime.
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(
            target=_timed_selection_worker,
            args=(method_name, payload, child_conn),
        )
        process.start()
        child_conn.close()

        result = None
        recv_error = None
        try:
            result = parent_conn.recv()
        except (EOFError, BrokenPipeError) as exc:
            recv_error = exc
        finally:
            parent_conn.close()
            process.join()

        if result is None:
            raise RuntimeError(
                f"{method_name} timed benchmark child exited with code "
                f"{process.exitcode}; no result was received: {recv_error!r}"
            )
        if not result.get("ok", False):
            raise RuntimeError(
                f"{method_name} timed benchmark failed: {result['error']}\n"
                f"{result['traceback']}"
            )

        elapsed = float(result["elapsed_seconds"])
        peak_rss = int(result["peak_memory_bytes"])
        subset_size = int(result["subset_size"])

        times.append(elapsed)
        peak_memories.append(peak_rss)
        print(
            f"    run {run_id}/{repeats}: "
            f"{elapsed:.6f} s, peak RSS={peak_rss / 1024**2:.2f} MB"
        )
        gc.collect()

    time_median = float(np.median(times))
    time_q1, time_q3 = np.percentile(times, [25, 75])
    time_iqr = float(time_q3 - time_q1)
    memory_median = float(np.median(peak_memories))

    return {
        "times": times,
        "peak_memories_bytes": peak_memories,
        "time_median": time_median,
        "time_iqr": time_iqr,
        "peak_memory_median_mb": memory_median / 1024**2,
        "subset_size": subset_size,
    }


def _timed_selection_worker(method_name, payload, conn):
    """Time one production selection call inside a fresh child process."""
    try:
        start = time.perf_counter()
        if method_name == "TAQS":
            result = run_taqs_selection(
                payload["candidate_std"],
                payload["task_std"],
                payload["config"],
                payload["k"],
            )
        elif method_name == "K-Medoids":
            result = run_kmedoids_selection(
                payload["candidate_std"],
                payload["config"],
                payload["subset_size"],
                seed=payload["seed"],
            )
        else:
            raise ValueError(f"Unknown benchmark method: {method_name}")
        elapsed = time.perf_counter() - start

        conn.send(
            {
                "ok": True,
                "elapsed_seconds": float(elapsed),
                "subset_size": int(result.shape[0]),
                "peak_memory_bytes": get_peak_memory_bytes(),
            }
        )
    except BaseException as exc:
        conn.send(
            {
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Dataset-specific utilities
# -----------------------------------------------------------------------------
def load_and_split_dataset(dataset_name: str):
    config, data = load_dataset(dataset_name)
    candidate_raw, task_raw = build_fixed_runtime_split(data)
    candidate_std, task_std = prepare_selection_inputs(
        candidate_raw,
        task_raw,
        config,
    )
    return config, data, candidate_raw, task_raw, candidate_std, task_std


def build_prefix_config(config, d: int):
    """
    Create a configuration for an ATRIV prefix feature subset.

    The d-scaling experiment is defined over the original ATRIV feature order.
    ATRIV is expected to place its numerical features first and categorical
    features contiguously afterward. Prefix d therefore includes numeric
    features first, then categorical features in their original single-column
    representation.
    """
    cfg = deepcopy(config)
    num_idx, cat_idx = get_feature_indices(config)

    if not num_idx:
        raise ValueError("ATRIV d-scaling requires at least one numeric feature.")
    if num_idx != list(range(0, len(num_idx))):
        raise ValueError(
            "ATRIV numeric features must occupy the leading contiguous columns "
            "0..num_end_idx for prefix d-scaling."
        )
    if cat_idx and cat_idx != list(range(len(num_idx), len(num_idx) + len(cat_idx))):
        raise ValueError(
            "ATRIV categorical features must follow the numeric block contiguously "
            "for prefix d-scaling."
        )

    full_feature_count = len(num_idx) + len(cat_idx)
    if d < 1 or d > full_feature_count:
        raise ValueError(
            f"Requested d={d}, but dataset configuration exposes "
            f"{full_feature_count} features."
        )

    n_numeric_selected = min(d, len(num_idx))
    n_categorical_selected = max(0, d - len(num_idx))

    cfg["num_start_idx"] = 0
    cfg["num_end_idx"] = n_numeric_selected - 1
    if n_categorical_selected > 0:
        # Preserve the original single-column categorical representation.
        cfg["cat_end_idx"] = len(num_idx) + n_categorical_selected - 1
    else:
        cfg["cat_end_idx"] = None

    weights = np.asarray(config["feature_weights"], dtype=np.float64)
    if weights.shape[0] < d:
        raise ValueError(
            f"feature_weights has length {weights.shape[0]}, cannot form d={d}."
        )
    cfg["feature_weights"] = weights[:d].tolist()
    return cfg


def prefix_features_with_label(data: np.ndarray, d: int):
    """Keep the first d feature columns in original order and reattach y."""
    if data.shape[1] - 1 < d:
        raise ValueError(
            f"Requested d={d}, but only {data.shape[1] - 1} features are available."
        )
    return np.hstack([data[:, :d], data[:, -1].reshape(-1, 1)])


def subset_rows(data: np.ndarray, n: int) -> np.ndarray:
    if n < 1 or n > data.shape[0]:
        raise ValueError(
            f"Requested n={n}, available rows={data.shape[0]}"
        )
    return data[:n]


# -----------------------------------------------------------------------------
# A. Main runtime table
# -----------------------------------------------------------------------------
def run_main_runtime() -> pd.DataFrame:
    rows = []

    print("\n" + "=" * 88)
    print("A. MAIN RUNTIME BENCHMARK")
    print("=" * 88)

    for dataset_name in MAIN_DATASETS:
        print(f"\n[Dataset] {dataset_name}")
        config, data, candidate_raw, task_raw, candidate_std, task_std = (
            load_and_split_dataset(dataset_name)
        )

        n_candidate = candidate_std.shape[0]
        m_task = task_std.shape[0]
        d_features = candidate_std.shape[1] - 1

        print(
            f"  input: Candidate={n_candidate}, Task={m_task}, d={d_features}, "
            f"K_RUNTIME={K_RUNTIME}"
        )

        # Pre-run once outside timing so the matched subset size is known.
        taqs_subset = run_taqs_selection(
            candidate_std,
            task_std,
            config,
            K_RUNTIME,
        )
        taqs_subset_size = int(taqs_subset.shape[0])
        del taqs_subset
        gc.collect()

        taqs_payload = {
            "candidate_std": candidate_std,
            "task_std": task_std,
            "config": config,
            "k": K_RUNTIME,
        }
        kmedoids_payload = {
            "candidate_std": candidate_std,
            "config": config,
            "subset_size": taqs_subset_size,
            "seed": RUNTIME_SEED,
        }

        taqs_result = measure_selection(
            "TAQS",
            taqs_payload,
            label=f"{dataset_name} / TAQS",
        )

        kmedoids_result = measure_selection(
            "K-Medoids",
            kmedoids_payload,
            label=f"{dataset_name} / K-Medoids",
        )

        rows.append(
            {
                "Dataset": dataset_name,
                "n_Candidate": n_candidate,
                "m_Task": m_task,
                "d": d_features,
                "K_Runtime": K_RUNTIME,
                "TAQS_SubsetSize": taqs_subset_size,
                "KMedoids_SubsetSize": taqs_subset_size,
                "TAQS_Time_Run1": taqs_result["times"][0],
                "TAQS_Time_Run2": taqs_result["times"][1],
                "TAQS_Time_Run3": taqs_result["times"][2],
                "TAQS_Time_Run4": taqs_result["times"][3],
                "TAQS_Time_Run5": taqs_result["times"][4],
                "TAQS_Time_Median": taqs_result["time_median"],
                "TAQS_Time_IQR": taqs_result["time_iqr"],
                "KMedoids_Time_Run1": kmedoids_result["times"][0],
                "KMedoids_Time_Run2": kmedoids_result["times"][1],
                "KMedoids_Time_Run3": kmedoids_result["times"][2],
                "KMedoids_Time_Run4": kmedoids_result["times"][3],
                "KMedoids_Time_Run5": kmedoids_result["times"][4],
                "KMedoids_Time_Median": kmedoids_result["time_median"],
                "KMedoids_Time_IQR": kmedoids_result["time_iqr"],
                "TAQS_PeakMemory_Median_MB": taqs_result[
                    "peak_memory_median_mb"
                ],
                "KMedoids_PeakMemory_Median_MB": kmedoids_result[
                    "peak_memory_median_mb"
                ],
            }
        )

        del candidate_raw, task_raw, candidate_std, task_std
        gc.collect()

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# B. n-scaling: PAIN, TAQS only
# -----------------------------------------------------------------------------
def run_n_scaling() -> pd.DataFrame:
    dataset_name = N_SCALING_DATASET
    config, data = load_dataset(dataset_name)
    candidate_raw, task_raw = build_fixed_runtime_split(data)

    fixed_m = N_SCALING_FIXED_M
    task_raw_fixed = subset_rows(task_raw, fixed_m)

    rows = []
    print("\n" + "=" * 88)
    print(
        f"B. N-SCALING ({dataset_name}) | fixed m={fixed_m}, "
        f"d={data.shape[1] - 1}, K={K_RUNTIME}"
    )
    print("=" * 88)

    for n_value in N_SCALING_VALUES:
        if n_value > candidate_raw.shape[0]:
            print(f"  [Skip] n={n_value}: exceeds Candidate Pool size")
            continue

        candidate_n_raw = subset_rows(candidate_raw, n_value)
        candidate_n_std, task_std = prepare_selection_inputs(
            candidate_n_raw,
            task_raw_fixed,
            config,
        )

        result = measure_selection(
            "TAQS",
            {
                "candidate_std": candidate_n_std,
                "task_std": task_std,
                "config": config,
                "k": K_RUNTIME,
            },
            label=f"{dataset_name} / n={n_value}",
        )

        rows.append(
            {
                "Dataset": dataset_name,
                "n": int(n_value),
                "m": int(fixed_m),
                "d": int(candidate_n_std.shape[1] - 1),
                "K_Runtime": K_RUNTIME,
                "Time_Run1": result["times"][0],
                "Time_Run2": result["times"][1],
                "Time_Run3": result["times"][2],
                "Time_Run4": result["times"][3],
                "Time_Run5": result["times"][4],
                "Time_Median": result["time_median"],
                "Time_IQR": result["time_iqr"],
                "PeakMemory_Median_MB": result[
                    "peak_memory_median_mb"
                ],
            }
        )

        del candidate_n_raw, candidate_n_std, task_std
        gc.collect()

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# C. m-scaling: PAIN, TAQS only
# -----------------------------------------------------------------------------
def run_m_scaling() -> pd.DataFrame:
    dataset_name = M_SCALING_DATASET
    config, data = load_dataset(dataset_name)
    candidate_raw, task_raw = build_fixed_runtime_split(data)

    fixed_n = M_SCALING_FIXED_N
    candidate_fixed_raw = subset_rows(candidate_raw, fixed_n)

    rows = []
    print("\n" + "=" * 88)
    print(
        f"C. M-SCALING ({dataset_name}) | fixed n={fixed_n}, "
        f"d={data.shape[1] - 1}, K={K_RUNTIME}"
    )
    print("=" * 88)

    for m_value in M_SCALING_VALUES:
        if m_value > task_raw.shape[0]:
            print(f"  [Skip] m={m_value}: exceeds Task Pool size")
            continue

        task_m_raw = subset_rows(task_raw, m_value)
        candidate_std, task_std = prepare_selection_inputs(
            candidate_fixed_raw,
            task_m_raw,
            config,
        )

        result = measure_selection(
            "TAQS",
            {
                "candidate_std": candidate_std,
                "task_std": task_std,
                "config": config,
                "k": K_RUNTIME,
            },
            label=f"{dataset_name} / m={m_value}",
        )

        rows.append(
            {
                "Dataset": dataset_name,
                "n": int(candidate_std.shape[0]),
                "m": int(m_value),
                "d": int(candidate_std.shape[1] - 1),
                "K_Runtime": K_RUNTIME,
                "Time_Run1": result["times"][0],
                "Time_Run2": result["times"][1],
                "Time_Run3": result["times"][2],
                "Time_Run4": result["times"][3],
                "Time_Run5": result["times"][4],
                "Time_Median": result["time_median"],
                "Time_IQR": result["time_iqr"],
                "PeakMemory_Median_MB": result[
                    "peak_memory_median_mb"
                ],
            }
        )

        del task_m_raw, candidate_std, task_std
        gc.collect()

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# D. d-scaling: ATRIV, TAQS only
# -----------------------------------------------------------------------------
def run_d_scaling() -> pd.DataFrame:
    dataset_name = D_SCALING_DATASET
    base_config, data = load_dataset(dataset_name)
    candidate_raw, task_raw = build_fixed_runtime_split(data)

    fixed_n = min(D_SCALING_FIXED_N, candidate_raw.shape[0])
    fixed_m = min(D_SCALING_FIXED_M, task_raw.shape[0])
    candidate_base_raw = subset_rows(candidate_raw, fixed_n)
    task_base_raw = subset_rows(task_raw, fixed_m)

    rows = []
    print("\n" + "=" * 88)
    print(
        f"D. D-SCALING ({dataset_name}) | fixed n={fixed_n}, m={fixed_m}, "
        f"K={K_RUNTIME}"
    )
    print("=" * 88)

    full_d = data.shape[1] - 1

    for d_value in D_SCALING_VALUES:
        if d_value > full_d:
            print(f"  [Skip] d={d_value}: exceeds available feature count")
            continue

        scaled_config = build_prefix_config(base_config, d_value)
        candidate_d_raw = prefix_features_with_label(
            candidate_base_raw,
            d_value,
        )
        task_d_raw = prefix_features_with_label(
            task_base_raw,
            d_value,
        )

        candidate_std, task_std = prepare_selection_inputs(
            candidate_d_raw,
            task_d_raw,
            scaled_config,
        )

        result = measure_selection(
            "TAQS",
            {
                "candidate_std": candidate_std,
                "task_std": task_std,
                "config": scaled_config,
                "k": K_RUNTIME,
            },
            label=f"{dataset_name} / d={d_value}",
        )

        rows.append(
            {
                "Dataset": dataset_name,
                "n": int(candidate_std.shape[0]),
                "m": int(task_std.shape[0]),
                "d": int(d_value),
                "K_Runtime": K_RUNTIME,
                "Time_Run1": result["times"][0],
                "Time_Run2": result["times"][1],
                "Time_Run3": result["times"][2],
                "Time_Run4": result["times"][3],
                "Time_Run5": result["times"][4],
                "Time_Median": result["time_median"],
                "Time_IQR": result["time_iqr"],
                "PeakMemory_Median_MB": result[
                    "peak_memory_median_mb"
                ],
            }
        )

        del candidate_d_raw, task_d_raw, candidate_std, task_std
        gc.collect()

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Excel output
# -----------------------------------------------------------------------------
def save_results(
    main_runtime: pd.DataFrame,
    n_scaling: pd.DataFrame,
    m_scaling: pd.DataFrame,
    d_scaling: pd.DataFrame,
):
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        main_runtime.to_excel(
            writer,
            sheet_name="main_runtime",
            index=False,
        )
        n_scaling.to_excel(
            writer,
            sheet_name="n_scaling",
            index=False,
        )
        m_scaling.to_excel(
            writer,
            sheet_name="m_scaling",
            index=False,
        )
        d_scaling.to_excel(
            writer,
            sheet_name="d_scaling",
            index=False,
        )

    print(f"\n[Saved] -> {output_path.resolve()}")
    print("Sheets: main_runtime, n_scaling, m_scaling, d_scaling")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 88)
    print("TAQS Runtime & Scalability Experiment")
    print("=" * 88)
    print(f"THREADS_PER_PROCESS={THREADS_PER_PROCESS}")
    print(f"K_RUNTIME={K_RUNTIME}")
    print(f"RUNTIME_SEED={RUNTIME_SEED}")
    print(f"N_REPEATS={N_REPEATS}, WARMUP_RUNS={WARMUP_RUNS}")
    print(f"OUTPUT_PATH={OUTPUT_PATH}")
    print("No CV; no k search; no downstream model training; serial isolated-process benchmark execution.")

    main_runtime = run_main_runtime()
    n_scaling = run_n_scaling()
    m_scaling = run_m_scaling()
    d_scaling = run_d_scaling()

    save_results(
        main_runtime=main_runtime,
        n_scaling=n_scaling,
        m_scaling=m_scaling,
        d_scaling=d_scaling,
    )


if __name__ == "__main__":
    main()

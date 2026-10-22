"""
TAQS Web Demo - Flask Backend
==============================
Run:  python app.py
Open: http://localhost:5000
"""

import io
import os
import time
import traceback

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

from src.taqs_algorithm import TaskAwareQueryAnchoredSelection

app = Flask(__name__, static_folder=".", static_url_path="")

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(BASE_DIR, "data", "demo")

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(".", "taqs_demo_flask.html")


# ---------------------------------------------------------------------------
# API: Load demo data
# ---------------------------------------------------------------------------

@app.route("/api/load_demo", methods=["GET"])
def api_load_demo():
    """
    Load demo task/train CSV files and readme text.

    Response JSON:
    {
        "success": true,
        "task_csv": "<csv text>",
        "train_csv": "<csv text>",
        "task_filename": "demo_task_P001_P010.csv",
        "train_filename": "demo_train_P011_P200.csv",
        "readme": "<readme text>",
        "demo_params": {
            "n_neighbors": 25,
            "feature_weights": [0.0208, 0.0203, ...],
            "weights_mode": "custom"
        }
    }
    """
    try:
        task_path = os.path.join(DEMO_DIR, "demo_task_P001_P010.csv")
        train_path = os.path.join(DEMO_DIR, "demo_train_P011_P200.csv")
        readme_path = os.path.join(DEMO_DIR, "readme.txt")

        if not os.path.exists(task_path):
            return jsonify({"success": False, "error": f"Demo file not found: {task_path}"}), 404
        if not os.path.exists(train_path):
            return jsonify({"success": False, "error": f"Demo file not found: {train_path}"}), 404

        with open(task_path, "r", encoding="utf-8") as f:
            task_csv = f.read()
        with open(train_path, "r", encoding="utf-8") as f:
            train_csv = f.read()

        readme = ""
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8") as f:
                readme = f.read().strip()

        return jsonify({
            "success": True,
            "task_csv": task_csv,
            "train_csv": train_csv,
            "task_filename": "demo_task_P001_P010.csv",
            "train_filename": "demo_train_P011_P200.csv",
            "readme": readme,
            "demo_params": {
                "n_neighbors": 25,
                "feature_weights": [
                    0.020804876, 0.020330146, 0.020003606,
                    0.59147507, 0.2098362, 0.062149666, 0.075400412
                ],
                "weights_mode": "custom",
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_csv_text(text: str) -> pd.DataFrame:
    """Parse raw CSV text sent from the front-end into a DataFrame."""
    return pd.read_csv(io.StringIO(text))


def _build_taqs(df_task: pd.DataFrame, df_train: pd.DataFrame, params: dict):
    """Construct a TaskAwareQueryAnchoredSelection instance from request params."""
    n_features = df_train.shape[1] - 1  # last column is target

    num_start = int(params.get("num_start_idx", 0))
    num_end = int(params.get("num_end_idx", n_features - 1))
    cat_end_raw = params.get("cat_end_idx", None)
    cat_end = int(cat_end_raw) if cat_end_raw is not None and int(cat_end_raw) >= 0 else None

    # Feature weights
    weights_raw = params.get("feature_weights", None)
    if weights_raw is not None and len(weights_raw) > 0:
        weights = np.array([float(w) for w in weights_raw], dtype=float)
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum
    else:
        weights = None  # uniform inside TAQS

    n_neighbors = int(params.get("n_neighbors", 50))

    # Convert DataFrames to numpy (keep original dtypes for Gower)
    data_task = df_task.values
    data_train = df_train.values

    taqs = TaskAwareQueryAnchoredSelection(
        data_task=data_task,
        data_train=data_train,
        n_neighbors=n_neighbors,
        num_start_idx=num_start,
        num_end_idx=num_end,
        cat_end_idx=cat_end,
        feature_weights=weights,
    )
    return taqs


def _subset_to_records(subset: np.ndarray, columns: list) -> list:
    """Convert numpy subset array to list-of-dicts for JSON."""
    return [dict(zip(columns, [str(v) for v in row])) for row in subset]


# ---------------------------------------------------------------------------
# API: run TAQS
# ---------------------------------------------------------------------------

@app.route("/api/run_taqs", methods=["POST"])
def api_run_taqs():
    """
    Request JSON body:
    {
        "task_csv":   "<raw csv text>",
        "train_csv":  "<raw csv text>",
        "n_neighbors": 50,
        "num_start_idx": 0,
        "num_end_idx": 6,
        "cat_end_idx": -1,          // -1 or null means no categorical
        "feature_weights": [0.1, 0.2, ...]  // optional, null/empty = uniform
    }

    Response JSON:
    {
        "success": true,
        "subset_size": 1234,
        "train_total": 5000,
        "task_total": 200,
        "selection_rate": 24.68,
        "elapsed_seconds": 1.23,
        "n_features": 7,
        "effective_k": 50,
        "headers": ["col1", "col2", ...],
        "subset": [ {"col1": "v1", ...}, ... ],   // first 200 rows for preview
        "subset_total_rows": 1234
    }
    """
    try:
        body = request.get_json(force=True)
        task_csv = body.get("task_csv", "")
        train_csv = body.get("train_csv", "")

        if not task_csv or not train_csv:
            return jsonify({"success": False, "error": "Please upload Task and Train data"}), 400

        df_task = _parse_csv_text(task_csv)
        df_train = _parse_csv_text(train_csv)

        if df_task.shape[1] != df_train.shape[1]:
            return jsonify({
                "success": False,
                "error": f"Column count mismatch: Task({df_task.shape[1]}) vs Train({df_train.shape[1]})"
            }), 400

        taqs = _build_taqs(df_task, df_train, body)

        t0 = time.time()
        subset = taqs.select_subset()
        elapsed = round(time.time() - t0, 3)

        columns = df_train.columns.tolist()
        n_features = df_train.shape[1] - 1
        effective_k = min(int(body.get("n_neighbors", 50)), df_train.shape[0])

        # Preview: first 200 rows
        preview_rows = _subset_to_records(subset[:200], columns)

        return jsonify({
            "success": True,
            "subset_size": int(subset.shape[0]),
            "train_total": int(df_train.shape[0]),
            "task_total": int(df_task.shape[0]),
            "selection_rate": round(100.0 * subset.shape[0] / df_train.shape[0], 1),
            "elapsed_seconds": elapsed,
            "n_features": n_features,
            "effective_k": effective_k,
            "headers": columns,
            "subset": preview_rows,
            "subset_total_rows": int(subset.shape[0]),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Multi-K analysis
# ---------------------------------------------------------------------------

@app.route("/api/multi_k", methods=["POST"])
def api_multi_k():
    """
    Request JSON body: same as /api/run_taqs plus
    {
        "k_list": [10, 20, 50, 100]
    }

    Response JSON:
    {
        "success": true,
        "results": [
            {"k": 10, "subset_size": 500, "selection_rate": 10.0, "avg_mean_distance": 0.123},
            ...
        ]
    }
    """
    try:
        body = request.get_json(force=True)
        k_list = body.get("k_list", [])
        if not k_list:
            return jsonify({"success": False, "error": "k_list is empty"}), 400

        k_list = [int(k) for k in k_list if int(k) > 0]

        task_csv = body.get("task_csv", "")
        train_csv = body.get("train_csv", "")
        df_task = _parse_csv_text(task_csv)
        df_train = _parse_csv_text(train_csv)

        taqs = _build_taqs(df_task, df_train, body)

        t0 = time.time()
        subset_dict = taqs.select_subsets_for_k_list(k_list)
        dist_dict = taqs.mean_query_topk_distance_for_k_list(k_list)
        elapsed = round(time.time() - t0, 3)

        results = []
        for k in k_list:
            subset_k = subset_dict.get(k, np.array([]))
            size = int(subset_k.shape[0]) if hasattr(subset_k, "shape") else 0
            results.append({
                "k": k,
                "subset_size": size,
                "selection_rate": round(100.0 * size / df_train.shape[0], 1),
                "avg_mean_distance": round(dist_dict.get(k, float("nan")), 6),
            })

        return jsonify({
            "success": True,
            "results": results,
            "elapsed_seconds": elapsed,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Download subset CSV
# ---------------------------------------------------------------------------

@app.route("/api/download_subset", methods=["POST"])
def api_download_subset():
    """
    Same body as /api/run_taqs.  Returns raw CSV text.
    Optional query param:  ?k=50   to download for a specific K.
    """
    try:
        body = request.get_json(force=True)
        k_specific = request.args.get("k", None)

        task_csv = body.get("task_csv", "")
        train_csv = body.get("train_csv", "")
        df_task = _parse_csv_text(task_csv)
        df_train = _parse_csv_text(train_csv)

        taqs = _build_taqs(df_task, df_train, body)

        if k_specific is not None:
            k_val = int(k_specific)
            subset_dict = taqs.select_subsets_for_k_list([k_val])
            subset = subset_dict.get(k_val, np.array([]))
        else:
            subset = taqs.select_subset()

        columns = df_train.columns.tolist()
        df_subset = pd.DataFrame(subset, columns=columns)

        csv_buffer = io.StringIO()
        df_subset.to_csv(csv_buffer, index=False)
        csv_text = csv_buffer.getvalue()

        return app.response_class(csv_text, mimetype="text/csv")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("  TAQS Web Demo")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)

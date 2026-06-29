"""
application_case_demonstration.py

Application demo on the PAIN dataset:
- exploring_k(): sweep neighbor count K with blocked CV on an exploring split
- main(k): train XGBoost on a TAQS subset vs full training data and predict task rows
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import data_processor as dp
from src import taqs_algorithm as taqs
from src import utils


DATASET_NAME = "PAIN"
K_NEIGHBORS = [1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200]
N_FOLDS = 10
OUTPUT_FILE = "_application_case_demonstration_exploring_K.xlsx"
OUTPUT_FILE_TASK = "_application_case_demonstration.xlsx"


def _to_optional_int(value):
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("none", "null", ""):
            return None
        return int(v)
    return int(value)


def standardize(train_x, test_x, num_slice):
    """Z-score quantitative columns using train-fold statistics; apply to test."""
    scaler = StandardScaler()
    scaler.fit(train_x[:, num_slice].astype(float))
    train_std = train_x.copy()
    test_std = test_x.copy()
    train_std[:, num_slice] = scaler.transform(train_x[:, num_slice].astype(float))
    test_std[:, num_slice] = scaler.transform(test_x[:, num_slice].astype(float))
    return train_std, test_std


def exploring_k():
    """Sweep K on blocked folds and write summary / fold / prediction sheets to Excel."""
    project_root = PROJECT_ROOT
    config_path = project_root / "configs" / "datasets_config.yaml"
    output_path = project_root / "results" / OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = dp.load_dataset_config(str(config_path), DATASET_NAME)
    processor = dp.DataProcessor()

    config_dir = config_path.parent
    data_rel = Path(config["file_path_exploring_k"])
    data_path = str(data_rel if data_rel.is_absolute() else (config_dir / data_rel).resolve())

    _, data_full = processor.load_data(data_path, random_state=42)

    num_slice = slice(config["num_start_idx"], config["num_end_idx"] + 1)

    n_samples = data_full.shape[0]
    indices = np.arange(n_samples)
    fold_size = n_samples // N_FOLDS

    rows_performance = []
    rows_predictions = []

    for fold_idx in range(N_FOLDS):
        val_start = fold_idx * fold_size
        val_end = (fold_idx + 1) * fold_size if fold_idx < N_FOLDS - 1 else n_samples

        val_mask = np.zeros(n_samples, dtype=bool)
        val_mask[val_start:val_end] = True
        train_mask = ~val_mask

        data_train = data_full[train_mask]
        data_val = data_full[val_mask]

        if data_train.shape[0] == 0 or data_val.shape[0] == 0:
            continue

        train_x = data_train[:, :-1]
        train_y = data_train[:, -1]
        val_x = data_val[:, :-1]
        val_y = data_val[:, -1]

        train_x_std, val_x_std = standardize(train_x, val_x, num_slice)
        data_train_std = np.hstack([train_x_std, train_y.reshape(-1, 1)])
        data_val_std = np.hstack([val_x_std, val_y.reshape(-1, 1)])

        selector = taqs.TaskAwareQueryAnchoredSelection(
            data_task=data_val_std,
            data_train=data_train_std,
            n_neighbors=max(K_NEIGHBORS),
            num_start_idx=config["num_start_idx"],
            num_end_idx=config["num_end_idx"],
            cat_end_idx=_to_optional_int(config["cat_end_idx"]),
            feature_weights=config["feature_weights"],
        )
        subset_by_n = selector.select_subsets_for_k_list(K_NEIGHBORS)

        for K in K_NEIGHBORS:
            subset = subset_by_n[K]
            sub_x = subset[:, :-1]
            sub_y = subset[:, -1]

            model = XGBRegressor(n_estimators=150, max_depth=15, min_child_weight=5,
                                 learning_rate=0.1, colsample_bytree=1, gamma=0,
                                 subsample=0.9, n_jobs=-1)
            model.fit(sub_x, sub_y)
            pred = model.predict(val_x_std)
            metrics = utils.evaluate_regression(val_y, pred, verbose=False)

            rows_performance.append({
                "Fold": fold_idx + 1,
                "K": K,
                "TrainSize": int(len(sub_y)),
                "ValSize": int(len(val_y)),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            })

            for i, (y_t, y_p) in enumerate(zip(val_y, pred)):
                rows_predictions.append({
                    "Fold": fold_idx + 1,
                    "K": K,
                    "ValSampleIdx": i,
                    "y_true": y_t,
                    "y_pred": y_p,
                })

    df_perf = pd.DataFrame(rows_performance)
    df_pred = pd.DataFrame(rows_predictions)

    agg_rows = []
    for K in K_NEIGHBORS:
        sub_df = df_perf[df_perf["K"] == K]
        agg_rows.append({
            "K": K,
            "AvgTrainSize": sub_df["TrainSize"].mean(),
            "MAE_mean": sub_df["MAE"].mean(),
            "MAE_std": sub_df["MAE"].std(),
            "RMSE_mean": sub_df["RMSE"].mean(),
            "RMSE_std": sub_df["RMSE"].std(),
            "R2_mean": sub_df["R2"].mean(),
            "R2_std": sub_df["R2"].std(),
        })

    df_agg = pd.DataFrame(agg_rows)

    with pd.ExcelWriter(output_path) as writer:
        df_agg.to_excel(writer, sheet_name="summary", index=False)
        df_perf.to_excel(writer, sheet_name="fold_details", index=False)
        df_pred.to_excel(writer, sheet_name="predictions", index=False)

    print(f"[Saved] -> {output_path}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 300)
    print(df_agg.to_string(index=False))


def main(k: int):
    """Run TAQS subset vs full-train XGBoost baselines on the configured task split."""
    print(f"best k: {k}")

    project_root = PROJECT_ROOT
    config_path = project_root / "configs" / "datasets_config.yaml"
    output_path = project_root / "results" / OUTPUT_FILE_TASK
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = dp.load_dataset_config(str(config_path), DATASET_NAME)
    processor = dp.DataProcessor()

    config_dir = config_path.parent
    train_rel = Path(config["file_path"])
    task_rel = Path(config["file_path_task"])
    train_path = str(train_rel if train_rel.is_absolute() else (config_dir / train_rel).resolve())
    task_path = str(task_rel if task_rel.is_absolute() else (config_dir / task_rel).resolve())

    _, data_train = processor.load_data(train_path)
    _, data_task = processor.load_data(task_path)
    num_slice = slice(config["num_start_idx"], config["num_end_idx"] + 1)

    data_train_std, data_task_std = standardize(
        train_x=data_train,
        test_x=data_task,
        num_slice=num_slice,
    )

    selector = taqs.TaskAwareQueryAnchoredSelection(
        data_task=data_task_std,
        data_train=data_train_std,
        n_neighbors=k,
        num_start_idx=config["num_start_idx"],
        num_end_idx=config["num_end_idx"],
        cat_end_idx=_to_optional_int(config["cat_end_idx"]),
        feature_weights=config["feature_weights"],
    )
    subset_train = selector.select_subset()

    x_task = data_task_std[:, :-1]
    y_task = data_task_std[:, -1]

    x_subset = subset_train[:, :-1]
    y_subset = subset_train[:, -1]

    x_full = data_train_std[:, :-1]
    y_full = data_train_std[:, -1]

    rf_subset = XGBRegressor(
        n_estimators=150,
        max_depth=15,
        min_child_weight=5,
        learning_rate=0.1,
        colsample_bytree=1,
        gamma=0,
        subsample=0.9,
        n_jobs=-1,
    )
    rf_subset.fit(x_subset, y_subset)
    y_pred_subset = rf_subset.predict(x_task)
    metrics_subset = utils.evaluate_regression(y_task, y_pred_subset, verbose=False)

    rf_full = XGBRegressor(
        n_estimators=150,
        max_depth=15,
        min_child_weight=5,
        learning_rate=0.1,
        colsample_bytree=1,
        gamma=0,
        subsample=0.9,
        n_jobs=-1,
    )
    rf_full.fit(x_full, y_full)
    y_pred_full = rf_full.predict(x_task)
    metrics_full = utils.evaluate_regression(y_task, y_pred_full, verbose=False)

    df_performance = pd.DataFrame(
        [
            {
                "Setting": f"TAQS subset (K={k})",
                "TrainSize": int(len(y_subset)),
                "MAE": metrics_subset["MAE"],
                "RMSE": metrics_subset["RMSE"],
                "R2": metrics_subset["R2"],
            },
            {
                "Setting": "Full training set",
                "TrainSize": int(len(y_full)),
                "MAE": metrics_full["MAE"],
                "RMSE": metrics_full["RMSE"],
                "R2": metrics_full["R2"],
            },
        ]
    )

    df_predictions = pd.DataFrame(
        {
            "y_true_task": y_task,
            "y_pred_taqs_subset_rf": y_pred_subset,
            "y_pred_full_train_rf": y_pred_full,
        }
    )

    with pd.ExcelWriter(output_path) as writer:
        df_performance.to_excel(writer, sheet_name="performance", index=False)
        df_predictions.to_excel(writer, sheet_name="predictions", index=False)

    print(f"[Saved] Demonstration results -> {output_path}")
    print(df_performance)


if __name__ == "__main__":
    exploring = False
    # Set exploring=True to search K on the exploring split; otherwise run main(k).
    if exploring:
        exploring_k()
    else:
        main(25)

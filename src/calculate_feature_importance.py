"""
calculate_feature_importance.py

Pipeline:
1. Load data via DataProcessor
2. Tune XGBoost with two-stage GridSearchCV (5-fold CV, R2)
3. Refit the best model on the full dataset
4. Compute normalized mean-|SHAP| feature contributions
5. Save results to Excel

Notes:
- Regression task; the last column is the target
- Categorical features must already be encoded in the input file
"""

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score

import data_processor as dp


def train_xgboost_with_cv(X, y):
    """
    Two-stage GridSearchCV tuning (5-fold CV, R2 scoring).

    Stage 1 searches tree-structure hyperparameters; stage 2 fine-tunes
    learning-rate and sampling parameters around the stage-1 winner.

    Returns
    -------
    best_model
        Fitted estimator from stage 2.
    tuning_info : dict
        Best params and CV R2 from both stages.
    """

    print("\n[Stage 1] Coarse Grid Search (structure parameters)")

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    )

    param_grid_stage1 = {
        "n_estimators": [100, 150, 200, 250, 300],
        "max_depth": [5, 10, 15],
        "min_child_weight": [0, 1, 3, 5, 10],
    }

    grid1 = GridSearchCV(
        base_model,
        param_grid_stage1,
        cv=5,
        scoring="r2",
        verbose=1,
        n_jobs=-1
    )

    grid1.fit(X, y)

    print("\n[Stage 1 Result]")
    print("Best Params:", grid1.best_params_)
    print("Best R2:", grid1.best_score_)

    best_params_stage1 = grid1.best_params_

    print("\n[Stage 2] Fine Grid Search (learning parameters)")

    model_stage2 = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **best_params_stage1,
    )

    param_grid_stage2 = {
        "learning_rate": [0.01, 0.1, 0.2, 0.3],
        "gamma": [0, 0.1, 0.5],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0]
    }

    grid2 = GridSearchCV(
        model_stage2,
        param_grid_stage2,
        cv=5,
        scoring="r2",
        verbose=1,
        n_jobs=-1
    )

    grid2.fit(X, y)

    print("\n[Stage 2 Result]")
    print("Best Params:", grid2.best_params_)
    print("Best R2:", grid2.best_score_)

    best_model = grid2.best_estimator_
    tuning_info = {
        "stage1_best_params": grid1.best_params_,
        "stage1_best_r2": float(grid1.best_score_),
        "stage2_best_params": grid2.best_params_,
        "stage2_best_r2": float(grid2.best_score_),
    }

    return best_model, tuning_info


def calculate_feature_contribution(X, model, feature_names):
    """
    Compute normalized mean absolute SHAP values per feature.

    Returns
    -------
    contributions : np.ndarray
        Feature weights that sum to 1 when total importance is positive.
    """

    print("\n[Step 2] Calculating SHAP feature importance")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X, check_additivity=False)

    shap_abs = np.abs(shap_values.values)

    importance = np.mean(shap_abs, axis=0)
    total = np.sum(importance)

    contributions = importance / total if total > 0 else np.zeros_like(importance)

    print("\nFeature Contributions:")
    for name, val in zip(feature_names, contributions):
        print(f"{name}: {val:.4f}")

    print("Sum:", contributions.sum())

    return contributions


if __name__ == "__main__":

    dataset_name = ["EVEE", "IHDP", "MHR", "OBES", "ROSS", "LAQ", "SCD",
                    "MN1B", "ATRIV", "DRGD", "PAIN"]
    config = dp.load_dataset_config("../configs/datasets_config.yaml", dataset_name[10])
    file_path = config["file_path"]

    processor = dp.DataProcessor()
    feature_names, data = processor.load_data(file_path, random_state=42)

    X = data[:, :-1]
    y = data[:, -1]

    print("\n[Data Info]")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Optional hold-out split for tuning diagnostics only.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    print("\nTrain size:", X_train.shape)
    print("Test size:", X_test.shape)

    best_model, tuning_info = train_xgboost_with_cv(X_train, y_train)

    print("\n[Step 3] Train best model on full dataset")
    best_model.fit(X, y)

    y_pred = best_model.predict(X)
    print("R2 (full data):", r2_score(y, y_pred))

    contributions = calculate_feature_contribution(X, best_model, feature_names[:-1])
    contributions = np.array(contributions)
    print(contributions)

    output_path = "../results/_temporary_feature_importance.xlsx"

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names[:-1],
        "Contribution": contributions
    })

    best_params = best_model.get_params()
    best_hyperparameters_df = pd.DataFrame(
        [
            {"Parameter": "stage1_best_params", "Value": str(tuning_info["stage1_best_params"])},
            {"Parameter": "stage1_best_r2", "Value": tuning_info["stage1_best_r2"]},
            {"Parameter": "stage2_best_params", "Value": str(tuning_info["stage2_best_params"])},
            {"Parameter": "stage2_best_r2", "Value": tuning_info["stage2_best_r2"]},
            {"Parameter": "full_data_r2", "Value": float(r2_score(y, y_pred))},
        ] + [
            {"Parameter": f"best_model.{k}", "Value": str(v)}
            for k, v in best_params.items()
        ]
    )

    with pd.ExcelWriter(output_path) as writer:
        feature_importance_df.to_excel(writer, sheet_name="feature_importance", index=False)
        best_hyperparameters_df.to_excel(writer, sheet_name="best_hyperparameters", index=False)

    print(f"[Saved] -> {output_path}")

    print("\n=== Done ===")

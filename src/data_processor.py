import json
from pathlib import Path

import yaml

import pandas as pd
import sklearn.preprocessing as sp
import numpy as np

from src import utils


def _load_feature_weights_from_path(weights_path: str) -> np.ndarray:
    """
    Load feature weights from .npy/.json/.csv and return a 1D float array.
    """
    ext = Path(weights_path).suffix.lower()

    if ext == ".npy":
        weights = np.load(weights_path)
    elif ext == ".json":
        with open(weights_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            if "feature_weights" in payload:
                payload = payload["feature_weights"]
            elif "weights" in payload:
                payload = payload["weights"]
            else:
                raise ValueError(
                    f"JSON at {weights_path} must contain 'feature_weights' or 'weights'"
                )
        weights = np.asarray(payload, dtype=float)
    elif ext == ".csv":
        weights_df = pd.read_csv(weights_path, header=None)
        if weights_df.shape[1] == 1:
            weights = weights_df.iloc[:, 0].to_numpy(dtype=float)
        elif weights_df.shape[0] == 1:
            weights = weights_df.iloc[0, :].to_numpy(dtype=float)
        else:
            weights = weights_df.to_numpy(dtype=float).reshape(-1)
    else:
        raise ValueError(
            f"Unsupported feature_weights file type: {ext}. Use .npy, .json, or .csv"
        )

    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size == 0:
        raise ValueError(f"feature_weights file is empty: {weights_path}")
    return weights


def load_dataset_config(config_path, dataset_name):
    """
    Read selected dataset config and normalize feature_weights.

    Returns
    -------
    config : dict
    """

    with open(config_path, "r", encoding="utf-8") as f:
        config_all = yaml.safe_load(f)

    dataset_config = config_all["datasets"][dataset_name]

    # feature_weights supports:
    # 1) inline array in YAML (existing behavior)
    # 2) external file path via feature_weights_path (.npy/.json/.csv)
    # Path-based weights take priority when both are provided.
    config_dir = Path(config_path).resolve().parent
    feature_weights = dataset_config.get("feature_weights")
    feature_weights_path = dataset_config.get("feature_weights_path")

    if feature_weights_path:
        weights_path = Path(feature_weights_path)
        if not weights_path.is_absolute():
            weights_path = (config_dir / weights_path).resolve()
        if weights_path.exists():
            feature_weights = _load_feature_weights_from_path(str(weights_path))
        elif feature_weights is not None and len(feature_weights) > 0:
            feature_weights = np.asarray(feature_weights, dtype=float)
            print(
                f"[Warning] feature_weights_path not found: {weights_path}. "
                f"Falling back to inline feature_weights."
            )
        else:
            feature_weights = None
            print(
                f"[Warning] feature_weights_path not found: {weights_path}. "
                f"Using uniform weights downstream."
            )
    elif feature_weights is not None and len(feature_weights) > 0:
        feature_weights = np.asarray(feature_weights, dtype=float)
    else:
        feature_weights = None

    dataset_config["feature_weights"] = feature_weights

    return dataset_config


class DataProcessor:
    def __init__(self):
        pass

    def load_data(self, file_path: str, random_state: int = None):
        """
        Load tabular data and optionally shuffle rows with a fixed random seed.

        Parameters
        ----------
        file_path : str
            Path to a .csv, .xls, or .xlsx file.
        random_state : int, optional
            Random seed for row shuffling. None keeps the original row order.

        Returns
        -------
        feature_names : list
            Column names.
        data_array : np.ndarray
            2D array matching the DataFrame; rows are shuffled when random_state is set.
        """

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel.")

        feature_names = df.columns.tolist()
        data_array = df.values

        if random_state is not None:
            rng = np.random.default_rng(random_state)
            rng.shuffle(data_array)

        print("Feature names:")
        print(feature_names)
        print("Data shape:", data_array.shape)

        return feature_names, data_array


    def standardize_continuous(self, data: np.ndarray, start_idx: int, end_idx: int):
        """
        Z-score standardize continuous columns in [start_idx, end_idx] via StandardScaler.

        Parameters
        ----------
        data : np.ndarray
            2D sample matrix. Only the specified column range must be numeric.
        start_idx : int
            Inclusive start column index for continuous features.
        end_idx : int
            Inclusive end column index for continuous features.

        Returns
        -------
        data_std : np.ndarray
            Copy of ``data`` with only the selected columns replaced by scaled values.
        """

        data_std = data.copy()
        cont_data = data_std[:, start_idx:end_idx + 1].astype(float)

        scaler = sp.StandardScaler()
        cont_data_std = scaler.fit_transform(cont_data)
        data_std[:, start_idx:end_idx + 1] = cont_data_std

        print("\n[Standardization Check]")
        print("Mean (should be close to 0):")
        print(cont_data_std.mean(axis=0))
        print("Std (should be close to 1):")
        print(cont_data_std.std(axis=0))

        return data_std


    def split_data(self, data: np.ndarray, train_ratio=0.9, verbose=True):
        """
        Split data into train and test sets by row order (not randomized).

        The last column is treated as the label; all preceding columns are features.

        Parameters
        ----------
        data : numpy.ndarray
            Sample matrix with features in all but the last column and y in the last column.
        train_ratio : float
            Fraction of rows assigned to training (taken from the front), default 0.9.
        verbose : bool
            Whether to print shapes and a sample row, default True.

        Returns
        -------
        x_train, x_test, y_train, y_test
            Sequential train/test feature and label arrays.
        """

        x = data[:, :-1]
        y = data[:, -1]

        if verbose:
            print("X shape:", x.shape)
            print("y shape:", y.shape)
            print("Example:", x[0], y[0])

        train_size = int(len(x) * train_ratio)

        x_train = x[:train_size]
        x_test = x[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        if verbose:
            print("Train shape:", x_train.shape, y_train.shape)

        return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    utils.shuffle_and_save_top_n_samples(
    input_path="../data/raw/superconductivty_data.csv",
    output_path = "../data/processed/superconductivty_data_5000.csv",
    n_samples = 5000,
    random_state = 42,
    )


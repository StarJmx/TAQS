import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union


class RegressionNoiseInjector:
    """
    Regression noise injector for mixed-type data.

    Supported noise:
    1. Continuous variables: outlier injection, Gaussian corruption.
    2. Categorical variables: label errors, category merge/split, encoding drift.
    3. Sample-level noise: inject both continuous and categorical noise on the same rows.

    Notes
    -----
    - No target(label) noise is injected by default (last column is protected).
    - `noise_strength` controls total noisy row ratio:
      small=50%, medium=70%, large=90%.
    """

    STRENGTH_TO_RATIO = {
        "small": 0.5,
        "medium": 0.7,
        "large": 0.9,
    }

    STRENGTH_PARAMS = {
        "small": {
            "outlier_magnitude": 1,
            "gaussian_std_ratio": 0.2,
            "categorical_error_rate": 0.2,
            "merge_rate": 0.3,
            "split_rate": 0.3,
            "drift_rate": 0.3,
            "sample_level_fraction": 0.2,
            "target_noise_std_ratio": 0.2,
            "target_noise_fraction": 0.2,
        },
        "medium": {
            "outlier_magnitude": 1.5,
            "gaussian_std_ratio": 0.4,
            "categorical_error_rate": 0.4,
            "merge_rate": 0.4,
            "split_rate": 0.4,
            "drift_rate": 0.4,
            "sample_level_fraction": 0.3,
            "target_noise_std_ratio": 0.3,
            "target_noise_fraction": 0.3,
        },
        "large": {
            "outlier_magnitude": 2,
            "gaussian_std_ratio": 0.8,
            "categorical_error_rate": 0.8,
            "merge_rate": 0.8,
            "split_rate": 0.8,
            "drift_rate": 0.8,
            "sample_level_fraction": 0.8,
            "target_noise_std_ratio": 0.8,
            "target_noise_fraction": 0.8,
        },
    }

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    # -------------------- column / row helpers --------------------

    @staticmethod
    def _as_dataframe(data: Union[pd.DataFrame, np.ndarray]) -> Tuple[pd.DataFrame, bool]:
        if isinstance(data, pd.DataFrame):
            return data.copy(), False
        if isinstance(data, np.ndarray):
            col_names = [f"col_{i}" for i in range(data.shape[1])]
            return pd.DataFrame(data.copy(), columns=col_names), True
        raise TypeError("`data` must be pd.DataFrame or np.ndarray.")

    @staticmethod
    def _restore_type(df: pd.DataFrame, as_ndarray: bool) -> Union[pd.DataFrame, np.ndarray]:
        return df.to_numpy() if as_ndarray else df

    @staticmethod
    def _normalize_columns(
            df: pd.DataFrame,
            columns: Optional[List[Union[str, int]]]
    ) -> List[str]:
        if not columns:
            return []
        normalized = []
        for c in columns:
            if isinstance(c, int):
                if 0 <= c < len(df.columns):
                    normalized.append(df.columns[c])
            elif c in df.columns:
                normalized.append(c)
        return normalized

    @staticmethod
    def _safe_cat_end_idx(cat_end_idx: Optional[Union[int, str]]) -> Optional[int]:
        if cat_end_idx is None:
            return None
        if isinstance(cat_end_idx, str):
            low = cat_end_idx.strip().lower()
            if low in ("none", "null", ""):
                return None
        return int(cat_end_idx)

    def _resolve_columns_from_config(
            self,
            df: pd.DataFrame,
            config: Dict
    ) -> Tuple[List[str], List[str], Optional[str]]:
        target_column = config.get("target_column", None)
        protect_target = config.get("protect_target", True)

        if target_column is None and protect_target and len(df.columns) > 0:
            target_col_name = df.columns[-1]
        elif isinstance(target_column, int):
            target_col_name = df.columns[target_column]
        elif isinstance(target_column, str):
            target_col_name = target_column if target_column in df.columns else None
        else:
            target_col_name = None

        num_cols = self._normalize_columns(df, config.get("continuous_columns"))
        cat_cols = self._normalize_columns(df, config.get("categorical_columns"))

        if not num_cols and not cat_cols:
            num_start = config.get("num_start_idx")
            num_end = config.get("num_end_idx")
            cat_end = self._safe_cat_end_idx(config.get("cat_end_idx"))

            if num_start is not None and num_end is not None:
                num_cols = list(df.columns[int(num_start): int(num_end) + 1])
            if num_end is not None and cat_end is not None and cat_end >= int(num_end) + 1:
                cat_cols = list(df.columns[int(num_end) + 1: cat_end + 1])

        if not num_cols and not cat_cols:
            # fallback by dtype if config is not provided
            feature_cols = list(df.columns)
            if target_col_name in feature_cols:
                feature_cols.remove(target_col_name)
            num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
            cat_cols = [c for c in feature_cols if c not in num_cols]

        if target_col_name in num_cols:
            num_cols.remove(target_col_name)
        if target_col_name in cat_cols:
            cat_cols.remove(target_col_name)

        return num_cols, cat_cols, target_col_name

    def _sample_rows(self, n_rows: int, ratio: float) -> np.ndarray:
        n = int(round(n_rows * ratio))
        if n <= 0:
            return np.array([], dtype=int)
        n = min(n, n_rows)
        return self.rng.choice(n_rows, size=n, replace=False)

    # -------------------- continuous noise --------------------

    def add_outlier_noise_continuous(
            self,
            data: pd.DataFrame,
            columns: List[str],
            outlier_fraction: float = 0.02,
            magnitude: float = 4.0,
            row_indices: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        noisy = data.copy()
        if not columns:
            return noisy

        candidate_rows = np.arange(len(noisy)) if row_indices is None else np.array(row_indices, dtype=int)
        if len(candidate_rows) == 0:
            return noisy

        n_outliers = int(round(len(candidate_rows) * outlier_fraction))
        if n_outliers <= 0:
            return noisy

        outlier_rows = self.rng.choice(candidate_rows, size=min(n_outliers, len(candidate_rows)), replace=False)

        for col in columns:
            if col not in noisy.columns:
                continue
            series = pd.to_numeric(noisy[col], errors="coerce")
            mean_val = float(series.mean())
            std_val = float(series.std())
            if np.isnan(std_val) or std_val == 0:
                continue

            directions = self.rng.choice([-1.0, 1.0], size=len(outlier_rows))
            outlier_values = mean_val + directions * magnitude * std_val
            noisy.loc[outlier_rows, col] = outlier_values

        return noisy

    def add_gaussian_noise_continuous(
            self,
            data: pd.DataFrame,
            columns: List[str],
            noise_std_ratio: float = 0.05,
            proportion: float = 1.0,
            row_indices: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        noisy = data.copy()
        if not columns:
            return noisy

        candidate_rows = np.arange(len(noisy)) if row_indices is None else np.array(row_indices, dtype=int)
        if len(candidate_rows) == 0:
            return noisy

        n_rows = int(round(len(candidate_rows) * proportion))
        if n_rows <= 0:
            return noisy
        target_rows = self.rng.choice(candidate_rows, size=min(n_rows, len(candidate_rows)), replace=False)

        for col in columns:
            if col not in noisy.columns:
                continue
            series = pd.to_numeric(noisy[col], errors="coerce")
            col_std = float(series.std())
            if np.isnan(col_std) or col_std == 0:
                continue

            noise = self.rng.normal(loc=0.0, scale=noise_std_ratio * col_std, size=len(target_rows))
            noisy.loc[target_rows, col] = series.loc[target_rows].to_numpy(dtype=float) + noise

        return noisy

    # backward-compatible alias
    def add_perturbation_noise_continuous(
            self,
            data: pd.DataFrame,
            columns: List[str],
            noise_std: float = 0.05,
            proportion: float = 1.0
    ) -> pd.DataFrame:
        return self.add_gaussian_noise_continuous(
            data=data,
            columns=columns,
            noise_std_ratio=float(noise_std),
            proportion=proportion,
        )

    # -------------------- categorical noise --------------------

    def add_label_error_categorical(
            self,
            data: pd.DataFrame,
            columns: List[str],
            error_rate: float = 0.05,
            row_indices: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        noisy = data.copy()
        if not columns or error_rate <= 0:
            return noisy

        candidate_rows = np.arange(len(noisy)) if row_indices is None else np.array(row_indices, dtype=int)
        if len(candidate_rows) == 0:
            return noisy

        for col in columns:
            if col not in noisy.columns:
                continue
            unique_vals = [v for v in pd.unique(noisy[col].dropna())]
            if len(unique_vals) < 2:
                continue

            flip_mask = self.rng.random(len(candidate_rows)) < error_rate
            flip_rows = candidate_rows[flip_mask]

            for idx in flip_rows:
                old_val = noisy.at[idx, col]
                if pd.isna(old_val):
                    continue
                candidates = [v for v in unique_vals if v != old_val]
                if candidates:
                    noisy.at[idx, col] = self.rng.choice(candidates)
        return noisy

    # backward-compatible alias
    def add_label_encoding_noise(
            self,
            data: pd.DataFrame,
            columns: List[str],
            error_rate: float = 0.05,
            error_type: str = "random",
            confusion_matrix: Optional[Dict] = None
    ) -> pd.DataFrame:
        _ = error_type, confusion_matrix
        return self.add_label_error_categorical(data, columns, error_rate)

    def add_category_merge_split_noise(
            self,
            data: pd.DataFrame,
            columns: List[str],
            merge_rate: float = 0.2,
            split_rate: float = 0.1,
            row_indices: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        noisy = data.copy()
        if not columns:
            return noisy

        candidate_rows = np.arange(len(noisy)) if row_indices is None else np.array(row_indices, dtype=int)
        if len(candidate_rows) == 0:
            return noisy

        for col in columns:
            if col not in noisy.columns:
                continue

            col_values = noisy[col]
            valid = col_values.dropna()
            if valid.nunique() < 2:
                continue

            # merge: map infrequent categories to the most frequent one
            counts = valid.value_counts()
            anchor = counts.index[0]
            other_categories = list(counts.index[1:])

            if other_categories and merge_rate > 0:
                merge_candidates = candidate_rows[self.rng.random(len(candidate_rows)) < merge_rate]
                for idx in merge_candidates:
                    val = noisy.at[idx, col]
                    if val in other_categories:
                        noisy.at[idx, col] = anchor

            # split: split dominant category into two new pseudo-categories
            if split_rate > 0:
                dominant_rows = np.array([idx for idx in candidate_rows if noisy.at[idx, col] == anchor], dtype=int)
                if len(dominant_rows) > 0:
                    split_rows = dominant_rows[self.rng.random(len(dominant_rows)) < split_rate]
                    if len(split_rows) > 0:
                        sample_val = anchor
                        if isinstance(sample_val, (int, np.integer, float, np.floating)):
                            max_v = pd.to_numeric(valid, errors="coerce").max()
                            split_a = max_v + 1
                            split_b = max_v + 2
                        else:
                            split_a = f"{sample_val}__split_a"
                            split_b = f"{sample_val}__split_b"
                            noisy[col] = noisy[col].astype(object)

                        half = len(split_rows) // 2
                        noisy.loc[split_rows[:half], col] = split_a
                        noisy.loc[split_rows[half:], col] = split_b

        return noisy

    # backward-compatible alias (merge-only behavior)
    def add_category_merging_noise(
            self,
            data: pd.DataFrame,
            columns: List[str],
            merge_pairs: Optional[List[Tuple[int, int]]] = None,
            merge_fraction: float = 0.3
    ) -> pd.DataFrame:
        _ = merge_pairs
        return self.add_category_merge_split_noise(
            data=data,
            columns=columns,
            merge_rate=merge_fraction,
            split_rate=0.0,
        )

    def add_encoding_drift_noise(
            self,
            data: pd.DataFrame,
            columns: List[str],
            drift_rate: float = 0.1,
            row_indices: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        noisy = data.copy()
        if not columns or drift_rate <= 0:
            return noisy

        candidate_rows = np.arange(len(noisy)) if row_indices is None else np.array(row_indices, dtype=int)
        if len(candidate_rows) == 0:
            return noisy

        for col in columns:
            if col not in noisy.columns:
                continue

            unique_vals = [v for v in pd.unique(noisy[col].dropna())]
            if len(unique_vals) < 2:
                continue

            # rotate encoding map to simulate code drift
            shifted_vals = unique_vals[1:] + unique_vals[:1]
            mapping = dict(zip(unique_vals, shifted_vals))

            drift_rows = candidate_rows[self.rng.random(len(candidate_rows)) < drift_rate]
            for idx in drift_rows:
                val = noisy.at[idx, col]
                if pd.isna(val):
                    continue
                if val in mapping:
                    noisy.at[idx, col] = mapping[val]

        return noisy

    # backward-compatible alias
    def add_drift_encoding_noise(
            self,
            data: pd.DataFrame,
            columns: List[str],
            drift_type: str = "gradual",
            drift_strength: float = 0.1,
            time_column: Optional[str] = None
    ) -> pd.DataFrame:
        _ = drift_type, time_column
        return self.add_encoding_drift_noise(data, columns, drift_rate=drift_strength)

    # -------------------- sample-level + comprehensive --------------------

    def add_sample_level_noise(
            self,
            data: pd.DataFrame,
            continuous_columns: List[str],
            categorical_columns: List[str],
            row_indices: np.ndarray,
            params: Dict
    ) -> pd.DataFrame:
        noisy = data.copy()
        if len(row_indices) == 0:
            return noisy

        noisy = self.add_outlier_noise_continuous(
            noisy,
            columns=continuous_columns,
            outlier_fraction=1.0,
            magnitude=params["outlier_magnitude"],
            row_indices=row_indices,
        )
        noisy = self.add_gaussian_noise_continuous(
            noisy,
            columns=continuous_columns,
            noise_std_ratio=params["gaussian_std_ratio"],
            proportion=1.0,
            row_indices=row_indices,
        )
        noisy = self.add_label_error_categorical(
            noisy,
            columns=categorical_columns,
            error_rate=params["categorical_error_rate"],
            row_indices=row_indices,
        )
        noisy = self.add_category_merge_split_noise(
            noisy,
            columns=categorical_columns,
            merge_rate=params["merge_rate"],
            split_rate=params["split_rate"],
            row_indices=row_indices,
        )
        noisy = self.add_encoding_drift_noise(
            noisy,
            columns=categorical_columns,
            drift_rate=params["drift_rate"],
            row_indices=row_indices,
        )
        return noisy

    def add_label_noise(
            self,
            data: pd.DataFrame,
            target_col: str,
            noise_std_ratio: float = 0.2,
            row_indices: Optional[np.ndarray] = None,
            fraction: float = 1.0
    ) -> pd.DataFrame:
        noisy = data.copy()
        if target_col not in noisy.columns:
            return noisy

        candidate_rows = np.arange(len(noisy)) if row_indices is None else np.array(row_indices, dtype=int)
        if len(candidate_rows) == 0:
            return noisy

        # Only apply to a subset of candidate rows when needed.
        n_apply = int(round(len(candidate_rows) * float(np.clip(fraction, 0.0, 1.0))))
        if n_apply <= 0:
            return noisy
        apply_rows = self.rng.choice(candidate_rows, size=min(n_apply, len(candidate_rows)), replace=False)

        # Numeric target: additive Gaussian label noise.
        if pd.api.types.is_numeric_dtype(noisy[target_col]):
            y = pd.to_numeric(noisy[target_col], errors="coerce")
            std_y = float(y.std())

            if std_y == 0 or np.isnan(std_y):
                return noisy

            noise = self.rng.normal(
                loc=0.0,
                scale=noise_std_ratio * std_y,
                size=len(apply_rows)
            )

            noisy.loc[apply_rows, target_col] = y.loc[apply_rows].to_numpy() + noise
            return noisy

        # Non-numeric target: categorical flip noise.
        return self.add_label_error_categorical(
            noisy,
            columns=[target_col],
            error_rate=float(np.clip(noise_std_ratio, 0.0, 1.0)),
            row_indices=apply_rows,
        )

    def add_target_noise(
            self,
            data: pd.DataFrame,
            target_col: str,
            noise_std_ratio: float = 0.2,
            row_indices: Optional[np.ndarray] = None,
            fraction: float = 1.0
    ) -> pd.DataFrame:
        return self.add_label_noise(
            data=data,
            target_col=target_col,
            noise_std_ratio=noise_std_ratio,
            row_indices=row_indices,
            fraction=fraction,
        )

    # backward-compatible alias
    def add_y_noise(
            self,
            data: pd.DataFrame,
            target_col: str,
            noise_std_ratio: float = 0.2,
            row_indices: Optional[np.ndarray] = None,
            fraction: float = 1.0
    ) -> pd.DataFrame:
        return self.add_target_noise(data, target_col, noise_std_ratio, row_indices, fraction)

        return noisy

    def add_comprehensive_noise(
            self,
            data: Union[pd.DataFrame, np.ndarray],
            config: Dict
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Main entry for TAQS-style experiments.

        Required / useful config fields:
        - noise_strength: "small" | "medium" | "large"  (default: "medium")
        - continuous_columns / categorical_columns (optional)
        - OR num_start_idx / num_end_idx / cat_end_idx (recommended for TAQS configs)
        - target_column (optional, default protects last column)
        - protect_target: bool (default True)
        """
        df, input_is_ndarray = self._as_dataframe(data)
        noisy = df.copy()

        noise_strength = str(config.get("noise_strength", "medium")).lower()
        if noise_strength not in self.STRENGTH_TO_RATIO:
            raise ValueError("`noise_strength` must be one of: small, medium, large.")

        num_cols, cat_cols, target_col = self._resolve_columns_from_config(df, config)

        total_ratio = self.STRENGTH_TO_RATIO[noise_strength]
        params = self.STRENGTH_PARAMS[noise_strength].copy()

        # Allow explicit override from config["noise_types"].
        # This keeps compatibility with earlier config style.
        noise_types = config.get("noise_types", {})
        if "outlier" in noise_types:
            params["outlier_magnitude"] = float(noise_types["outlier"].get("magnitude", params["outlier_magnitude"]))
        if "gaussian" in noise_types:
            params["gaussian_std_ratio"] = float(
                noise_types["gaussian"].get("noise_std_ratio", params["gaussian_std_ratio"])
            )
        if "perturbation" in noise_types:
            params["gaussian_std_ratio"] = float(
                noise_types["perturbation"].get("noise_std", params["gaussian_std_ratio"])
            )
        if "label_error" in noise_types:
            params["categorical_error_rate"] = float(
                noise_types["label_error"].get("error_rate", params["categorical_error_rate"])
            )
        if "category_merge_split" in noise_types:
            params["merge_rate"] = float(noise_types["category_merge_split"].get("merge_rate", params["merge_rate"]))
            params["split_rate"] = float(noise_types["category_merge_split"].get("split_rate", params["split_rate"]))
        if "drift" in noise_types:
            params["drift_rate"] = float(noise_types["drift"].get("drift_rate", params["drift_rate"]))
        if "sample_level" in noise_types:
            params["sample_level_fraction"] = float(
                noise_types["sample_level"].get("fraction", params["sample_level_fraction"])
            )
        if "target_noise" in noise_types:
            params["target_noise_std_ratio"] = float(
                noise_types["target_noise"].get("noise_std_ratio", params["target_noise_std_ratio"])
            )
            params["target_noise_fraction"] = float(
                noise_types["target_noise"].get("fraction", params["target_noise_fraction"])
            )

        # Target noise options:
        # 1) config["target_noise"] = {"enabled": bool, "noise_std_ratio": ..., "fraction": ...}
        # 2) legacy config["label_noise_ratio"] > 0
        target_noise_cfg = config.get("target_noise", {})
        target_noise_enabled = bool(target_noise_cfg.get("enabled", False))
        legacy_label_noise_ratio = float(config.get("label_noise_ratio", 0.0))
        if legacy_label_noise_ratio > 0:
            target_noise_enabled = True
            params["target_noise_std_ratio"] = legacy_label_noise_ratio
        if "noise_std_ratio" in target_noise_cfg:
            params["target_noise_std_ratio"] = float(target_noise_cfg["noise_std_ratio"])
        if "fraction" in target_noise_cfg:
            params["target_noise_fraction"] = float(target_noise_cfg["fraction"])
        target_noise_apply_on_noisy_rows = bool(target_noise_cfg.get("only_noisy_rows", True))

        if not num_cols and not cat_cols and not (target_noise_enabled and target_col is not None):
            return self._restore_type(noisy, input_is_ndarray)

        noisy_rows = self._sample_rows(len(noisy), total_ratio)
        if len(noisy_rows) == 0:
            return self._restore_type(noisy, input_is_ndarray)

        sample_level_fraction = float(np.clip(params["sample_level_fraction"], 0.0, 1.0))
        n_sample_level = int(round(len(noisy_rows) * sample_level_fraction))
        sample_level_rows = self.rng.choice(noisy_rows, size=n_sample_level, replace=False) if n_sample_level > 0 else np.array([], dtype=int)
        sample_level_set = set(sample_level_rows.tolist())
        feature_level_rows = np.array([r for r in noisy_rows if r not in sample_level_set], dtype=int)

        if len(feature_level_rows) > 0:
            noisy = self.add_outlier_noise_continuous(
                noisy,
                columns=num_cols,
                outlier_fraction=0.5,
                magnitude=params["outlier_magnitude"],
                row_indices=feature_level_rows,
            )
            noisy = self.add_gaussian_noise_continuous(
                noisy,
                columns=num_cols,
                noise_std_ratio=params["gaussian_std_ratio"],
                proportion=0.8,
                row_indices=feature_level_rows,
            )
            noisy = self.add_label_error_categorical(
                noisy,
                columns=cat_cols,
                error_rate=params["categorical_error_rate"],
                row_indices=feature_level_rows,
            )
            noisy = self.add_category_merge_split_noise(
                noisy,
                columns=cat_cols,
                merge_rate=params["merge_rate"],
                split_rate=params["split_rate"],
                row_indices=feature_level_rows,
            )
            noisy = self.add_encoding_drift_noise(
                noisy,
                columns=cat_cols,
                drift_rate=params["drift_rate"],
                row_indices=feature_level_rows,
            )

        noisy = self.add_sample_level_noise(
            noisy,
            continuous_columns=num_cols,
            categorical_columns=cat_cols,
            row_indices=sample_level_rows,
            params=params,
        )

        if target_noise_enabled and target_col is not None:
            target_rows = noisy_rows if target_noise_apply_on_noisy_rows else None
            noisy = self.add_target_noise(
                noisy,
                target_col=target_col,
                noise_std_ratio=params["target_noise_std_ratio"],
                row_indices=target_rows,
                fraction=params["target_noise_fraction"],
            )

        return self._restore_type(noisy, input_is_ndarray)

    # -------------------- utility --------------------

    def compare_datasets(
            self,
            original: pd.DataFrame,
            noisy: pd.DataFrame,
            columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if columns is None:
            columns = list(original.columns)

        rows = []
        for col in columns:
            if col not in original.columns or col not in noisy.columns:
                continue
            o = original[col]
            n = noisy[col]

            if pd.api.types.is_numeric_dtype(o):
                diff = pd.to_numeric(n, errors="coerce") - pd.to_numeric(o, errors="coerce")
                rows.append({
                    "column": col,
                    "type": "continuous",
                    "change_rate": float((diff != 0).mean()),
                    "mean_abs_change": float(diff.abs().mean()),
                    "max_abs_change": float(diff.abs().max()),
                    "std_change": float(diff.std()),
                })
            else:
                change = (o != n)
                rows.append({
                    "column": col,
                    "type": "categorical",
                    "change_rate": float(change.mean()),
                    "unique_values_original": int(o.nunique(dropna=True)),
                    "unique_values_noisy": int(n.nunique(dropna=True)),
                })
        return pd.DataFrame(rows)

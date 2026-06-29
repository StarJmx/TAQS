import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from typing import Optional, List, Dict
from sklearn_extra.cluster import KMedoids

import random


class KCenter_Gower:
    """
    K-Center Greedy baseline with unweighted Gower distance.
    Subset size is matched to TAQS subset sizes.
    """
    def __init__(
        self,
        data_task: np.ndarray,
        data_train: np.ndarray,
        n_neighbors: int,
        num_start_idx: int,
        num_end_idx: int,
        cat_end_idx: Optional[int] = None,
        feature_weights: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
        size_excel_path: Optional[str] = None,
        sheet_name: Optional[str] = None,
    ):

        self.data_task = np.asarray(data_task)
        self.data_train = np.asarray(data_train)
        self.n_neighbors = int(n_neighbors)
        self.num_start_idx = num_start_idx
        self.num_end_idx = num_end_idx
        self.cat_end_idx = cat_end_idx
        self.random_state = random_state
        self.size_excel_path = size_excel_path
        self.sheet_name = sheet_name
        self._rng = np.random.default_rng(random_state)

        self._prepare()
        self._load_size_mapping()

    # ---------------------------------------------------------
    # Prepare
    # ---------------------------------------------------------
    def _prepare(self):

        self.X_train = self.data_train[:, :-1]
        self.quantitative_idx = list(
            range(self.num_start_idx, self.num_end_idx + 1)
        )

        self.qualitative_idx = (
            list(range(self.num_end_idx + 1, self.cat_end_idx + 1))
            if self.cat_end_idx is not None else []
        )

        self.ranges = {}
        for idx in self.quantitative_idx:
            col = self.X_train[:, idx]
            r = np.max(col) - np.min(col)
            self.ranges[idx] = r if r != 0 else 1.0

        self.distance_matrix = self.pairwise_gower_matrix()

    # ---------------------------------------------------------
    # Load TAQS subset sizes
    # ---------------------------------------------------------

    def _load_size_mapping(self):

        self.size_map = {}
        if self.size_excel_path is None or self.sheet_name is None:
            return
        df = pd.read_excel(
            self.size_excel_path,
            sheet_name=self.sheet_name
        )
        row = df[
            (df["Model"] == "TAQS") &
            (df["Split"] == "subset") &
            (df["Metric"] == "SampleSize")
        ]
        if row.empty:
            raise ValueError(
                "Cannot find TAQS SampleSize row in Excel"
            )
        row = row.iloc[0]
        for col in df.columns:
            if col.startswith("n=") and not col.endswith("_std"):
                n = int(col.split("=")[1])
                self.size_map[n] = float(row[col])

    # ---------------------------------------------------------
    # Match subset size
    # ---------------------------------------------------------
    def _get_subset_size(self, k: int) -> int:
        n_train = self.data_train.shape[0]
        if k in self.size_map:
            size = int(round(self.size_map[k]))
        else:
            size = k
        return min(size, n_train)

    # ---------------------------------------------------------
    # Pairwise Gower matrix
    # ---------------------------------------------------------

    def pairwise_gower_matrix(self) -> np.ndarray:
        n = self.X_train.shape[0]
        D = np.zeros((n, n), dtype=np.float32)
        # quantitative
        if len(self.quantitative_idx) > 0:
            X_quant = self.X_train[:, self.quantitative_idx].astype(np.float32)
            range_arr = np.array(
                [self.ranges[i] for i in self.quantitative_idx],
                dtype=np.float32
            )
            for idx in range(len(self.quantitative_idx)):
                col = X_quant[:, idx]
                diff = np.abs(col[:, None] - col[None, :])
                D += diff / range_arr[idx]

        # qualitative
        if len(self.qualitative_idx) > 0:
            X_qual = self.X_train[:, self.qualitative_idx]
            for idx in range(len(self.qualitative_idx)):
                col = X_qual[:, idx]
                neq = (
                    col[:, None] != col[None, :]
                ).astype(np.float32)
                D += neq

        return D

    # ---------------------------------------------------------
    # Core K-Center Greedy
    # ---------------------------------------------------------
    def _kcenter_subset(self, size: int) -> np.ndarray:
        n = self.X_train.shape[0]
        size = min(size, n)
        D = self.distance_matrix
        # random init
        first_idx = int(self._rng.integers(n))
        selected = [first_idx]
        # current min distance to centers
        min_dist = D[first_idx].copy()
        for _ in range(size - 1):
            next_idx = np.argmax(np.argmax(min_dist))
            selected.append(next_idx)
            # update nearest-center distance
            min_dist = np.minimum(
                min_dist,
                D[next_idx]
            )

        indices = np.sort(selected)

        return self.data_train[indices]

    # ---------------------------------------------------------
    # Public APIs
    # ---------------------------------------------------------
    def select_task_aware_subset(self) -> np.ndarray:
        size = self._get_subset_size(self.n_neighbors)
        return self._kcenter_subset(size)

    def select_subsets_for_k_list(
        self,
        k_list: List[int]
    ) -> Dict[int, np.ndarray]:
        subset_dict = {}
        for k in k_list:
            size = self._get_subset_size(k)
            subset_dict[k] = self._kcenter_subset(size)

        return subset_dict

    def select_subset(self) -> np.ndarray:
        return self.select_task_aware_subset()


class KMedoids_Gower:
    """
    K-Medoids baseline with unweighted Gower-style distance.

    The subset size is matched to TAQS subset sizes loaded from Excel,
    ensuring fair comparison under identical sample budgets.
    """

    def __init__(
        self,
        data_task: np.ndarray,
        data_train: np.ndarray,
        n_neighbors: int,
        num_start_idx: int,
        num_end_idx: int,
        cat_end_idx: Optional[int] = None,
        feature_weights: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
        size_excel_path: Optional[str] = None,
        sheet_name: Optional[str] = None,
    ):

        # keep interface compatible with TAQS
        self.data_task = np.asarray(data_task)
        self.data_train = np.asarray(data_train)
        self.n_neighbors = int(n_neighbors)
        self.num_start_idx = num_start_idx
        self.num_end_idx = num_end_idx
        self.cat_end_idx = cat_end_idx
        self.feature_weights = feature_weights
        self.random_state = random_state
        self.size_excel_path = size_excel_path
        self.sheet_name = sheet_name

        self._prepare()
        self._load_size_mapping()

    def _prepare(self):

        if self.data_task.ndim != 2 or self.data_train.ndim != 2:
            raise ValueError("data_task and data_train must be 2D arrays")

        if self.data_task.shape[1] != self.data_train.shape[1]:
            raise ValueError("column mismatch")

        if self.data_train.shape[0] == 0:
            raise ValueError("empty train set")

        # remove target column
        self.X_task = self.data_task[:, :-1]
        self.X_train = self.data_train[:, :-1]

        num_features = self.X_train.shape[1]

        self.quantitative_idx = list(
            range(self.num_start_idx, self.num_end_idx + 1)
        )
        self.qualitative_idx = (
            list(range(self.num_end_idx + 1, self.cat_end_idx + 1))
            if self.cat_end_idx is not None else []
        )
        # quantitative ranges
        self.ranges = {}
        for idx in self.quantitative_idx:
            col = self.X_train[:, idx]
            r = np.max(col) - np.min(col)
            self.ranges[idx] = r if r != 0 else 1.0

        self.distance_matrix = self.pairwise_gower_matrix()

    # ---------------------------------------------------------
    # Load TAQS subset size mapping
    # ---------------------------------------------------------
    def _load_size_mapping(self):
        self.size_map = {}
        if self.size_excel_path is None or self.sheet_name is None:
            return
        df = pd.read_excel(
            self.size_excel_path,
            sheet_name=self.sheet_name
        )
        row = df[
            (df["Model"] == "TAQS") &
            (df["Split"] == "subset") &
            (df["Metric"] == "SampleSize")
        ]
        if row.empty:
            raise ValueError(
                "Cannot find TAQS SampleSize row in Excel"
            )
        row = row.iloc[0]
        for col in df.columns:
            if col.startswith("n=") and not col.endswith("_std"):
                n = int(col.split("=")[1])
                self.size_map[n] = float(row[col])

    # ---------------------------------------------------------
    # Match TAQS subset size
    # ---------------------------------------------------------

    def _get_subset_size(self, k: int) -> int:
        n_train = self.data_train.shape[0]
        if k in self.size_map:
            size = int(round(self.size_map[k]))
        else:
            size = k

        return min(size, n_train)

    # ---------------------------------------------------------
    # Pairwise unweighted Gower distance matrix
    # ---------------------------------------------------------

    def pairwise_gower_matrix(self) -> np.ndarray:
        """Pairwise unweighted Gower distance matrix over training samples."""
        n = self.X_train.shape[0]
        D = np.zeros((n, n), dtype=np.float32)

        if len(self.quantitative_idx) > 0:
            X_quant = self.X_train[:, self.quantitative_idx].astype(np.float32)
            range_arr = np.array(
                [self.ranges[i] for i in self.quantitative_idx],
                dtype=np.float32
            )
            for idx in range(len(self.quantitative_idx)):
                col = X_quant[:, idx]
                diff = np.abs(col[:, None] - col[None, :])
                D += diff / range_arr[idx]

        if len(self.qualitative_idx) > 0:
            X_qual = self.X_train[:, self.qualitative_idx]
            for idx in range(len(self.qualitative_idx)):
                col = X_qual[:, idx]
                neq = (
                    col[:, None] != col[None, :]
                ).astype(np.float32)
                D += neq

        return D

    # ---------------------------------------------------------
    # Core K-Medoids selection
    # ---------------------------------------------------------

    def _kmedoids_subset(self, size: int) -> np.ndarray:
        size = min(size, self.X_train.shape[0])
        D = self.distance_matrix
        model = KMedoids(
            n_clusters=size,
            metric="precomputed",
            random_state=self.random_state,
        )

        model.fit(D)
        indices = np.sort(model.medoid_indices_)

        return self.data_train[indices]

    # ---------------------------------------------------------
    # Public APIs
    # ---------------------------------------------------------
    def select_task_aware_subset(self) -> np.ndarray:
        size = self._get_subset_size(self.n_neighbors)
        return self._kmedoids_subset(size)

    def select_subsets_for_k_list(
        self,
        k_list: List[int]
    ) -> Dict[int, np.ndarray]:

        subset_dict = {}
        for k in k_list:
            size = self._get_subset_size(k)
            subset_dict[k] = self._kmedoids_subset(size)
        return subset_dict

    def select_subset(self) -> np.ndarray:
        return self.select_task_aware_subset()


class TAQS_Gower:
    """
    Query-anchored subset selection using unweighted Gower distance.

    Same neighbor-union procedure as TAQS, but without task-aware feature
    weighting. Unused in current experiment scripts (ablation baseline).
    """

    def __init__(
        self,
        data_task: np.ndarray,
        data_train: np.ndarray,
        n_neighbors: int = 50,
        num_start_idx: int = 0,
        num_end_idx: Optional[int] = None,
        cat_end_idx: Optional[int] = None,
        feature_weights: Optional[np.ndarray] = None,
    ):
        self.data_task = np.asarray(data_task)
        self.data_train = np.asarray(data_train)
        self.n_neighbors = int(n_neighbors)
        self.num_start_idx = num_start_idx
        self.num_end_idx = num_end_idx
        self.cat_end_idx = cat_end_idx
        self.feature_weights = feature_weights
        if self.num_end_idx is None:
            self.num_end_idx = self.data_train.shape[1] - 2

        self._prepare()

    def _prepare(self):
        if self.data_task.ndim != 2 or self.data_train.ndim != 2:
            raise ValueError("data_task and data_train must be 2D arrays")
        if self.data_task.shape[1] != self.data_train.shape[1]:
            raise ValueError("data_task and data_train must have the same number of columns")
        if self.data_train.shape[0] == 0:
            raise ValueError("data_train must contain at least one sample")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer")

        self.X_task = self.data_task[:, :-1]
        self.X_train = self.data_train[:, :-1]

        num_features = self.X_train.shape[1]
        if self.num_start_idx < 0 or self.num_start_idx >= num_features:
            raise ValueError(f"num_start_idx={self.num_start_idx} is out of feature range")
        if self.num_end_idx < self.num_start_idx or self.num_end_idx >= num_features:
            raise ValueError("num_end_idx must satisfy num_start_idx <= num_end_idx < num_features")
        if self.cat_end_idx is not None:
            if self.cat_end_idx < self.num_end_idx + 1 or self.cat_end_idx >= num_features:
                raise ValueError("cat_end_idx must satisfy num_end_idx+1 <= cat_end_idx < num_features")

        self.quantitative_idx = list(range(self.num_start_idx, self.num_end_idx + 1))
        self.qualitative_idx = (
            list(range(self.num_end_idx + 1, self.cat_end_idx + 1))
            if self.cat_end_idx is not None else []
        )

        if self.feature_weights is None:
            self.feature_weights = np.ones(num_features, dtype=float) / num_features
        else:
            self.feature_weights = np.asarray(self.feature_weights, dtype=float)

        if len(self.feature_weights) != num_features:
            raise ValueError("feature_weights length must equal number of features")

        self.quantitative_weights = self.feature_weights[self.quantitative_idx]
        self.qualitative_weights = (
            self.feature_weights[self.qualitative_idx]
            if len(self.qualitative_idx) > 0 else np.array([])
        )

        self.ranges = {}
        for idx in self.quantitative_idx:
            col = self.X_train[:, idx]
            r = np.max(col) - np.min(col)
            self.ranges[idx] = r if r != 0 else 1.0

    def unweighted_gower_distance(self, x: np.ndarray) -> np.ndarray:
        """
        Compute standard Gower-style distances between one query sample `x`
        and all training samples (without task-aware weighting).
        """

        distances = []
        for train_sample in self.X_train:
            total_distance = 0.0
            # Quantitative variables
            for idx in self.quantitative_idx:
                diff = abs(x[idx] - train_sample[idx])
                total_distance += diff / self.ranges[idx]

            # Qualitative variables
            for idx in self.qualitative_idx:
                if x[idx] != train_sample[idx]:
                    total_distance += 1.0

            distances.append(total_distance)

        return np.array(distances)

    def select_task_aware_subset(self) -> np.ndarray:
        """
        For each task sample, select n_neighbors nearest training rows by
        unweighted Gower distance, then return the union subset.
        """
        all_neighbor_indices = set()
        k = min(self.n_neighbors, self.X_train.shape[0])

        for x in self.X_task:
            distances = self.unweighted_gower_distance(x)
            top_k_indices = np.argsort(distances)[:k]
            all_neighbor_indices.update(top_k_indices.tolist())

        if not all_neighbor_indices:
            return np.empty((0, self.data_train.shape[1]), dtype=self.data_train.dtype)

        union_indices = sorted(all_neighbor_indices)
        subset = self.data_train[union_indices]
        return subset

    def _top_k_indices_for_query(self, x: np.ndarray, k: int) -> np.ndarray:
        """
        Return sorted top-k neighbor indices for a single query sample.
        """
        distances = self.unweighted_gower_distance(x)
        if k >= len(distances):
            return np.argsort(distances)
        top_k_unsorted = np.argpartition(distances, k - 1)[:k]
        order = np.argsort(distances[top_k_unsorted])
        return top_k_unsorted[order]

    def select_subsets_for_k_list(self, k_list: List[int]) -> Dict[int, np.ndarray]:
        """
        Compute TAQS subsets for multiple k values with one distance pass per query.
        Returns a dict: k -> subset array.
        """
        if k_list is None or len(k_list) == 0:
            return {}

        unique_k = sorted(set(int(k) for k in k_list))
        if unique_k[0] <= 0:
            raise ValueError("All k values must be positive integers")

        k_max = min(unique_k[-1], self.X_train.shape[0])
        # One top-kmax extraction per query, then reuse prefixes for all smaller k.
        topk_per_query = [self._top_k_indices_for_query(x, k_max) for x in self.X_task]

        index_sets = {k: set() for k in unique_k}
        for topk in topk_per_query:
            for k in unique_k:
                k_eff = min(k, len(topk))
                index_sets[k].update(topk[:k_eff].tolist())

        subset_dict = {}
        for k in unique_k:
            if len(index_sets[k]) == 0:
                subset = np.empty((0, self.data_train.shape[1]), dtype=self.data_train.dtype)
            else:
                union_indices = sorted(index_sets[k])
                subset = self.data_train[union_indices]
            subset_dict[k] = subset

        return subset_dict

    def mean_query_topk_distance_for_k_list(self, k_list: List[int]) -> Dict[int, float]:
        """
        For each query sample in data_task, compute the mean distance of its top-k
        nearest neighbors in data_train, then average over all query samples.
        Returns a dict: k -> averaged top-k mean distance.
        """
        if k_list is None or len(k_list) == 0:
            return {}

        unique_k = sorted(set(int(k) for k in k_list))
        if unique_k[0] <= 0:
            raise ValueError("All k values must be positive integers")

        if self.X_task.shape[0] == 0:
            return {k: float("nan") for k in unique_k}

        k_max = min(unique_k[-1], self.X_train.shape[0])
        per_k_query_means = {k: [] for k in unique_k}

        for x in self.X_task:
            distances = self.unweighted_gower_distance(x)
            if k_max >= len(distances):
                topk_sorted = np.sort(distances)
            else:
                topk_unsorted = np.partition(distances, k_max - 1)[:k_max]
                topk_sorted = np.sort(topk_unsorted)

            prefix_sums = np.cumsum(topk_sorted)
            for k in unique_k:
                k_eff = min(k, len(topk_sorted))
                per_k_query_means[k].append(float(prefix_sums[k_eff - 1] / k_eff))

        return {
            k: float(np.mean(per_k_query_means[k])) if per_k_query_means[k] else float("nan")
            for k in unique_k
        }

    def select_subset(self) -> np.ndarray:
        """
        Short alias of `select_task_aware_subset`.
        """
        return self.select_task_aware_subset()


class RandomQueryAnchoredSelection:
    """
    RQAS (Random Query-Anchored Selection with TAQS-size matching).

    Randomly select samples from data_train, where subset size is determined
    by TAQS results stored in an Excel file.
    NOTE:
    These parameters are kept for compatibility with TAQS interface.
    They are not used in random selection.
    """

    def __init__(
        self,
        data_task: np.ndarray,
        data_train: np.ndarray,
        n_neighbors: int,
        num_start_idx: int,
        num_end_idx: int,
        cat_end_idx: Optional[int] = None,
        feature_weights: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
        size_excel_path: Optional[str] = None,
        sheet_name: Optional[str] = None,
    ):
        self.data_task = np.asarray(data_task)
        self.data_train = np.asarray(data_train)
        self.n_neighbors = int(n_neighbors)
        self.num_start_idx = num_start_idx
        self.num_end_idx = num_end_idx
        self.cat_end_idx = cat_end_idx
        self.feature_weights = feature_weights
        self.random_state = random_state

        self.size_excel_path = size_excel_path
        self.sheet_name = sheet_name

        self._rng = np.random.default_rng(self.random_state)

        self._prepare()
        self._load_size_mapping()

    def _prepare(self):
        if self.data_task.ndim != 2 or self.data_train.ndim != 2:
            raise ValueError("data_task and data_train must be 2D arrays")
        if self.data_task.shape[1] != self.data_train.shape[1]:
            raise ValueError("column mismatch")
        if self.data_train.shape[0] == 0:
            raise ValueError("empty train set")

        self.X_task = self.data_task[:, :-1]
        self.X_train = self.data_train[:, :-1]

    def _load_size_mapping(self):
        self.size_map = {}

        if self.size_excel_path is None or self.sheet_name is None:
            return

        df = pd.read_excel(self.size_excel_path, sheet_name=self.sheet_name)

        row = df[
            (df["Model"] == "TAQS") &
            (df["Split"] == "subset") &
            (df["Metric"] == "SampleSize")
        ]

        if row.empty:
            raise ValueError("Cannot find TAQS SampleSize row in Excel")

        row = row.iloc[0]

        for col in df.columns:
            if col.startswith("n=") and not col.endswith("_std"):
                n = int(col.split("=")[1])
                self.size_map[n] = float(row[col])

    def _get_subset_size(self, k: int) -> int:
        n_train = self.data_train.shape[0]

        if k in self.size_map:
            size = int(round(self.size_map[k]))
        else:
            size = k

        return min(size, n_train)

    def _random_subset(self, size: int) -> np.ndarray:
        indices = self._rng.choice(
            self.data_train.shape[0],
            size=size,
            replace=False
        )
        return self.data_train[np.sort(indices)]

    def select_task_aware_subset(self) -> np.ndarray:
        size = self._get_subset_size(self.n_neighbors)
        return self._random_subset(size)

    def select_subsets_for_k_list(self, k_list: List[int]) -> Dict[int, np.ndarray]:
        subset_dict = {}

        for k in k_list:
            size = self._get_subset_size(k)
            subset_dict[k] = self._random_subset(size)

        return subset_dict

    def select_subset(self) -> np.ndarray:
        return self.select_task_aware_subset()


class VanillaGowerKNNRegressor:
    """KNN regressor with standard (unweighted) Gower distance for mixed-type data."""

    def __init__(self, n_neighbors, num_start_idx, num_end_idx, cat_end_idx=None):
        """
        Parameters
        ----------
        n_neighbors : int
            Number of neighbors (k).
        num_start_idx : int
            Inclusive start index of quantitative columns.
        num_end_idx : int
            Inclusive end index of quantitative columns.
        cat_end_idx : int, optional
            Inclusive end index of categorical columns. None means no categorical
            features; categorical columns are assumed to follow quantitative ones.

        Notes
        -----
        Quantitative features must appear before categorical features.
        Gower distance uses normalized absolute difference for numeric features
        and 0/1 mismatch for categorical features, averaged over active features.
        """

        self.n_neighbors = n_neighbors
        self.num_start_idx = num_start_idx
        self.num_end_idx = num_end_idx
        self.cat_end_idx = cat_end_idx

    def fit(self, X, y):
        """
        Store training data and precompute Gower normalization statistics.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training features.
        y : np.ndarray, shape (n_samples,)
            Training targets.
        """

        self.X_train = X
        self.y_train = y

        num_features = X.shape[1]

        if self.num_start_idx < 0 or self.num_start_idx >= num_features:
            raise ValueError(f"num_start_idx={self.num_start_idx} is out of feature range")

        if self.num_end_idx < self.num_start_idx or self.num_end_idx >= num_features:
            raise ValueError("num_end_idx must satisfy num_start_idx <= num_end_idx < n_features")

        if self.cat_end_idx is not None:
            if self.cat_end_idx < self.num_end_idx + 1 or self.cat_end_idx >= num_features:
                raise ValueError("cat_end_idx must satisfy num_end_idx + 1 <= cat_end_idx < n_features")

        self.quantitative_idx = list(range(self.num_start_idx, self.num_end_idx + 1))
        self.qualitative_idx = (
            list(range(self.num_end_idx + 1, self.cat_end_idx + 1))
            if self.cat_end_idx is not None else []
        )

        self.ranges = {}
        for idx in self.quantitative_idx:
            col = self.X_train[:, idx]
            R = np.max(col) - np.min(col)
            self.ranges[idx] = R if R != 0 else 1.0

        self.p = len(self.quantitative_idx) + len(self.qualitative_idx)

        return self

    def _compute_distances(self, x):
        """Gower distances from one query row to all training rows."""

        distances = []

        for j in range(self.X_train.shape[0]):
            train_sample = self.X_train[j]
            total_distance = 0.0

            for idx in self.quantitative_idx:
                diff = abs(x[idx] - train_sample[idx])
                total_distance += diff / self.ranges[idx]

            for idx in self.qualitative_idx:
                if x[idx] != train_sample[idx]:
                    total_distance += 1.0

            if self.p > 0:
                total_distance /= self.p

            distances.append(total_distance)

        return np.array(distances)

    def predict(self, X):
        """Predict target values for query rows."""

        predictions = []

        for x in X:
            distances = self._compute_distances(x)
            neighbor_idx = np.argsort(distances)[:self.n_neighbors]
            y_pred = np.mean(self.y_train[neighbor_idx])
            predictions.append(y_pred)

        return np.array(predictions)


class TaskAwareWeightedGowerKNNRegressor:
    """KNN regressor with task-aware weighted Gower distance."""

    def __init__(self, n_neighbors, num_start_idx, num_end_idx, cat_end_idx=None, feature_weights=None):
        self.n_neighbors = n_neighbors
        self.feature_weights = feature_weights
        self.num_start_idx = num_start_idx
        self.num_end_idx = num_end_idx
        self.cat_end_idx = cat_end_idx

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.feature_weights = np.array(self.feature_weights, dtype=float)
        num_features = X.shape[1]
        if len(self.feature_weights) != num_features:
            raise ValueError("feature_weights length must equal number of features")
        if self.num_start_idx < 0 or self.num_start_idx >= num_features:
            raise ValueError(f"num_start_idx={self.num_start_idx} is out of feature range")
        if self.num_end_idx < self.num_start_idx or self.num_end_idx >= num_features:
            raise ValueError("num_end_idx must satisfy num_start_idx <= num_end_idx < n_features")
        if self.cat_end_idx is not None:
            if self.cat_end_idx < self.num_end_idx + 1 or self.cat_end_idx >= num_features:
                raise ValueError("cat_end_idx must satisfy num_end_idx+1 <= cat_end_idx < n_features")
        self.quantitative_idx = list(range(self.num_start_idx, self.num_end_idx + 1))
        self.qualitative_idx = list(range(self.num_end_idx + 1, self.cat_end_idx + 1)) if self.cat_end_idx is not None else []
        self.quantitative_weights = self.feature_weights[self.quantitative_idx]
        if len(self.qualitative_idx) > 0:
            self.qualitative_weights = self.feature_weights[self.qualitative_idx]
        else:
            self.qualitative_weights = np.array([])
        self.weight_sum = np.sum(self.feature_weights)
        self.ranges = {}
        for idx in self.quantitative_idx:
            col = self.X_train[:, idx]
            R = np.max(col) - np.min(col)
            self.ranges[idx] = R if R != 0 else 1.0

        return self

    def _compute_distances(self, x):
        distances = []
        for train_sample in self.X_train:
            total_distance = 0.0
            for idx, weight in zip(self.quantitative_idx, self.quantitative_weights):
                diff = abs(x[idx] - train_sample[idx])
                total_distance += weight * (diff / self.ranges[idx])
            for idx, weight in zip(self.qualitative_idx, self.qualitative_weights):
                if x[idx] != train_sample[idx]:
                    total_distance += weight
            distances.append(total_distance)
        return np.array(distances)

    def predict(self, X):
        predictions = []
        for x in X:
            distances = self._compute_distances(x)
            neighbor_idx = np.argsort(distances)[:self.n_neighbors]
            y_pred = np.mean(self.y_train[neighbor_idx])
            predictions.append(y_pred)

        return np.array(predictions)


class ShuffleTaskAwareWeightedGowerKNNRegressor:
    """Task-aware weighted Gower KNN with feature weights shuffled at fit time."""

    def __init__(self, n_neighbors, num_start_idx, num_end_idx, cat_end_idx=None, feature_weights=None):
        self.n_neighbors = n_neighbors
        self.feature_weights = feature_weights
        self.num_start_idx = num_start_idx
        self.num_end_idx = num_end_idx
        self.cat_end_idx = cat_end_idx
        self.rng = random.Random(42)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        if self.feature_weights is not None:
            weights = list(self.feature_weights)
            self.rng.shuffle(weights)
            self.feature_weights = weights
        self.feature_weights = np.array(self.feature_weights, dtype=float)
        num_features = X.shape[1]
        if len(self.feature_weights) != num_features:
            raise ValueError("feature_weights length must equal number of features")
        if self.num_start_idx < 0 or self.num_start_idx >= num_features:
            raise ValueError(f"num_start_idx={self.num_start_idx} is out of feature range")
        if self.num_end_idx < self.num_start_idx or self.num_end_idx >= num_features:
            raise ValueError("num_end_idx must satisfy num_start_idx <= num_end_idx < n_features")
        if self.cat_end_idx is not None:
            if self.cat_end_idx < self.num_end_idx + 1 or self.cat_end_idx >= num_features:
                raise ValueError("cat_end_idx must satisfy num_end_idx+1 <= cat_end_idx < n_features")
        self.quantitative_idx = list(range(self.num_start_idx, self.num_end_idx + 1))
        self.qualitative_idx = list(range(self.num_end_idx + 1, self.cat_end_idx + 1)) if self.cat_end_idx is not None else []
        self.quantitative_weights = self.feature_weights[self.quantitative_idx]
        if len(self.qualitative_idx) > 0:
            self.qualitative_weights = self.feature_weights[self.qualitative_idx]
        else:
            self.qualitative_weights = np.array([])
        self.weight_sum = np.sum(self.feature_weights)
        self.ranges = {}
        for idx in self.quantitative_idx:
            col = self.X_train[:, idx]
            R = np.max(col) - np.min(col)
            self.ranges[idx] = R if R != 0 else 1.0

        return self

    def _compute_distances(self, x):
        distances = []
        for train_sample in self.X_train:
            total_distance = 0.0
            for idx, weight in zip(self.quantitative_idx, self.quantitative_weights):
                diff = abs(x[idx] - train_sample[idx])
                total_distance += weight * (diff / self.ranges[idx])
            for idx, weight in zip(self.qualitative_idx, self.qualitative_weights):
                if x[idx] != train_sample[idx]:
                    total_distance += weight
            distances.append(total_distance)
        return np.array(distances)

    def predict(self, X):
        predictions = []
        for x in X:
            distances = self._compute_distances(x)
            neighbor_idx = np.argsort(distances)[:self.n_neighbors]
            y_pred = np.mean(self.y_train[neighbor_idx])
            predictions.append(y_pred)

        return np.array(predictions)


class DAFIGowerKNNRegressor:
    """
    KNN regressor with DAFI-Gower distance (IQR normalization and NMI-based weights).

    Uses interquartile range instead of min-max range to reduce sensitivity to outliers.
    """

    def __init__(self, n_neighbors, num_start_idx, num_end_idx, cat_end_idx=None,
                 sample_size_for_alpha=5000):
        """
        Parameters
        ----------
        n_neighbors : int
            Number of neighbors (k).
        num_start_idx : int
            Inclusive start index of quantitative columns.
        num_end_idx : int
            Inclusive end index of quantitative columns.
        cat_end_idx : int or None
            Inclusive end index of categorical columns, or None if absent.
        sample_size_for_alpha : int
            Max training rows subsampled when estimating mean pairwise distances.
        """
        self.n_neighbors = n_neighbors
        self.num_start_idx = num_start_idx
        self.num_end_idx = num_end_idx
        self.cat_end_idx = cat_end_idx
        self.sample_size_for_alpha = sample_size_for_alpha

    def fit(self, X, y):
        """Fit the model and compute IQR, alpha scaling, and NMI-based feature weights."""
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        n_samples, n_features = self.X_train.shape

        self.quantitative_idx = list(range(self.num_start_idx, self.num_end_idx + 1))
        if self.cat_end_idx is not None:
            self.qualitative_idx = list(range(self.num_end_idx + 1, self.cat_end_idx + 1))
        else:
            self.qualitative_idx = []

        self.iqr = {}
        for idx in self.quantitative_idx:
            col = self.X_train[:, idx]
            q75, q25 = np.percentile(col, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                iqr = 1.0
            self.iqr[idx] = iqr

        if n_samples > self.sample_size_for_alpha:
            np.random.seed(42)
            indices = np.random.choice(n_samples, self.sample_size_for_alpha, replace=False)
            sub_X = self.X_train[indices]
        else:
            sub_X = self.X_train

        cont_dists = []
        for idx in self.quantitative_idx:
            col = sub_X[:, idx]
            iqr = self.iqr[idx]
            diffs = np.abs(np.subtract.outer(col, col))
            cont_dists.append(diffs / iqr)

        cat_dists = []
        for idx in self.qualitative_idx:
            col = sub_X[:, idx]
            diff_mat = (col[:, np.newaxis] != col).astype(float)
            cat_dists.append(diff_mat)

        n_sub = sub_X.shape[0]
        total_pairs = n_sub * n_sub
        cont_avg = []
        for d in cont_dists:
            avg = np.sum(d) / total_pairs
            cont_avg.append(avg)
        cat_avg = []
        for d in cat_dists:
            avg = np.sum(d) / total_pairs
            cat_avg.append(avg)

        self.cont_mean_dist = np.mean(cont_avg) if cont_avg else 0.0

        self.alphas = np.ones(n_features)
        for i, idx in enumerate(self.qualitative_idx):
            if cat_avg[i] > 0:
                self.alphas[idx] = self.cont_mean_dist / cat_avg[i]
            else:
                self.alphas[idx] = 1.0

        X_discrete = self._discretize_features(self.X_train)

        n_features = X_discrete.shape[1]
        nmi_scores = np.zeros(n_features)
        for i in range(n_features):
            total_nmi = 0.0
            for j in range(n_features):
                if i == j:
                    continue
                nmi = self._normalized_mutual_info(X_discrete[:, i], X_discrete[:, j])
                total_nmi += nmi
            nmi_scores[i] = total_nmi / (n_features - 1)

        if np.sum(nmi_scores) > 0:
            self.weights = nmi_scores / np.sum(nmi_scores)
        else:
            self.weights = np.ones(n_features) / n_features

        self.alpha_weight = self.alphas * self.weights
        self.p = n_features

        return self

    def _discretize_features(self, X):
        """Discretize quantitative features by quartiles; encode categorical features as integers."""
        X = np.asarray(X)
        n_samples, n_features = X.shape
        X_disc = np.zeros_like(X, dtype=int)

        for idx in self.quantitative_idx:
            col = X[:, idx]
            q25, q50, q75 = np.percentile(col, [25, 50, 75])
            bins = [-np.inf, q25, q50, q75, np.inf]
            X_disc[:, idx] = np.digitize(col, bins)
        for idx in self.qualitative_idx:
            col = X[:, idx]
            unique_vals = np.unique(col)
            mapping = {v: i for i, v in enumerate(unique_vals)}
            X_disc[:, idx] = np.array([mapping[v] for v in col])
        return X_disc

    def _normalized_mutual_info(self, x, y):
        """Normalized mutual information: I(x;y) / max(H(x), H(y))."""
        mi = mutual_info_score(x, y)
        def entropy(arr):
            _, counts = np.unique(arr, return_counts=True)
            probs = counts / len(arr)
            return -np.sum(probs * np.log(probs))
        hx = entropy(x)
        hy = entropy(y)
        denom = max(hx, hy)
        if denom == 0:
            return 0.0
        return mi / denom

    def _compute_distances(self, x):
        """DAFI-Gower distances from one query row to all training rows."""
        distances = np.zeros(len(self.X_train))
        for i, train_sample in enumerate(self.X_train):
            total = 0.0
            for idx in self.quantitative_idx:
                diff = abs(x[idx] - train_sample[idx])
                d = diff / self.iqr[idx]
                total += self.alpha_weight[idx] * d
            for idx in self.qualitative_idx:
                if x[idx] != train_sample[idx]:
                    total += self.alpha_weight[idx]
            distances[i] = total / self.p
        return distances

    def predict(self, X):
        """Predict target values for query rows."""
        X = np.asarray(X)
        predictions = []
        for x in X:
            dists = self._compute_distances(x)
            neighbor_idx = np.argsort(dists)[:self.n_neighbors]
            y_pred = np.mean(self.y_train[neighbor_idx])
            predictions.append(y_pred)
        return np.array(predictions)


class EntropyWeightedGowerKNNRegressor:
    """
    KNN regressor with entropy-weighted Gower distance.

    Numeric features are auto-binned before entropy estimation; categorical
    features use raw category entropy.
    """

    def __init__(self, n_neighbors: int, num_start_idx: int, num_end_idx: int, cat_end_idx: Optional[int] = None):
        self.n_neighbors = n_neighbors
        self.num_start_idx = num_start_idx
        self.num_end_idx = num_end_idx
        self.cat_end_idx = cat_end_idx

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model, estimate entropy-based feature weights, and store training data."""
        self.X_train = X
        self.y_train = y
        n_samples, n_features = X.shape

        if self.num_start_idx < 0 or self.num_start_idx >= n_features:
            raise ValueError(f"num_start_idx={self.num_start_idx} is out of feature range")
        if self.num_end_idx < self.num_start_idx or self.num_end_idx >= n_features:
            raise ValueError("num_end_idx must satisfy num_start_idx <= num_end_idx < n_features")
        if self.cat_end_idx is not None:
            if self.cat_end_idx < self.num_end_idx + 1 or self.cat_end_idx >= n_features:
                raise ValueError("cat_end_idx must satisfy num_end_idx+1 <= cat_end_idx < n_features")

        self.quantitative_idx = list(range(self.num_start_idx, self.num_end_idx + 1))
        self.qualitative_idx = list(range(self.num_end_idx + 1, self.cat_end_idx + 1)) if self.cat_end_idx is not None else []

        feature_entropies = np.zeros(n_features)

        for idx in self.quantitative_idx:
            col = X[:, idx]
            labels = self._auto_categorize(col)
            feature_entropies[idx] = self._modified_entropy(labels)

        for idx in self.qualitative_idx:
            col = X[:, idx]
            feature_entropies[idx] = self._modified_entropy(col)

        total_entropy = np.sum(feature_entropies)
        if total_entropy > 0:
            self.feature_weights = feature_entropies / total_entropy
        else:
            self.feature_weights = np.ones(n_features) / n_features

        self.quantitative_weights = self.feature_weights[self.quantitative_idx]
        if len(self.qualitative_idx) > 0:
            self.qualitative_weights = self.feature_weights[self.qualitative_idx]
        else:
            self.qualitative_weights = np.array([])

        self.ranges = {}
        for idx in self.quantitative_idx:
            col = self.X_train[:, idx]
            R = np.max(col) - np.min(col)
            self.ranges[idx] = R if R != 0 else 1.0

        return self

    def _auto_categorize(self, col: np.ndarray) -> np.ndarray:
        """
        Auto-bin a 1D numeric column into cluster labels.

        Uses Calinski-Harabasz scores over k=2..q_max and picks the first local
        peak; if none exists, uses the k with the highest score.
        """
        n_samples = len(col)
        n_unique = len(np.unique(col))
        q_max = min(20, n_samples // 5, n_unique)
        if q_max < 2:
            return np.zeros(n_samples, dtype=int)

        scores = []
        for q in range(2, q_max + 1):
            kmeans = KMeans(n_clusters=q, random_state=0, n_init=10)
            labels = kmeans.fit_predict(col.reshape(-1, 1))
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) < q or np.min(counts) < 1:
                scores.append(-np.inf)
                continue
            score = calinski_harabasz_score(col.reshape(-1, 1), labels)
            scores.append(score)

        if all(np.isneginf(scores)):
            return np.zeros(n_samples, dtype=int)
        best_q = 2
        for i in range(1, len(scores) - 1):
            if scores[i] > scores[i-1] and scores[i] > scores[i+1]:
                best_q = i + 2
                break
        else:
            best_q = np.argmax(scores) + 2

        kmeans = KMeans(n_clusters=best_q, random_state=0, n_init=10)
        labels = kmeans.fit_predict(col.reshape(-1, 1))
        return labels

    def _modified_entropy(self, labels: np.ndarray) -> float:
        """Modified entropy: H / log2(h), where h is the number of categories."""
        unique, counts = np.unique(labels, return_counts=True)
        h = len(unique)
        if h == 0:
            return 0.0
        probs = counts / len(labels)
        entropy = -np.sum(probs * np.log2(probs))
        modified_entropy = entropy / np.log2(h)
        return modified_entropy

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """Entropy-weighted Gower distances from one query row to all training rows."""
        distances = []
        for train_sample in self.X_train:
            total_distance = 0.0
            for idx, weight in zip(self.quantitative_idx, self.quantitative_weights):
                diff = abs(x[idx] - train_sample[idx])
                total_distance += weight * (diff / self.ranges[idx])
            for idx, weight in zip(self.qualitative_idx, self.qualitative_weights):
                if x[idx] != train_sample[idx]:
                    total_distance += weight
            distances.append(total_distance)
        return np.array(distances)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for query rows."""
        predictions = []
        for x in X:
            distances = self._compute_distances(x)
            neighbor_idx = np.argsort(distances)[:self.n_neighbors]
            y_pred = np.mean(self.y_train[neighbor_idx])
            predictions.append(y_pred)
        return np.array(predictions)

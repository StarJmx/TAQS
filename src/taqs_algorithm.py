import numpy as np
from typing import Optional, List, Dict


class TaskAwareQueryAnchoredSelection:
    """
    TAQS (Task-Aware Query-anchored Selection).

    Use Task-Aware Weighted Gower Distance (TAWG) to find nearest neighbors in
    `data_train` for each sample in `data_task`, then return the union subset.
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
        # Validate shapes, column counts, non-empty train set, and positive n_neighbors.
        if self.data_task.ndim != 2 or self.data_train.ndim != 2:
            raise ValueError("data_task and data_train must be 2D arrays")
        if self.data_task.shape[1] != self.data_train.shape[1]:
            raise ValueError("data_task and data_train must have the same number of columns")
        if self.data_train.shape[0] == 0:
            raise ValueError("data_train must contain at least one sample")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer")

        # Last column is the target y; distances use feature columns only.
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

        # Split feature indices into quantitative and qualitative (if any).
        self.quantitative_idx = list(range(self.num_start_idx, self.num_end_idx + 1))
        self.qualitative_idx = (
            list(range(self.num_end_idx + 1, self.cat_end_idx + 1))
            if self.cat_end_idx is not None else []
        )

        # Default to uniform weights when none are provided.
        if self.feature_weights is None:
            self.feature_weights = np.ones(num_features, dtype=float) / num_features
        else:
            self.feature_weights = np.asarray(self.feature_weights, dtype=float)

        if len(self.feature_weights) != num_features:
            raise ValueError("feature_weights length must equal number of features")

        # Slice weights by variable type for block-wise distance computation.
        self.quantitative_weights = self.feature_weights[self.quantitative_idx]
        self.qualitative_weights = (
            self.feature_weights[self.qualitative_idx]
            if len(self.qualitative_idx) > 0 else np.array([])
        )

        # Per-feature range for Gower normalization; use 1.0 when range is zero.
        self.ranges = {}
        for idx in self.quantitative_idx:
            col = self.X_train[:, idx]
            r = np.max(col) - np.min(col)
            self.ranges[idx] = r if r != 0 else 1.0

    def task_aware_weighted_gower_distance(self, x: np.ndarray) -> np.ndarray:
        """
        Compute TAWG distances between query sample ``x`` and all training samples.

        Quantitative features contribute weight * |diff| / range; qualitative
        mismatches contribute their feature weight.
        """
        distances = np.zeros(len(self.X_train))

        if len(self.quantitative_idx) > 0:
            X_quant = self.X_train[:, self.quantitative_idx]
            x_quant = x[self.quantitative_idx]
            diff_quant = np.abs(X_quant - x_quant)

            range_arr = np.array([self.ranges[i] for i in self.quantitative_idx])
            weights_quant = np.asarray(self.quantitative_weights)
            distances += np.sum(weights_quant * diff_quant / range_arr, axis=1)

        if len(self.qualitative_idx) > 0:
            X_qual = self.X_train[:, self.qualitative_idx]
            x_qual = x[self.qualitative_idx]
            neq_qual = (X_qual != x_qual).astype(np.float64)

            weights_qual = np.asarray(self.qualitative_weights)
            distances += np.sum(weights_qual * neq_qual, axis=1)

        return distances

    def select_task_aware_subset(self) -> np.ndarray:
        """
        For each sample in data_task, select n_neighbors nearest samples in
        data_train by TAWG distance, then return the union subset.
        """
        all_neighbor_indices = set()
        k = min(self.n_neighbors, self.X_train.shape[0])

        for x in self.X_task:
            distances = self.task_aware_weighted_gower_distance(x)
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
        distances = self.task_aware_weighted_gower_distance(x)
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
            distances = self.task_aware_weighted_gower_distance(x)
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

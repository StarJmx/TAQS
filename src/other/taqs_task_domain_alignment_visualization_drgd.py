# ============================================================
# TAQS Task-domain Alignment Visualization (UMAP)
# Dataset: DRGD
# ============================================================
# Functionality:
# 1. First 40 samples as task/query set
# 2. Remaining samples as candidate/full set
# 3. Use TAQS(k=7) to select TAQS subset
# 4. Build random baseline (sample size matches TAQS subset)
# 5. Fit a single UMAP on the full dataset
# 6. Visualization:
#    - Gray: Full dataset
#    - Red: Task set
#    - Blue: TAQS subset
#    - Yellow: Random subset
# 7. Export Excel:
#    - TAQS subset
#    - Random subset
#    - Task set
#    - Task neighbors mapping
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap.umap_ as umap

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src import data_processor as dp
from src import taqs_algorithm as taqs

# ============================================================
# Config
# ============================================================

DATASET_NAME = "DRGD"
K_NEIGHBORS = 7
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

OUTPUT_DIR = "../../results/umap_task_alignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXCEL_PATH = os.path.join(
    OUTPUT_DIR,
    "_DRGD_UMAP_task_alignment.xlsx"
)

FIG_PATH = os.path.join(
    OUTPUT_DIR,
    "_DRGD_UMAP_visualization.svg"
)


def _to_optional_int(value):
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("none", "null", ""):
            return None
        return int(v)
    return int(value)


def task_aware_weighted_gower_distance(x, y):

    total_distance = 0.0

    # quantitative branch
    for local_i, idx in enumerate(quantitative_idx):

        diff = abs(float(x[idx]) - float(y[idx]))

        total_distance += (
            quantitative_weights[local_i]
            * (diff / ranges[local_i])
        )

    # qualitative branch
    for local_i, idx in enumerate(qualitative_idx):

        if x[idx] != y[idx]:

            total_distance += qualitative_weights[local_i]

    return total_distance


# ============================================================
# Load dataset
# ============================================================

config = dp.load_dataset_config(
    "../../configs/datasets_config.yaml",
    DATASET_NAME
)

processor = dp.DataProcessor()
columns, data = processor.load_data("../" + config["file_path"])

# Keep original DataFrame for export
df_all = pd.DataFrame(data, columns=columns)

# ============================================================
# Split task set & candidate set
# ============================================================

task_indices = np.arange(40)

candidate_indices = np.arange(40, len(data))

task_data = data[task_indices]
candidate_data = data[candidate_indices]

# ============================================================
# Standardize numerical features
# ============================================================

num_slice = slice(
    config["num_start_idx"],
    config["num_end_idx"] + 1
)

candidate_x = candidate_data[:, :-1].copy()
task_x = task_data[:, :-1].copy()

scaler = StandardScaler()

scaler.fit(candidate_x[:, num_slice].astype(float))

candidate_x[:, num_slice] = scaler.transform(
    candidate_x[:, num_slice].astype(float)
)

task_x[:, num_slice] = scaler.transform(
    task_x[:, num_slice].astype(float)
)

candidate_data_std = np.hstack([
    candidate_x,
    candidate_data[:, -1].reshape(-1, 1)
])

task_data_std = np.hstack([
    task_x,
    task_data[:, -1].reshape(-1, 1)
])

# ============================================================
# Run TAQS
# ============================================================

selector = taqs.TaskAwareQueryAnchoredSelection(
    data_task=task_data_std,
    data_train=candidate_data_std,
    n_neighbors=K_NEIGHBORS,
    num_start_idx=config["num_start_idx"],
    num_end_idx=config["num_end_idx"],
    cat_end_idx=_to_optional_int(config["cat_end_idx"]),
    feature_weights=config["feature_weights"],
)

subset_dict = selector.select_subsets_for_k_list([K_NEIGHBORS])

taqs_subset = subset_dict[K_NEIGHBORS]

# ============================================================
# Recover TAQS subset indices
# ============================================================

candidate_df = pd.DataFrame(candidate_data_std)

taqs_indices_local = []

for row in taqs_subset:
    matches = np.where(
        np.all(candidate_data_std == row, axis=1)
    )[0]

    taqs_indices_local.extend(matches.tolist())

taqs_indices_local = sorted(list(set(taqs_indices_local)))

taqs_indices_global = candidate_indices[taqs_indices_local]

# ============================================================
# Random baseline subset
# ============================================================

random_subset_size = len(taqs_indices_global)

random_indices_global = np.random.choice(
    candidate_indices,
    size=random_subset_size,
    replace=False
)

random_indices_global = sorted(random_indices_global.tolist())

# ============================================================
# Build neighbor mapping
# ============================================================
"""
# Neighbor indices for each task sample (7 nearest neighbors)
neighbor_mapping_rows = []

# selector.neighbor_indices
# shape: [n_task, k]

neighbor_indices_local = selector.neighbor_indices

for task_i in range(len(task_indices)):

    task_global_idx = task_indices[task_i]

    neighbors_local = neighbor_indices_local[task_i]

    neighbors_global = candidate_indices[neighbors_local]

    row = {
        "task_sample_index": int(task_global_idx)
    }

    for j, n_idx in enumerate(neighbors_global):
        row[f"neighbor_{j+1}"] = int(n_idx)

    neighbor_mapping_rows.append(row)

neighbor_df = pd.DataFrame(neighbor_mapping_rows)
"""
# ============================================================
# Build unified embedding features
# ============================================================

full_x = data[:, :-1]

num_cols = list(
    range(
        config["num_start_idx"],
        config["num_end_idx"] + 1
    )
)

cat_start = config["num_end_idx"] + 1
cat_end = _to_optional_int(config["cat_end_idx"])

if cat_end is not None and cat_end >= cat_start:
    cat_cols = list(range(cat_start, cat_end + 1))
else:
    cat_cols = []

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

X_processed = preprocessor.fit_transform(full_x)


# ===== TAWG parameters =====

quantitative_idx = list(
    range(config["num_start_idx"], config["num_end_idx"] + 1)
)

cat_end_idx = _to_optional_int(config["cat_end_idx"])

qualitative_idx = (
    list(range(config["num_end_idx"] + 1, cat_end_idx + 1))
    if cat_end_idx is not None and cat_end_idx >= config["num_end_idx"] + 1
    else []
)

feature_weights = np.array(config["feature_weights"])

# Continuous variable weights
quantitative_weights = feature_weights[quantitative_idx]

# Categorical variable weights
qualitative_weights = feature_weights[qualitative_idx]

# Continuous variable ranges (avoid division by zero)
ranges = np.ptp(X_processed[:, quantitative_idx].astype(float), axis=0)
ranges[ranges == 0] = 1e-8

# ============================================================
# UMAP embedding
# ============================================================

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric=task_aware_weighted_gower_distance,
    random_state=RANDOM_SEED
)

embedding = reducer.fit_transform(X_processed)

# ============================================================
# Visualization
# ============================================================

def set_paper_style():
    """Simplified version of utils.set_paper_style"""
    plt.style.use("default")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


set_paper_style()

plt.figure(figsize=(3, 3.5))
dpi = 600
fontsize = 12

# Full dataset
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    marker="o",
    s=4,
    c="lightgray",
    alpha=0.6,
    label="Full"
)

# Random subset
plt.scatter(
    embedding[random_indices_global, 0],
    embedding[random_indices_global, 1],
    s=6,
    c="gold",
    alpha=0.9,
    label="Random"
)

# TAQS subset
plt.scatter(
    embedding[taqs_indices_global, 0],
    embedding[taqs_indices_global, 1],
    s=6,
    c="#1f77b4",
    alpha=0.8,
    label="TAQS"
)

# Task set
plt.scatter(
    embedding[task_indices, 0],
    embedding[task_indices, 1],
    s=10,
    marker="*",
    c="#d62728",
    # edgecolors="black",
    # linewidths=0.5,
    alpha=1.0,
    label="Task"
)

plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")

ax = plt.gca()
ax.set_xlim(left=0)

plt.legend(loc="best")
plt.tight_layout()

plt.savefig(
    FIG_PATH,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print(f"[Saved Figure] {FIG_PATH}")

# ============================================================
# Export Excel
# ============================================================

with pd.ExcelWriter(EXCEL_PATH) as writer:

    # TAQS subset
    df_all.iloc[taqs_indices_global].to_excel(
        writer,
        sheet_name="TAQS_subset",
        index=True
    )

    # Random subset
    df_all.iloc[random_indices_global].to_excel(
        writer,
        sheet_name="Random_subset",
        index=True
    )

    # Task set
    df_all.iloc[task_indices].to_excel(
        writer,
        sheet_name="Task_set",
        index=True
    )

    # Neighbor mapping
"""    neighbor_df.to_excel(
        writer,
        sheet_name="Task_neighbors",
        index=False
    )"""

print(f"[Saved Excel] {EXCEL_PATH}")

print("=== DONE ===")
import pandas as pd
import numpy as np
import os
from scipy.stats import ks_2samp


# ==========================
# File paths
# ==========================

# Locate project root based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

taqs_file = os.path.join(PROJECT_ROOT, "results", "taqs_subset_PAIN.csv")

train_file = os.path.join(PROJECT_ROOT, "data", "processed", "Application_case_train_P011_P200.csv")

task_file = os.path.join(PROJECT_ROOT, "data", "processed", "Application_case_task_P001_P010.csv")

# ==========================
# Target variable name
# Modify according to your data
# ==========================

target_column = "pain_scale"


# ==========================
# Read data
# ==========================

taqs_df = pd.read_csv(taqs_file)

train_df = pd.read_csv(train_file)

task_df = pd.read_csv(task_file)


print("TAQS subset shape:", taqs_df.shape)
print("Original train shape:", train_df.shape)
print("Task shape:", task_df.shape)


# ==========================
# Automatically get feature columns
# Exclude the target variable
# ==========================

features = [
    col for col in task_df.columns
    if col != target_column
]


print("\nFeatures:")
print(features)


# ==========================
# KS calculation function
# ==========================

def calculate_ks(data1, data2, feature):

    x1 = data1[feature].dropna()
    x2 = data2[feature].dropna()

    # KS test
    ks_stat, p_value = ks_2samp(
        x1,
        x2
    )

    # Round p_value to 6 decimal places
    return ks_stat, round(p_value, 6)



# ==========================
# TAQS vs Task
# ==========================

taqs_results = []


for feature in features:

    ks, p = calculate_ks(
        taqs_df,
        task_df,
        feature
    )

    taqs_results.append(
        {
            "Feature": feature,
            "KS_statistic": ks,
            "p_value": p
        }
    )


taqs_results = pd.DataFrame(taqs_results)



# ==========================
# Train vs Task
# baseline
# ==========================

train_results = []


for feature in features:

    ks, p = calculate_ks(
        train_df,
        task_df,
        feature
    )

    train_results.append(
        {
            "Feature": feature,
            "KS_statistic": ks,
            "p_value": p
        }
    )


train_results = pd.DataFrame(train_results)



# ==========================
# Print results
# ==========================


print("\n==============================")
print("TAQS subset vs Task KS Test")
print("==============================")

print(
    taqs_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)



print("\n==============================")
print("Original Train vs Task KS Test")
print("==============================")

print(
    train_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)



# ==========================
# Summary metrics
# ==========================

print("\n==============================")
print("Average KS distance")
print("==============================")


print(
    "TAQS mean KS:",
    taqs_results["KS_statistic"].mean()
)


print(
    "Train mean KS:",
    train_results["KS_statistic"].mean()
)



# ==========================
# Save results
# ==========================

"""taqs_results.to_csv(
    "TAQS_vs_Task_KS_results.csv",
    index=False
)


train_results.to_csv(
    "Train_vs_Task_KS_results.csv",
    index=False
)


print("\nResults saved.")"""
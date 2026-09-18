# Task-Aware Query-Anchored Selection (TAQS)

## Overview

This repository contains the implementation of **Task-Aware Query-Anchored Selection (TAQS)**, a novel data selection framework that leverages task-aware weighted distance metrics to identify the most relevant training samples for target tasks. TAQS improves data efficiency and model performance by strategically selecting subsets that maximize task alignment while minimizing distributional shift.

## Key Features

- **Task-Aware Weighted Gower Distance (TAWG)**: A customized distance metric that incorporates feature importance weights for mixed-type data (numerical and categorical features)
- **Query-Anchored Selection**: TAQS adopts a query anchoring mechanism, using the task set as a reference and achieving task-domain alignment through local neighborhood coverage
- **Comprehensive Validation Framework**: Includes comparative experiments, convergence analysis, and ablation studies

## Repository Structure

```
TAQS/
├── configs/              # Configuration files
│   ├── datasets_config.yaml    # Dataset specifications
│   └── plot.yaml              # Visualization settings
├── src/                  # Source code
│   ├── taqs_algorithm.py       # Core TAQS implementation
│   ├── data_processor.py       # Data loading and preprocessing
│   ├── baselines.py            # Baseline methods (Random, K-Medoids, etc.)
│   ├── utils.py                # Utility functions
│   ├── comparative_experiment.py    # Main experiment pipeline of TAWG
│   ├── validation_taqs_paradigm.py # Main experiment pipeline of TAQS
│   └── other/                  # Additional analysis scripts
├── data/                 # Data directory (not included in repo)
│   ├── feature_weights/  # Weight of high-dimensional dataset
│   ├── raw/              # Original datasets (not included in repo)
│   └── processed/        # Preprocessed datasets
├── results/              # Experimental results (generated during execution)
└── requirements.txt      # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone <anonymous repository URL>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Availability

The datasets used in this work are collected from publicly available sources.
Preprocessed datasets used in the experiments are included in this repository.
Original datasets can be obtained from their respective public repositories.

## License

This code is provided for academic research and reproducibility purposes. Full licensing terms will be provided upon publication.

---

**Note**: This repository is submitted for double-blind peer review. Some identifying information has been temporarily removed to maintain anonymity. Complete metadata and documentation will be restored upon acceptance.

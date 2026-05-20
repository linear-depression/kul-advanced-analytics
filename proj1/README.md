# Advanced Analytics Assignment 1

This project predicts future customer revenue from historical customer transactions. The original exploratory notebook has been reorganized into a small Python package so the notebook can focus on orchestration, tables, plots, and results.

## Project Structure

```text
proj1/
├── data/                  # Input CSV files; generated feature matrices live in data/features/
├── config/
│   └── default.toml       # Versioned workflow settings and modeling constants
├── notebooks/
│   └── project_1.ipynb    # Clean orchestrator notebook
├── plots/                 # Saved EDA plots
├── report/                # Report material
├── src/proj1/             # Reusable project package
│   ├── config.py          # Typed config loading and path helpers
│   ├── data.py            # Data loading and feature persistence
│   ├── eda.py             # EDA tables and diagnostics
│   ├── features.py        # Customer-level feature engineering
│   ├── metrics.py         # Evaluation metrics
│   ├── modeling.py        # Splits, baselines, XGBoost, CV, submissions
│   ├── plots.py           # Plot functions with show/save support
│   └── workflow.py        # High-level orchestration used by the notebook
├── submissions/           # Generated leaderboard submissions
├── tests/                 # Unit tests with synthetic data
├── pyproject.toml
└── uv.lock
```

## Data Files

The workflow expects these CSV files:

```text
customer_clv_train.csv
customer_clv_test.csv
transactions_2016_2017.csv
```

Preferred location:

```text
proj1/data/
```

If your data lives elsewhere, configure it explicitly at the top of
`notebooks/project_1.ipynb`:

```python
DATA_DIR = Path("/path/to/folder/with/csv/files")
```

Alternatively, create a local `proj1/.env` file:

```text
PROJ1_DATA_DIR=/path/to/folder/with/csv/files
PROJ1_FEATURES_DIR=data/features
PROJ1_PLOTS_DIR=plots
PROJ1_SUBMISSIONS_DIR=submissions
PROJ1_REPORT_DIR=report
```

Then switch the notebook setup cell from `ProjectPaths.from_root(...)` to
`ProjectPaths.from_env(...)`. The `.env` file is local configuration and should
not be committed.

Generated feature matrices are saved to `proj1/data/features/` unless you
override `PROJ1_FEATURES_DIR`.

## Project Settings

Project constants are stored in:

```text
proj1/config/default.toml
```

This includes input filenames, output filenames, target/prediction column names,
the feature reference date, EDA bins, feature thresholds, model hyperparameters,
cross-validation settings, and notebook workflow switches.

For local changes that should not be committed, create:

```text
proj1/config/local.toml
```

Only include the values you want to override:

```toml
[features]
reference_date = "2018-01-01"

[workflow]
save_plots = false

[modeling]
validation_size = 0.25
```

`config/local.toml` is ignored by git. The notebook loads `default.toml` and
will merge `local.toml` when it exists.

## Setup

Run commands from the `proj1` folder.

```bash
uv sync
```

If you do not use `uv`, create an environment with Python 3.10+ and install the dependencies listed in `pyproject.toml`.

## Run the Notebook

```bash
uv run jupyter notebook notebooks/project_1.ipynb
```

The notebook is now an orchestrator. It loads data, runs EDA, builds features, trains models, validates performance, and writes submissions by calling functions from `src/proj1/`.

## Run Tests and Linting

```bash
uv run pytest
uv run ruff check
```

The tests use small synthetic data, so they do not need the private coursework CSV files.

## Outputs

- EDA plots: `plots/`
- Raw feature matrix: `data/features/customer_features_v4.parquet`
- Clean modeling feature matrix: `data/features/customer_features_v4_clean.parquet`
- Submission file: `submissions/v01_champion_xgb_mae_pp.csv`

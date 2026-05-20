"""Data loading and feature matrix persistence helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from proj1.config import DataConfig, ProjectPaths


def load_train(path: Path | str) -> pd.DataFrame:
    """Load the customer training table."""
    return pd.read_csv(path)


def load_test(path: Path | str) -> pd.DataFrame:
    """Load the customer test table."""
    return pd.read_csv(path)


def load_transactions(path: Path | str, parse_dates: list[str]) -> pd.DataFrame:
    """Load the transaction table with parsed date columns."""
    return pd.read_csv(path, parse_dates=parse_dates)


def load_datasets(
    paths: ProjectPaths,
    config: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and transaction tables."""
    validate_data_files(paths.data_dir, config)
    train = load_train(paths.train_file(config))
    test = load_test(paths.test_file(config))
    transactions = load_transactions(paths.transactions_file(config), config.transaction_parse_dates)
    return train, test, transactions


def validate_data_files(data_dir: Path | str, config: DataConfig) -> None:
    """Raise an error when required input CSV files are missing."""
    data_path = Path(data_dir)
    required_files = [
        config.train_filename,
        config.test_filename,
        config.transactions_filename,
    ]
    missing = [name for name in required_files if not (data_path / name).exists()]
    if missing:
        files = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required data file(s) in {data_path}: {files}. "
            "Pass ProjectPaths with the correct data_dir, set PROJ1_DATA_DIR in .env, "
            "or place the three CSV files in proj1/data/."
        )


def save_feature_matrix(features: pd.DataFrame, path: Path | str) -> Path:
    """Save a feature matrix as a parquet file and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path)
    return output_path


def load_feature_matrix(path: Path | str) -> pd.DataFrame:
    """Load a parquet feature matrix from disk."""
    return pd.read_parquet(path)

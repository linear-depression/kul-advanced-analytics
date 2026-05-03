from __future__ import annotations

from pathlib import Path

from proj1.config import ProjectPaths, load_project_config
from proj1.data import load_datasets


def test_load_datasets_reads_required_files(tmp_path, make_train_test_transactions, project_config):
    """Verify project data loaders read the required CSV files."""
    train, test, transactions = make_train_test_transactions
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train.to_csv(data_dir / project_config.data.train_filename, index=False)
    test.to_csv(data_dir / project_config.data.test_filename, index=False)
    transactions.to_csv(data_dir / project_config.data.transactions_filename, index=False)

    paths = ProjectPaths(
        root=tmp_path,
        data_dir=data_dir,
        features_dir=tmp_path / "data" / "features",
        plots_dir=tmp_path / "plots",
        submissions_dir=tmp_path / "submissions",
        report_dir=tmp_path / "report",
    )
    loaded_train, loaded_test, loaded_transactions = load_datasets(paths, project_config.data)

    assert loaded_train.shape == train.shape
    assert loaded_test.shape == test.shape
    assert loaded_transactions["order_date"].dtype.kind == "M"


def test_project_paths_can_be_loaded_from_env_file(tmp_path):
    """Verify local .env path overrides are parsed without hardcoded paths."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "PROJ1_DATA_DIR=external_data",
                "PROJ1_PLOTS_DIR=custom_plots",
            ]
        )
    )

    paths = ProjectPaths.from_env(env_file, root=tmp_path)

    assert paths.data_dir == tmp_path / "external_data"
    assert paths.plots_dir == tmp_path / "custom_plots"
    assert paths.features_dir == tmp_path / "data" / "features"


def test_project_config_supports_local_overrides(tmp_path):
    """Verify TOML config values can be overridden without touching defaults."""
    default_config = Path(__file__).resolve().parents[1] / "config" / "default.toml"
    local_config = tmp_path / "local.toml"
    local_config.write_text(
        """
[data]
train_filename = "custom_train.csv"

[modeling]
validation_size = 0.4
"""
    )

    config = load_project_config(default_config, local_config)

    assert config.data.train_filename == "custom_train.csv"
    assert config.data.test_filename == "customer_clv_test.csv"
    assert config.modeling.validation_size == 0.4

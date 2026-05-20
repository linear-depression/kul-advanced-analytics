from __future__ import annotations

import matplotlib

from proj1.eda import customer_statistics, missing_values_report, quick_customer_features
from proj1.plots import (
    plot_customer_activity,
    plot_missing_values,
    plot_recency_vs_target,
    plot_target_distribution,
)

matplotlib.use("Agg")


def test_plots_can_save_without_showing(tmp_path, make_train_test_transactions, project_config):
    """Verify plot functions can save files without displaying figures."""
    train, _, transactions = make_train_test_transactions
    missing = missing_values_report(transactions)
    stats = customer_statistics(transactions, project_config.data)
    quick = quick_customer_features(
        train,
        transactions,
        project_config.data,
        project_config.features,
        project_config.eda,
    )

    plot_target_distribution(
        train,
        project_config.data,
        project_config.plots,
        output_dir=tmp_path,
        show=False,
        save=True,
    )
    plot_missing_values(
        missing,
        project_config.plots,
        output_dir=tmp_path,
        show=False,
        save=True,
    )
    plot_customer_activity(
        stats,
        project_config.plots,
        output_dir=tmp_path,
        show=False,
        save=True,
    )
    plot_recency_vs_target(
        quick,
        project_config.data,
        project_config.plots,
        output_dir=tmp_path,
        show=False,
        save=True,
    )

    assert (tmp_path / "01_target_distribution.png").exists()
    assert (tmp_path / "03_missing_values.png").exists()
    assert (tmp_path / "04_orders_and_spend_per_customer.png").exists()
    assert (tmp_path / "06_recency_vs_target.png").exists()

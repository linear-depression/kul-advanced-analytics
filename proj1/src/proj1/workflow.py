"""High-level orchestration functions used by the notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from proj1.config import ProjectConfig, ProjectPaths
from proj1.data import load_datasets
from proj1.eda import (
    categorical_summary,
    customer_coverage,
    customer_statistics,
    missing_values_report,
    product_quality_diagnostics,
    quick_customer_features,
    quick_signal_tables,
    target_summary,
    transaction_quality_summary,
)
from proj1.features import build_customer_features, feature_quality_report
from proj1.metrics import describe_target, results_frame
from proj1.modeling import (
    cross_validate_champion,
    make_submission,
    prepare_modeling_data,
    run_course_baselines,
    run_sanity_baselines,
    run_xgboost_suite,
    save_submission,
    train_final_champion,
    train_validation_split,
)
from proj1.plots import (
    plot_categorical_distributions,
    plot_customer_activity,
    plot_frequency_vs_target,
    plot_missing_values,
    plot_monetary_vs_target,
    plot_recency_vs_target,
    plot_target_distribution,
    plot_target_log_distribution,
)


@dataclass
class EDAResults:
    """Collected EDA tables, diagnostics, quick features, and plot outputs."""
    target: dict[str, object]
    product_diagnostics: dict[str, object]
    transaction_quality: dict[str, object]
    missing_values: pd.DataFrame
    customer_stats: pd.DataFrame
    coverage: dict[str, int]
    categorical: dict[str, object]
    quick_features: pd.DataFrame
    quick_signals: dict[str, pd.DataFrame | pd.Series]
    plots: dict[str, Any]


@dataclass
class ModelingResults:
    """Prepared modeling matrices and validation split outputs."""
    X_train_full: pd.DataFrame
    y_train_full: pd.Series
    X_test: pd.DataFrame
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    split_summary: pd.DataFrame


def load_project_data(paths: ProjectPaths, config: ProjectConfig):
    """Load the project train, test, and transaction tables."""
    return load_datasets(paths, config.data)


def run_eda(
    train: pd.DataFrame,
    test: pd.DataFrame,
    transactions: pd.DataFrame,
    paths: ProjectPaths,
    config: ProjectConfig,
    show_plots: bool | None = None,
    save_plots: bool | None = None,
) -> EDAResults:
    """Run EDA tables and optional plot generation."""
    paths.ensure_output_dirs()
    show_plots = config.workflow.show_plots if show_plots is None else show_plots
    save_plots = config.workflow.save_plots if save_plots is None else save_plots

    target = target_summary(train, config.data, config.eda)
    product_diag = product_quality_diagnostics(transactions, config.eda)
    transaction_quality = transaction_quality_summary(transactions, config.data)
    missing = missing_values_report(transactions)
    customer_stats = customer_statistics(transactions, config.data)
    coverage = customer_coverage(train, test, customer_stats, config.data)
    categorical = categorical_summary(transactions, config.eda)
    quick = quick_customer_features(train, transactions, config.data, config.features, config.eda)
    quick_signals = quick_signal_tables(quick, config.data, config.eda)

    plot_outputs = {
        "target_distribution": plot_target_distribution(
            train,
            config.data,
            config.plots,
            output_dir=paths.plots_dir,
            show=show_plots,
            save=save_plots,
        ),
        "target_log_distribution": plot_target_log_distribution(
            train,
            config.data,
            config.plots,
            output_dir=paths.plots_dir,
            show=show_plots,
            save=save_plots,
        ),
        "missing_values": plot_missing_values(
            missing,
            config.plots,
            output_dir=paths.plots_dir,
            show=show_plots,
            save=save_plots,
        ),
        "customer_activity": plot_customer_activity(
            customer_stats,
            config.plots,
            output_dir=paths.plots_dir,
            show=show_plots,
            save=save_plots,
        ),
        "categorical_distributions": plot_categorical_distributions(
            transactions,
            config.plots,
            output_dir=paths.plots_dir,
            show=show_plots,
            save=save_plots,
        ),
        "recency_vs_target": plot_recency_vs_target(
            quick,
            config.data,
            config.plots,
            output_dir=paths.plots_dir,
            show=show_plots,
            save=save_plots,
        ),
        "frequency_vs_target": plot_frequency_vs_target(
            quick,
            config.data,
            config.plots,
            output_dir=paths.plots_dir,
            show=show_plots,
            save=save_plots,
        ),
        "monetary_vs_target": plot_monetary_vs_target(
            quick,
            config.data,
            config.plots,
            output_dir=paths.plots_dir,
            show=show_plots,
            save=save_plots,
        ),
    }
    return EDAResults(
        target=target,
        product_diagnostics=product_diag,
        transaction_quality=transaction_quality,
        missing_values=missing,
        customer_stats=customer_stats,
        coverage=coverage,
        categorical=categorical,
        quick_features=quick,
        quick_signals=quick_signals,
        plots=plot_outputs,
    )


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    transactions: pd.DataFrame,
    paths: ProjectPaths,
    config: ProjectConfig,
    save: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build, optionally save, and summarize engineered customer features."""
    paths.ensure_output_dirs()
    save = config.workflow.save_features if save is None else save
    features = build_customer_features(
        train,
        test,
        transactions,
        data_config=config.data,
        feature_config=config.features,
        paths=paths,
        save=save,
    )
    return features, feature_quality_report(features, train, test, config.data)


def prepare_modeling_workflow(
    features: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: ProjectConfig,
) -> ModelingResults:
    """Prepare train/test matrices and validation split summaries."""
    X_train_full, y_train_full, X_test = prepare_modeling_data(features, train, test, config.data)
    X_train, X_val, y_train, y_val = train_validation_split(
        X_train_full,
        y_train_full,
        config.modeling,
    )
    split_summary = pd.DataFrame(
        [
            describe_target(y_train_full, "Full train"),
            describe_target(y_train, "Train (80%)"),
            describe_target(y_val, "Validation (20%)"),
        ]
    ).set_index("split")
    return ModelingResults(
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        X_test=X_test,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        split_summary=split_summary,
    )


def run_baseline_models(
    modeling: ModelingResults,
    config: ProjectConfig,
    include_knn: bool | None = None,
) -> pd.DataFrame:
    """Run sanity and course-method baseline model groups."""
    include_knn = config.workflow.include_knn if include_knn is None else include_knn
    sanity = run_sanity_baselines(modeling.y_train, modeling.y_val)
    course = run_course_baselines(
        modeling.X_train,
        modeling.X_val,
        modeling.y_train,
        modeling.y_val,
        config.modeling,
        include_knn=include_knn,
    )
    return results_frame(sanity.to_dict("records") + course.results.to_dict("records"))


def run_main_models(
    modeling: ModelingResults,
    config: ProjectConfig,
    include_extended: bool | None = None,
):
    """Run the main XGBoost model suite."""
    include_extended = (
        config.workflow.run_extended_xgboost if include_extended is None else include_extended
    )
    return run_xgboost_suite(
        modeling.X_train,
        modeling.X_val,
        modeling.y_train,
        modeling.y_val,
        config.modeling,
        include_extended=include_extended,
    )


def run_champion_cv(
    modeling: ModelingResults,
    config: ProjectConfig,
    n_splits: int | None = None,
    zero_threshold: float | None = None,
):
    """Cross-validate the champion model configuration."""
    return cross_validate_champion(
        modeling.X_train_full,
        modeling.y_train_full,
        config.modeling,
        n_splits=n_splits,
        zero_threshold=zero_threshold,
    )


def generate_submission_files(
    modeling: ModelingResults,
    test: pd.DataFrame,
    paths: ProjectPaths,
    config: ProjectConfig,
    zero_threshold: float | None = None,
) -> dict[str, Path]:
    """Train the final champion model and write submission files."""
    paths.ensure_output_dirs()
    _, predictions = train_final_champion(
        modeling.X_train_full,
        modeling.y_train_full,
        modeling.X_test,
        config.modeling,
        zero_threshold=zero_threshold,
    )
    submission = make_submission(test, predictions, config.data)
    output_path = save_submission(submission, paths.submission_file(config.data))
    return {"champion": output_path}

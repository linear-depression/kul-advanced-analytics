"""Reusable EDA plotting functions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proj1.config import DataConfig, PlotConfig


def finalize_figure(
    fig,
    config: PlotConfig,
    filename: str | None = None,
    output_dir: Path | str | None = None,
    show: bool = True,
    save: bool = False,
    close: bool = False,
):
    """Show and/or save a matplotlib figure, then return it for notebook display."""
    saved_path = None
    fig.tight_layout()
    if save:
        if filename is None:
            raise ValueError("filename is required when save=True")
        target_dir = Path(output_dir or ".")
        target_dir.mkdir(parents=True, exist_ok=True)
        saved_path = target_dir / filename
        fig.savefig(saved_path, dpi=config.dpi, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, saved_path


def plot_target_distribution(
    train: pd.DataFrame,
    data_config: DataConfig,
    plot_config: PlotConfig,
    output_dir: Path | str | None = None,
    show: bool = True,
    save: bool = False,
):
    """Plot raw and positive-only target revenue distributions."""
    target_col = data_config.target_col
    y = train[target_col]
    y_nonzero = y[y > 0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(y, bins=plot_config.target_hist_bins, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].set_title("Target distribution (all customers)")
    axes[0].set_xlabel("revenue")
    axes[0].set_ylabel("# customers")
    axes[0].axvline(y.median(), color="red", ls="--", label=f"median={y.median():.1f}")
    axes[0].axvline(y.mean(), color="orange", ls="--", label=f"mean={y.mean():.1f}")
    axes[0].legend()

    axes[1].hist(
        y_nonzero,
        bins=plot_config.target_hist_bins,
        color="seagreen",
        edgecolor="black",
        alpha=0.8,
    )
    axes[1].set_title("Target distribution (returners only)")
    axes[1].set_xlabel("revenue")
    axes[1].set_ylabel("# customers")
    if len(y_nonzero) > 0:
        axes[1].axvline(
            y_nonzero.median(),
            color="red",
            ls="--",
            label=f"median={y_nonzero.median():.1f}",
        )
        axes[1].axvline(
            y_nonzero.mean(),
            color="orange",
            ls="--",
            label=f"mean={y_nonzero.mean():.1f}",
        )
        axes[1].legend()

    return finalize_figure(
        fig,
        plot_config,
        "01_target_distribution.png",
        output_dir,
        show,
        save,
    )


def plot_target_log_distribution(
    train: pd.DataFrame,
    data_config: DataConfig,
    plot_config: PlotConfig,
    output_dir: Path | str | None = None,
    show: bool = True,
    save: bool = False,
):
    """Plot the log-transformed positive target distribution."""
    target_col = data_config.target_col
    y_nonzero = train.loc[train[target_col] > 0, target_col]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        np.log1p(y_nonzero),
        bins=plot_config.target_log_hist_bins,
        color="seagreen",
        edgecolor="black",
        alpha=0.8,
    )
    ax.set_title("log1p(revenue) distribution among returners")
    ax.set_xlabel("log1p(revenue)")
    ax.set_ylabel("# customers")
    return finalize_figure(fig, plot_config, "02_target_log_scale.png", output_dir, show, save)


def plot_missing_values(
    missing_report: pd.DataFrame,
    plot_config: PlotConfig,
    output_dir: Path | str | None = None,
    show: bool = True,
    save: bool = False,
):
    """Plot missing-value percentages by transaction column."""
    fig, ax = plt.subplots(figsize=(10, 6))
    report = missing_report[missing_report["n_missing"] > 0].sort_values("pct_missing")
    if report.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.set_axis_off()
    else:
        report.plot.barh(y="pct_missing", legend=False, ax=ax, color="firebrick")
        ax.set_xlabel("% missing")
    ax.set_title("Missing values per column")
    return finalize_figure(fig, plot_config, "03_missing_values.png", output_dir, show, save)


def plot_customer_activity(
    customer_stats: pd.DataFrame,
    plot_config: PlotConfig,
    output_dir: Path | str | None = None,
    show: bool = True,
    save: bool = False,
):
    """Plot order frequency and spend distributions by customer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    customer_stats["n_orders"].clip(upper=plot_config.customer_order_clip).hist(
        bins=plot_config.customer_order_bins,
        color="teal",
        edgecolor="black",
        ax=axes[0],
    )
    axes[0].set_title(f"# orders per customer (capped at {plot_config.customer_order_clip})")
    axes[0].set_xlabel("# orders")
    axes[0].set_ylabel("# customers")

    customer_stats["total_revenue"].clip(upper=plot_config.customer_revenue_clip).hist(
        bins=plot_config.customer_revenue_bins,
        color="darkorange",
        edgecolor="black",
        ax=axes[1],
    )
    axes[1].set_title(
        f"Total revenue per customer (capped at {plot_config.customer_revenue_clip:g})"
    )
    axes[1].set_xlabel("total revenue")
    axes[1].set_ylabel("# customers")
    return finalize_figure(
        fig,
        plot_config,
        "04_orders_and_spend_per_customer.png",
        output_dir,
        show,
        save,
    )


def plot_categorical_distributions(
    transactions: pd.DataFrame,
    plot_config: PlotConfig,
    output_dir: Path | str | None = None,
    show: bool = True,
    save: bool = False,
):
    """Plot top product type, brand, and season categories."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    transactions["prod_type_1"].value_counts(dropna=False).head(plot_config.top_type_count).plot.bar(
        ax=axes[0],
        color="steelblue",
    )
    axes[0].set_title("prod_type_1")
    axes[0].tick_params(axis="x", rotation=45)

    transactions["prod_brand"].value_counts().head(plot_config.top_brand_count).plot.bar(
        ax=axes[1],
        color="seagreen",
    )
    axes[1].set_title(f"Top {plot_config.top_brand_count} prod_brand")
    axes[1].tick_params(axis="x", rotation=45)

    transactions["prod_season"].value_counts(dropna=False).head(plot_config.top_season_count).plot.bar(
        ax=axes[2],
        color="darkorange",
    )
    axes[2].set_title(f"Top {plot_config.top_season_count} prod_season")
    axes[2].tick_params(axis="x", rotation=45)
    return finalize_figure(
        fig,
        plot_config,
        "05_categorical_distributions.png",
        output_dir,
        show,
        save,
    )


def plot_recency_vs_target(
    quick_features: pd.DataFrame,
    data_config: DataConfig,
    plot_config: PlotConfig,
    output_dir: Path | str | None = None,
    show: bool = True,
    save: bool = False,
):
    """Plot returner probability and mean revenue by recency decile."""
    df = quick_features.copy()
    df["recency_bin"] = pd.qcut(
        df["recency_days"],
        plot_config.recency_quantile_bins,
        duplicates="drop",
    )
    recency_returner = df.groupby("recency_bin", observed=False)["is_returner"].mean()
    recency_revenue = df.groupby("recency_bin", observed=False)[data_config.target_col].mean()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    recency_returner.plot.bar(ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title("P(returner) by recency decile")
    axes[0].set_ylabel("P(revenue > 0)")
    axes[0].tick_params(axis="x", rotation=45)

    recency_revenue.plot.bar(ax=axes[1], color="darkorange", edgecolor="black")
    axes[1].set_title("Mean revenue by recency decile")
    axes[1].set_ylabel("mean revenue")
    axes[1].tick_params(axis="x", rotation=45)
    return finalize_figure(fig, plot_config, "06_recency_vs_target.png", output_dir, show, save)


def plot_frequency_vs_target(
    quick_features: pd.DataFrame,
    data_config: DataConfig,
    plot_config: PlotConfig,
    output_dir: Path | str | None = None,
    show: bool = True,
    save: bool = False,
):
    """Plot returner probability and mean revenue by order-frequency bucket."""
    df = quick_features.copy()
    df["freq_bucket"] = pd.cut(
        df["n_orders"],
        bins=plot_config.frequency_bucket_bins,
        labels=plot_config.frequency_bucket_labels,
    )
    freq_returner = df.groupby("freq_bucket", observed=False)["is_returner"].mean()
    freq_revenue = df.groupby("freq_bucket", observed=False)[data_config.target_col].mean()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    freq_returner.plot.bar(ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title("P(returner) by # orders")
    axes[0].set_ylabel("P(revenue > 0)")

    freq_revenue.plot.bar(ax=axes[1], color="darkorange", edgecolor="black")
    axes[1].set_title("Mean revenue by # orders")
    axes[1].set_ylabel("mean revenue")
    return finalize_figure(fig, plot_config, "07_frequency_vs_target.png", output_dir, show, save)


def plot_monetary_vs_target(
    quick_features: pd.DataFrame,
    data_config: DataConfig,
    plot_config: PlotConfig,
    output_dir: Path | str | None = None,
    show: bool = True,
    save: bool = False,
):
    """Plot returner probability and mean revenue by spend decile."""
    df = quick_features.copy()
    df["monetary_bin"] = pd.qcut(
        df["total_revenue_1617"],
        plot_config.monetary_quantile_bins,
        duplicates="drop",
    )
    mon_returner = df.groupby("monetary_bin", observed=False)["is_returner"].mean()
    mon_revenue = df.groupby("monetary_bin", observed=False)[data_config.target_col].mean()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    mon_returner.plot.bar(ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title("P(returner) by spend decile")
    axes[0].set_ylabel("P(revenue > 0)")
    axes[0].tick_params(axis="x", rotation=45)

    mon_revenue.plot.bar(ax=axes[1], color="darkorange", edgecolor="black")
    axes[1].set_title("Mean revenue by spend decile")
    axes[1].set_ylabel("mean revenue")
    axes[1].tick_params(axis="x", rotation=45)
    return finalize_figure(fig, plot_config, "08_monetary_vs_target.png", output_dir, show, save)

"""Exploratory data analysis tables and diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from proj1.config import DataConfig, EDAConfig, FeatureConfig


def target_summary(
    train: pd.DataFrame,
    data_config: DataConfig,
    eda_config: EDAConfig,
) -> dict[str, object]:
    """Summarize the target revenue distribution."""
    target_col = data_config.target_col
    y = train[target_col]
    y_nonzero = y[y > eda_config.positive_target_threshold]
    return {
        "n_customers": len(y),
        "n_zero": int((y == eda_config.positive_target_threshold).sum()),
        "pct_zero": float((y == eda_config.positive_target_threshold).mean() * 100),
        "n_positive": int((y > eda_config.positive_target_threshold).sum()),
        "pct_positive": float((y > eda_config.positive_target_threshold).mean() * 100),
        "n_negative": int((y < eda_config.positive_target_threshold).sum()),
        "summary": y.describe(),
        "positive_quantiles": y_nonzero.quantile(eda_config.target_quantiles),
    }


def product_quality_diagnostics(
    transactions: pd.DataFrame,
    eda_config: EDAConfig,
) -> dict[str, object]:
    """Diagnose non-numeric product fields that need cleaning."""
    diagnostics: dict[str, object] = {}
    for col in eda_config.product_quality_columns:
        numeric = pd.to_numeric(transactions[col], errors="coerce")
        non_numeric = numeric.isna() & transactions[col].notna()
        diagnostics[col] = {
            "top_values": transactions[col].value_counts(dropna=False).head(
                eda_config.top_product_quality_values
            ),
            "n_non_numeric": int(non_numeric.sum()),
            "non_numeric_values": transactions.loc[non_numeric, col]
            .value_counts()
            .head(eda_config.top_product_quality_values),
        }
    return diagnostics


def missing_values_report(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build a missing-value count and percentage table."""
    missing = transactions.isna().sum().sort_values(ascending=False)
    pct_missing = (missing / len(transactions) * 100).round(2)
    return pd.DataFrame({"n_missing": missing, "pct_missing": pct_missing})


def transaction_quality_summary(
    transactions: pd.DataFrame,
    data_config: DataConfig,
) -> dict[str, object]:
    """Summarize transaction table shape, dates, and revenue fields."""
    return {
        "n_rows": len(transactions),
        "n_customers": transactions[data_config.customer_id_col].nunique(),
        "n_orders": transactions["sale_id"].nunique(),
        "n_products": transactions["prod_id"].nunique(),
        "order_date_min": transactions["order_date"].min(),
        "order_date_max": transactions["order_date"].max(),
        "pack_date_min": transactions["pack_date"].min(),
        "pack_date_max": transactions["pack_date"].max(),
        "sale_revenue": transactions["sale_revenue"].describe(),
        "sale_discount_applied": transactions["sale_discount_applied"].describe(),
        "n_returned_items": int(transactions["returned_to_shop_id"].notna().sum()),
    }


def customer_statistics(transactions: pd.DataFrame, data_config: DataConfig) -> pd.DataFrame:
    """Aggregate transaction activity to the customer level."""
    stats = transactions.groupby(data_config.customer_id_col, observed=True).agg(
        n_rows=("sale_id", "size"),
        n_orders=("sale_id", "nunique"),
        n_products=("prod_id", "nunique"),
        total_revenue=("sale_revenue", "sum"),
        first_order=("order_date", "min"),
        last_order=("order_date", "max"),
    )
    stats["customer_lifespan_days"] = (stats["last_order"] - stats["first_order"]).dt.days
    return stats


def customer_coverage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    customer_stats: pd.DataFrame,
    data_config: DataConfig,
) -> dict[str, int]:
    """Count train and test customers represented in transactions."""
    customer_id_col = data_config.customer_id_col
    return {
        "train_in_transactions": int(train[customer_id_col].isin(customer_stats.index).sum()),
        "n_train": len(train),
        "test_in_transactions": int(test[customer_id_col].isin(customer_stats.index).sum()),
        "n_test": len(test),
    }


def categorical_summary(transactions: pd.DataFrame, eda_config: EDAConfig) -> dict[str, object]:
    """Summarize cardinality and top values for categorical product fields."""
    cardinality = []
    for col in eda_config.categorical_columns:
        if col in transactions.columns:
            cardinality.append(
                {
                    "column": col,
                    "n_unique": transactions[col].nunique(dropna=True),
                    "pct_missing": transactions[col].isna().mean() * 100,
                }
            )
    return {
        "cardinality": pd.DataFrame(cardinality),
        "top_brands": transactions["prod_brand"].value_counts().head(eda_config.top_brand_count),
        "prod_type_1": transactions["prod_type_1"].value_counts(dropna=False),
        "prod_season": transactions["prod_season"]
        .value_counts(dropna=False)
        .head(eda_config.top_season_count),
    }


def quick_customer_features(
    train: pd.DataFrame,
    transactions: pd.DataFrame,
    data_config: DataConfig,
    feature_config: FeatureConfig,
    eda_config: EDAConfig,
) -> pd.DataFrame:
    """Build lightweight customer features for EDA signal checks."""
    reference_date = pd.Timestamp(feature_config.reference_date)
    quick = transactions.groupby(data_config.customer_id_col, observed=True).agg(
        n_orders=("sale_id", "nunique"),
        n_items=("sale_id", "size"),
        total_revenue_1617=("sale_revenue", "sum"),
        mean_revenue=("sale_revenue", "mean"),
        last_order=("order_date", "max"),
        first_order=("order_date", "min"),
        total_discount=("sale_discount_applied", "sum"),
        n_unique_brands=("prod_brand", "nunique"),
        n_returns=("returned_to_shop_id", lambda s: s.notna().sum()),
    )
    quick["recency_days"] = (reference_date - quick["last_order"]).dt.days
    quick["tenure_days"] = (quick["last_order"] - quick["first_order"]).dt.days
    quick["return_rate"] = quick["n_returns"] / quick["n_items"]

    df = train.merge(quick, left_on=data_config.customer_id_col, right_index=True, how="left")
    df["is_returner"] = (
        df[data_config.target_col] > eda_config.positive_target_threshold
    ).astype(int)
    return df


def quick_signal_tables(
    quick_features: pd.DataFrame,
    data_config: DataConfig,
    eda_config: EDAConfig,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Compute quick feature correlations and returner comparisons."""
    feature_cols = eda_config.quick_feature_columns
    corr = (
        quick_features[feature_cols + [data_config.target_col]]
        .corr()[data_config.target_col]
        .drop(data_config.target_col)
        .sort_values(ascending=False)
    )
    group_means = quick_features.groupby("is_returner")[feature_cols].mean().T
    group_means.columns = ["non_returner", "returner"]
    group_means["ratio_R_to_NR"] = group_means["returner"] / group_means["non_returner"].replace(
        0, np.nan
    )
    return {"correlations": corr, "returner_comparison": group_means}

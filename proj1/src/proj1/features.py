"""Customer-level feature engineering for revenue prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from proj1.config import DataConfig, FeatureConfig, ProjectPaths
from proj1.data import save_feature_matrix


def parse_season_year(value) -> float:
    """Parse a product season label into a four-digit collection year."""
    if pd.isna(value):
        return np.nan
    digits = "".join(c for c in str(value) if c.isdigit())
    if len(digits) >= 2:
        yy = int(digits[-2:])
        return 2000 + yy if yy < 50 else 1900 + yy
    return np.nan


def prepare_transactions(
    transactions: pd.DataFrame,
    train_customer_ids,
    data_config: DataConfig,
    feature_config: FeatureConfig,
) -> pd.DataFrame:
    """Clean transactions and add helper columns for feature engineering."""
    tx = transactions.copy()
    _ensure_expected_columns(tx, data_config)
    tx["order_date"] = pd.to_datetime(tx["order_date"])
    tx["pack_date"] = pd.to_datetime(tx["pack_date"])
    tx["prod_size"] = pd.to_numeric(tx["prod_size"], errors="coerce")
    tx["order_year"] = tx["order_date"].dt.year
    tx["order_month"] = tx["order_date"].dt.to_period("M").astype(str)
    tx["order_quarter"] = tx["order_date"].dt.to_period("Q").astype(str)
    tx["order_month_num"] = tx["order_date"].dt.month
    tx["is_returned"] = tx["returned_to_shop_id"].notna().astype(int)
    tx["abs_discount"] = tx["sale_discount_applied"].abs()
    tx["has_discount"] = (tx["abs_discount"] > 0).astype(int)
    tx["collection_year"] = tx["prod_season"].apply(parse_season_year)

    train_ids = set(pd.Series(train_customer_ids).dropna())
    tx_train_only = tx[tx[data_config.customer_id_col].isin(train_ids)]
    brand_count = tx_train_only.groupby("prod_brand", observed=True).size()
    product_count = tx_train_only.groupby("prod_id", observed=True).size()

    rare_brand_threshold = brand_count.quantile(feature_config.rare_quantile) if len(brand_count) else 0
    rare_product_threshold = (
        product_count.quantile(feature_config.rare_quantile) if len(product_count) else 0
    )

    tx["brand_popularity"] = tx["prod_brand"].map(brand_count).fillna(0).astype("int32")
    tx["product_popularity"] = tx["prod_id"].map(product_count).fillna(0).astype("int32")
    tx["is_rare_brand"] = (tx["brand_popularity"] < rare_brand_threshold).astype("int8")
    tx["is_rare_product"] = (tx["product_popularity"] < rare_product_threshold).astype("int8")
    tx.attrs["ref_date"] = pd.Timestamp(feature_config.reference_date)
    tx.attrs["rare_brand_threshold"] = rare_brand_threshold
    tx.attrs["rare_product_threshold"] = rare_product_threshold
    return tx


def build_customer_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    transactions: pd.DataFrame,
    data_config: DataConfig,
    feature_config: FeatureConfig,
    paths: ProjectPaths | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Build the customer-level feature matrix used by the models."""
    if save and paths is None:
        raise ValueError("paths is required when save=True.")
    tx = prepare_transactions(
        transactions,
        train[data_config.customer_id_col],
        data_config=data_config,
        feature_config=feature_config,
    )
    fm = _build_raw_feature_matrix(
        tx,
        pd.Timestamp(feature_config.reference_date),
        data_config,
        feature_config,
    )
    fm = _ensure_requested_customers(fm, train, test, data_config)

    if save:
        assert paths is not None
        save_feature_matrix(fm, paths.raw_feature_file(data_config))

    fm = clean_feature_matrix(fm, data_config, feature_config)
    if save:
        assert paths is not None
        save_feature_matrix(fm, paths.clean_feature_file(data_config))
    return fm


def clean_feature_matrix(
    features: pd.DataFrame,
    data_config: DataConfig,
    feature_config: FeatureConfig,
) -> pd.DataFrame:
    """Convert raw engineered features into a numeric modeling matrix."""
    fm = features.copy()
    fm.index.name = data_config.customer_id_col

    brand_flag_cols = [c for c in fm.columns if c.startswith("buys_")]
    if brand_flag_cols:
        fm[brand_flag_cols] = fm[brand_flag_cols].fillna(0).astype("int8")
    if "n_unique_top_brands" in fm.columns:
        fm["n_unique_top_brands"] = fm["n_unique_top_brands"].fillna(0).astype("int8")

    if "dominant_type_1" in fm.columns:
        allowed_values = [value for value in feature_config.dominant_type_values if value != "unknown"]
        dominant = fm["dominant_type_1"].where(fm["dominant_type_1"].isin(allowed_values))
        one_hot = pd.get_dummies(dominant, prefix="dominant_type_1", dtype="int8")
        for value in allowed_values:
            col = f"dominant_type_1_{value}"
            if col not in one_hot.columns:
                one_hot[col] = 0
        expected_columns = [f"dominant_type_1_{value}" for value in allowed_values]
        fm = fm.drop(columns=["dominant_type_1"]).join(one_hot[expected_columns])

    date_cols = data_config.dropped_feature_date_columns
    fm = fm.drop(columns=[c for c in date_cols if c in fm.columns])

    object_cols = list(fm.select_dtypes(include="object").columns)
    if object_cols:
        raise ValueError(f"Feature matrix still has object columns: {object_cols}")
    return fm.sort_index()


def feature_quality_report(
    features: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    data_config: DataConfig,
) -> dict[str, object]:
    """Report feature matrix coverage and missing-value rates."""
    customer_id_col = data_config.customer_id_col
    train_in_fm = train[customer_id_col].isin(features.index).sum()
    test_in_fm = test[customer_id_col].isin(features.index).sum()
    nan_pct = (features.isna().mean() * 100).sort_values(ascending=False)
    return {
        "shape": features.shape,
        "train_in_fm": int(train_in_fm),
        "n_train": len(train),
        "test_in_fm": int(test_in_fm),
        "n_test": len(test),
        "top_nan_pct": nan_pct.head(10),
    }


def _build_raw_feature_matrix(
    tx: pd.DataFrame,
    ref_date: pd.Timestamp,
    data_config: DataConfig,
    feature_config: FeatureConfig,
) -> pd.DataFrame:
    """Build all customer-level feature groups before final cleaning."""
    customer_id_col = data_config.customer_id_col
    order_level = tx.groupby([customer_id_col, "sale_id"], observed=True).agg(
        order_revenue=("sale_revenue", "sum"),
        order_returned=("is_returned", "max"),
    ).reset_index()

    agg_item = tx.groupby(customer_id_col, observed=True).agg(
        n_items=("sale_id", "size"),
        total_revenue=("sale_revenue", "sum"),
        median_item_revenue=("sale_revenue", "median"),
        max_item_revenue=("sale_revenue", "max"),
        std_item_revenue=("sale_revenue", "std"),
        last_order_date=("order_date", "max"),
        first_order_date=("order_date", "min"),
        n_unique_brands=("prod_brand", "nunique"),
    )
    agg_order = order_level.groupby(customer_id_col, observed=True).agg(
        n_orders=("sale_id", "nunique"),
        mean_order_revenue=("order_revenue", "mean"),
    )
    fm = agg_item.join(agg_order)
    fm["n_items_per_order"] = fm["n_items"] / fm["n_orders"]
    fm["recency_days"] = (ref_date - fm["last_order_date"]).dt.days
    fm["first_order_days_ago"] = (ref_date - fm["first_order_date"]).dt.days
    fm["tenure_days"] = (fm["last_order_date"] - fm["first_order_date"]).dt.days

    gross = (
        tx[tx["is_returned"] == 0]
        .assign(gross_item=lambda d: d["sale_revenue"] + d["abs_discount"])
        .groupby(customer_id_col, observed=True)["gross_item"]
        .sum()
        .rename("gross_revenue")
    )
    fm = fm.join(gross)
    fm["gross_revenue"] = fm["gross_revenue"].fillna(0)
    fm["net_to_gross_revenue_ratio"] = np.where(
        fm["gross_revenue"] > 0,
        fm["total_revenue"] / fm["gross_revenue"],
        1.0,
    )

    fm = _add_returns(fm, tx, order_level, customer_id_col)
    fm = _add_discounts(fm, tx, customer_id_col)
    fm = _add_engagement(fm, feature_config)
    fm = _add_time_dynamics(fm, tx, ref_date, feature_config, customer_id_col)
    fm = _add_product_diversity(fm, tx, customer_id_col)
    fm = _add_gender_age_preferences(fm, tx, feature_config, customer_id_col)
    fm = _add_brand_preferences(fm, tx, feature_config, customer_id_col)
    fm = _add_season_collection(fm, tx, feature_config, customer_id_col)
    fm = _add_channel_comfort_missingness(fm, tx, customer_id_col)
    fm = _add_popularity(fm, tx, customer_id_col)
    return fm


def _add_returns(
    fm: pd.DataFrame,
    tx: pd.DataFrame,
    order_level: pd.DataFrame,
    customer_id_col: str,
) -> pd.DataFrame:
    """Add return-count, return-rate, and lost-revenue features."""
    returns_item = tx.groupby(customer_id_col, observed=True).agg(n_returns=("is_returned", "sum"))
    returns_order = order_level.groupby(customer_id_col, observed=True).agg(
        n_orders_with_return=("order_returned", "sum")
    )
    returns = returns_item.join(returns_order)
    lost = (
        tx[tx["sale_revenue"] < 0]
        .groupby(customer_id_col, observed=True)["sale_revenue"]
        .sum()
        .abs()
        .rename("revenue_lost_to_returns")
    )
    returns = returns.join(lost)
    returns["revenue_lost_to_returns"] = returns["revenue_lost_to_returns"].fillna(0)
    fm = fm.join(returns)
    fm["return_rate"] = fm["n_returns"] / fm["n_items"]
    fm["pct_orders_with_return"] = fm["n_orders_with_return"] / fm["n_orders"]
    return fm


def _add_discounts(fm: pd.DataFrame, tx: pd.DataFrame, customer_id_col: str) -> pd.DataFrame:
    """Add discount amount and discount-rate features."""
    discounts = tx.groupby(customer_id_col, observed=True).agg(
        total_discount_amt=("abs_discount", "sum"),
        mean_discount_amt=("abs_discount", "mean"),
        max_discount_amt=("abs_discount", "max"),
        n_discounted_items=("has_discount", "sum"),
    )
    fm = fm.join(discounts)
    fm["pct_items_discounted"] = fm["n_discounted_items"] / fm["n_items"]
    denom = fm["total_revenue"] + fm["total_discount_amt"]
    fm["discount_rate"] = np.where(denom > 0, fm["total_discount_amt"] / denom, 0)
    return fm


def _add_engagement(fm: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add customer engagement flags and log-transformed features."""
    fm["log_n_orders"] = np.log1p(fm["n_orders"])
    fm["log_total_revenue"] = np.log1p(fm["total_revenue"].clip(lower=0))
    fm["is_one_time_buyer"] = (fm["n_orders"] == config.one_time_order_count).astype("int8")
    fm["is_repeat_buyer"] = (fm["n_orders"] >= config.repeat_order_min).astype("int8")
    fm["is_loyal"] = (fm["n_orders"] >= config.loyal_order_min).astype("int8")
    fm["is_high_value_customer"] = (fm["total_revenue"] > config.high_value_revenue_min).astype(
        "int8"
    )
    fm["is_high_frequency_customer"] = (fm["n_orders"] >= config.high_frequency_order_min).astype(
        "int8"
    )
    fm["bought_recently"] = (fm["recency_days"] <= config.bought_recently_days).astype("int8")
    return fm


def _add_time_dynamics(
    fm: pd.DataFrame,
    tx: pd.DataFrame,
    ref_date: pd.Timestamp,
    config: FeatureConfig,
    customer_id_col: str,
) -> pd.DataFrame:
    """Add active-period, trend, interval, and recent-window features."""
    time_diversity = tx.groupby(customer_id_col, observed=True).agg(
        n_active_months=("order_month", "nunique"),
        n_active_quarters=("order_quarter", "nunique"),
    )
    year_rev = tx.pivot_table(
        index=customer_id_col,
        columns="order_year",
        values="sale_revenue",
        aggfunc="sum",
        fill_value=0,
        observed=True,
    )
    year_rev.columns = [f"revenue_{int(y)}" for y in year_rev.columns]
    for year in config.history_years:
        col = f"revenue_{year}"
        if col not in year_rev.columns:
            year_rev[col] = 0
    start_col = f"revenue_{config.trend_start_year}"
    end_col = f"revenue_{config.trend_end_year}"
    year_rev["revenue_trend"] = year_rev[end_col] - year_rev[start_col]
    year_rev["revenue_growth_ratio"] = np.where(
        year_rev[start_col] > 0,
        year_rev[end_col] / year_rev[start_col],
        np.where(year_rev[end_col] > 0, config.revenue_growth_fallback, 0.0),
    )
    year_rev["revenue_growth_ratio"] = year_rev["revenue_growth_ratio"].clip(
        upper=config.revenue_growth_cap
    )

    order_dates = (
        tx.groupby([customer_id_col, "sale_id"], observed=True)["order_date"].min().reset_index()
    )
    inter = order_dates.groupby(customer_id_col, observed=True).apply(
        _inter_purchase_stats, include_groups=False
    )

    recent_3m = tx[tx["order_date"] >= ref_date - pd.Timedelta(days=config.recent_3m_days)].groupby(
        customer_id_col, observed=True
    ).agg(
        n_orders_last_3m=("sale_id", "nunique"),
        revenue_last_3m=("sale_revenue", "sum"),
    )
    recent_6m = tx[tx["order_date"] >= ref_date - pd.Timedelta(days=config.recent_6m_days)].groupby(
        customer_id_col, observed=True
    ).agg(
        n_orders_last_6m=("sale_id", "nunique"),
        revenue_last_6m=("sale_revenue", "sum"),
        n_returns_last_6m=("is_returned", "sum"),
        discount_amt_last_6m=("abs_discount", "sum"),
    )
    t5 = time_diversity.join(year_rev).join(inter).join(recent_3m).join(recent_6m)
    recent_cols = [
        "n_orders_last_3m",
        "revenue_last_3m",
        "n_orders_last_6m",
        "revenue_last_6m",
        "n_returns_last_6m",
        "discount_amt_last_6m",
    ]
    for col in recent_cols:
        if col in t5.columns:
            t5[col] = t5[col].fillna(0)
    fm = fm.join(t5)
    fm["share_revenue_last_6m"] = np.where(
        fm["total_revenue"] > 0,
        fm["revenue_last_6m"] / fm["total_revenue"],
        0,
    )
    fm["share_orders_last_6m"] = np.where(
        fm["n_orders"] > 0,
        fm["n_orders_last_6m"] / fm["n_orders"],
        0,
    )
    returner_year_col = f"revenue_{config.returner_history_year}"
    fm[f"bought_in_{config.returner_history_year}"] = (fm[returner_year_col] > 0).astype("int8")
    return fm


def _inter_purchase_stats(group: pd.DataFrame) -> pd.Series:
    """Compute mean and standard deviation of days between orders."""
    if len(group) < 2:
        return pd.Series({"mean_inter_purchase_days": np.nan, "std_inter_purchase_days": np.nan})
    dates_sorted = group["order_date"].sort_values()
    diffs = dates_sorted.diff().dt.days.dropna()
    return pd.Series(
        {
            "mean_inter_purchase_days": diffs.mean(),
            "std_inter_purchase_days": diffs.std() if len(diffs) >= 2 else np.nan,
        }
    )


def _add_product_diversity(
    fm: pd.DataFrame,
    tx: pd.DataFrame,
    customer_id_col: str,
) -> pd.DataFrame:
    """Add product, color, season, and size diversity features."""
    diversity = tx.groupby(customer_id_col, observed=True).agg(
        n_unique_products=("prod_id", "nunique"),
        n_unique_colors=("prod_color", "nunique"),
        n_unique_seasons=("prod_season", "nunique"),
        n_unique_sizes=("prod_size", "nunique"),
        mean_prod_size=("prod_size", "mean"),
        std_prod_size=("prod_size", "std"),
    )
    return fm.join(diversity)


def _add_gender_age_preferences(
    fm: pd.DataFrame,
    tx: pd.DataFrame,
    config: FeatureConfig,
    customer_id_col: str,
) -> pd.DataFrame:
    """Add customer gender and age-category preference features."""
    ct_type1 = pd.crosstab(tx[customer_id_col], tx["prod_type_1"], normalize="index")
    ct_type1.columns = [f"pct_{str(c)}" for c in ct_type1.columns]
    for col in ["pct_women", "pct_men", "pct_boys", "pct_girls"]:
        if col not in ct_type1.columns:
            ct_type1[col] = 0
    n_distinct_type1 = tx.groupby(customer_id_col, observed=True)["prod_type_1"].nunique().rename(
        "n_distinct_type_1"
    )
    dominant_type1 = (
        tx.groupby(customer_id_col, observed=True)["prod_type_1"]
        .agg(_safe_mode)
        .rename("dominant_type_1")
    )
    t7 = ct_type1.join(n_distinct_type1).join(dominant_type1)
    t7["buys_kids"] = ((t7["pct_boys"] + t7["pct_girls"]) > 0).astype("int8")
    t7["buys_adults"] = ((t7["pct_men"] + t7["pct_women"]) > 0).astype("int8")
    t7["buys_for_family"] = (t7["n_distinct_type_1"] >= config.family_type_min_count).astype(
        "int8"
    )
    t7["is_purely_women_buyer"] = (t7["pct_women"] == config.purely_women_share).astype("int8")
    return fm.join(t7)


def _safe_mode(series: pd.Series):
    """Return the most frequent non-null value or an unknown fallback."""
    mode = series.mode(dropna=True)
    return mode.iloc[0] if len(mode) > 0 else "unknown"


def _add_brand_preferences(
    fm: pd.DataFrame,
    tx: pd.DataFrame,
    config: FeatureConfig,
    customer_id_col: str,
) -> pd.DataFrame:
    """Add top-brand flags and aggregated brand preference features."""
    brand_pivot = (
        tx[tx["prod_brand"].isin(config.top_brands)]
        .groupby([customer_id_col, "prod_brand"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    brand_pivot = brand_pivot.reindex(columns=config.top_brands, fill_value=0)
    brand_pivot = (brand_pivot > 0).astype("int8")
    brand_pivot.columns = [_brand_flag_name(c) for c in brand_pivot.columns]
    n_top_brands = brand_pivot.sum(axis=1).rename("n_unique_top_brands").astype("int8")

    pct_top_items = (
        tx.assign(is_top=tx["prod_brand"].isin(config.top_brands).astype(int))
        .groupby(customer_id_col, observed=True)["is_top"]
        .mean()
        .rename("pct_orders_top15_brand")
    )
    t8 = (
        brand_pivot.join(n_top_brands)
        .join(pct_top_items)
        .join(
            _has_any_brand_in(
                tx,
                config.premium_brands,
                "is_premium_brand_buyer",
                customer_id_col,
            )
        )
        .join(_has_any_brand_in(tx, config.sport_brands, "is_sport_brand_buyer", customer_id_col))
        .join(
            _has_any_brand_in(
                tx,
                config.classic_brands,
                "is_classic_brand_buyer",
                customer_id_col,
            )
        )
    )
    return fm.join(t8)


def _brand_flag_name(brand: str) -> str:
    """Convert a brand name into a stable binary feature name."""
    return "buys_" + brand.lower().replace(" ", "_").replace("and", "_").replace("__", "_")


def _has_any_brand_in(
    tx: pd.DataFrame,
    brand_list: list[str],
    name: str,
    customer_id_col: str,
) -> pd.Series:
    """Return a customer-level flag for membership in a brand group."""
    return (
        tx.assign(_x=tx["prod_brand"].isin(brand_list).astype(int))
        .groupby(customer_id_col, observed=True)["_x"]
        .max()
        .astype("int8")
        .rename(name)
    )


def _add_season_collection(
    fm: pd.DataFrame,
    tx: pd.DataFrame,
    config: FeatureConfig,
    customer_id_col: str,
) -> pd.DataFrame:
    """Add product collection and holiday-season features."""
    season_feat = tx.groupby(customer_id_col, observed=True).agg(
        mean_collection_year=("collection_year", "mean"),
        n_unique_collection_years=("collection_year", "nunique"),
    )
    season_feat["pct_recent_collection"] = (
        tx.assign(recent=(tx["collection_year"] >= config.recent_collection_min_year).astype(int))
        .groupby(customer_id_col, observed=True)["recent"]
        .mean()
    )
    season_feat["pct_old_stock"] = (
        tx.assign(old=(tx["collection_year"] <= config.old_stock_max_year).astype(int))
        .groupby(customer_id_col, observed=True)["old"]
        .mean()
    )
    season_feat["mean_collection_lag"] = (
        tx.assign(lag=tx["order_year"] - tx["collection_year"])
        .groupby(customer_id_col, observed=True)["lag"]
        .mean()
    )
    season_feat["n_orders_in_holiday"] = (
        tx[tx["order_month_num"].isin(config.holiday_months)]
        .groupby(customer_id_col, observed=True)["sale_id"]
        .nunique()
    )
    season_feat["n_orders_in_holiday"] = season_feat["n_orders_in_holiday"].fillna(0).astype(
        "int16"
    )
    return fm.join(season_feat)


def _add_channel_comfort_missingness(
    fm: pd.DataFrame,
    tx: pd.DataFrame,
    customer_id_col: str,
) -> pd.DataFrame:
    """Add channel, comfort, and product-missingness features."""
    channel = tx.groupby(customer_id_col, observed=True).agg(
        pct_web_only=("prod_web_only", "mean"),
        mean_outlet=("prod_outlet", "mean"),
    )
    outlet_median = tx["prod_outlet"].median()
    channel["pct_outlet_high"] = (
        tx.assign(high=(tx["prod_outlet"] > outlet_median).astype(int))
        .groupby(customer_id_col, observed=True)["high"]
        .mean()
    )
    comfort = tx.groupby(customer_id_col, observed=True).agg(
        pct_with_comfort_features=("prod_comfort_sole", lambda s: s.notna().mean()),
        pct_with_comfort_wear=("prod_comfort_wear", lambda s: s.notna().mean()),
    )
    missing = tx.groupby(customer_id_col, observed=True).agg(
        missing_prod_heel_rate=("prod_heel", lambda s: s.isna().mean()),
        missing_prod_clasp_rate=("prod_clasp", lambda s: s.isna().mean()),
    )
    return fm.join(channel.join(comfort).join(missing))


def _add_popularity(fm: pd.DataFrame, tx: pd.DataFrame, customer_id_col: str) -> pd.DataFrame:
    """Add train-only brand and product popularity encoding features."""
    tx = tx.copy()
    tx["log_brand_pop"] = np.log1p(tx["brand_popularity"])
    tx["log_product_pop"] = np.log1p(tx["product_popularity"])
    popularity = tx.groupby(customer_id_col, observed=True).agg(
        mean_log_brand_popularity=("log_brand_pop", "mean"),
        mean_log_product_popularity=("log_product_pop", "mean"),
        rare_brand_item_rate=("is_rare_brand", "mean"),
    )
    return fm.join(popularity)


def _ensure_requested_customers(
    features: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    data_config: DataConfig,
) -> pd.DataFrame:
    """Ensure all train and test customers appear in the feature index."""
    customer_id_col = data_config.customer_id_col
    all_customer_ids = pd.Index(
        pd.concat([train[customer_id_col], test[customer_id_col]], ignore_index=True).drop_duplicates(),
        name=customer_id_col,
    )
    return features.reindex(features.index.union(all_customer_ids))


def _ensure_expected_columns(transactions: pd.DataFrame, config: DataConfig) -> None:
    """Validate required transaction columns and add optional defaults."""
    missing_required = [
        col for col in config.required_transaction_columns if col not in transactions.columns
    ]
    if missing_required:
        raise ValueError(f"Transactions missing required columns: {missing_required}")
    for col, value in config.normalized_transaction_defaults().items():
        if col not in transactions.columns:
            transactions[col] = value

"""Project configuration, typed settings, and path helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised only on Python 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10.
    import tomli as tomllib


@dataclass(frozen=True)
class DataConfig:
    """Input, output, and schema settings for project data files."""

    train_filename: str
    test_filename: str
    transactions_filename: str
    raw_feature_filename: str
    clean_feature_filename: str
    submission_filename: str
    customer_id_col: str
    target_col: str
    prediction_col: str
    transaction_parse_dates: list[str]
    required_transaction_columns: list[str]
    transaction_defaults: dict[str, Any]
    dropped_feature_date_columns: list[str]

    def normalized_transaction_defaults(self) -> dict[str, Any]:
        """Return transaction defaults with TOML string sentinels converted."""
        return {key: _normalize_default_value(value) for key, value in self.transaction_defaults.items()}


@dataclass(frozen=True)
class EDAConfig:
    """Settings used by exploratory data analysis tables."""

    target_quantiles: list[float]
    product_quality_columns: list[str]
    top_product_quality_values: int
    categorical_columns: list[str]
    top_brand_count: int
    top_season_count: int
    quick_feature_columns: list[str]
    positive_target_threshold: float


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering settings and business-rule thresholds."""

    reference_date: str
    rare_quantile: float
    top_brands: list[str]
    premium_brands: list[str]
    sport_brands: list[str]
    classic_brands: list[str]
    dominant_type_values: list[str]
    one_time_order_count: int
    repeat_order_min: int
    loyal_order_min: int
    high_value_revenue_min: float
    high_frequency_order_min: int
    bought_recently_days: int
    recent_3m_days: int
    recent_6m_days: int
    revenue_growth_fallback: float
    revenue_growth_cap: float
    history_years: list[int]
    trend_start_year: int
    trend_end_year: int
    returner_history_year: int
    family_type_min_count: int
    purely_women_share: float
    recent_collection_min_year: int
    old_stock_max_year: int
    holiday_months: list[int]


@dataclass(frozen=True)
class PlotConfig:
    """Plot styling and binning settings."""

    dpi: int
    target_hist_bins: int
    target_log_hist_bins: int
    customer_order_clip: int
    customer_order_bins: int
    customer_revenue_clip: float
    customer_revenue_bins: int
    top_type_count: int
    top_brand_count: int
    top_season_count: int
    recency_quantile_bins: int
    monetary_quantile_bins: int
    frequency_bucket_bins: list[float]
    frequency_bucket_labels: list[str]


@dataclass(frozen=True)
class ModelConfig:
    """Model split, hyperparameter, and post-processing settings."""

    random_seed: int
    validation_size: float
    ridge_params: dict[str, Any]
    decision_tree_params: dict[str, Any]
    knn_params: dict[str, Any]
    random_forest_params: dict[str, Any]
    xgb_common_params: dict[str, Any]
    xgb_extended_params: dict[str, Any]
    xgb_cv_params: dict[str, Any]
    xgb_final_params: dict[str, Any]
    xgb_single_objective: str
    xgb_tweedie_objective: str
    xgb_tweedie_params: dict[str, Any]
    xgb_hurdle_classifier_objective: str
    xgb_hurdle_classifier_metric: str
    xgb_hurdle_regressor_objective: str
    xgb_mse_objective: str
    xgb_log_objective: str
    xgb_pseudo_huber_objective: str
    xgb_fallback_objective: str
    zero_threshold_grid: list[float]
    champion_n_splits: int
    champion_zero_threshold: float


@dataclass(frozen=True)
class WorkflowConfig:
    """Notebook orchestration switches for expensive or optional stages."""

    show_plots: bool
    save_plots: bool
    save_features: bool
    run_course_baselines: bool
    run_xgboost_models: bool
    run_extended_xgboost: bool
    run_cross_validation: bool
    run_submissions: bool
    include_knn: bool


@dataclass(frozen=True)
class ProjectConfig:
    """Complete typed configuration for the customer revenue workflow."""

    data: DataConfig
    eda: EDAConfig
    features: FeatureConfig
    plots: PlotConfig
    modeling: ModelConfig
    workflow: WorkflowConfig


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout used by the project workflow."""

    root: Path
    data_dir: Path
    features_dir: Path
    plots_dir: Path
    submissions_dir: Path
    report_dir: Path

    @classmethod
    def from_root(
        cls,
        root: Path | str,
        data_dir: Path | str | None = None,
        features_dir: Path | str | None = None,
        plots_dir: Path | str | None = None,
        submissions_dir: Path | str | None = None,
        report_dir: Path | str | None = None,
    ) -> "ProjectPaths":
        """Build project paths from explicit root and optional folder overrides."""
        project_root = Path(root).expanduser().resolve()
        return cls(
            root=project_root,
            data_dir=_resolve_path(data_dir, project_root) if data_dir is not None else project_root / "data",
            features_dir=_resolve_path(features_dir, project_root)
            if features_dir is not None
            else project_root / "data" / "features",
            plots_dir=_resolve_path(plots_dir, project_root)
            if plots_dir is not None
            else project_root / "plots",
            submissions_dir=_resolve_path(submissions_dir, project_root)
            if submissions_dir is not None
            else project_root / "submissions",
            report_dir=_resolve_path(report_dir, project_root)
            if report_dir is not None
            else project_root / "report",
        )

    @classmethod
    def from_env(cls, env_file: Path | str, root: Path | str) -> "ProjectPaths":
        """Build project paths from a simple .env file and explicit project root."""
        values = read_env_file(env_file)
        return cls.from_root(
            root=root,
            data_dir=values.get("PROJ1_DATA_DIR"),
            features_dir=values.get("PROJ1_FEATURES_DIR"),
            plots_dir=values.get("PROJ1_PLOTS_DIR"),
            submissions_dir=values.get("PROJ1_SUBMISSIONS_DIR"),
            report_dir=values.get("PROJ1_REPORT_DIR"),
        )

    def ensure_output_dirs(self) -> None:
        """Create output directories used by the workflow."""
        for path in [self.features_dir, self.plots_dir, self.submissions_dir, self.report_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def train_file(self, config: DataConfig) -> Path:
        """Return the configured path to the training customer revenue CSV."""
        return self.data_dir / config.train_filename

    def test_file(self, config: DataConfig) -> Path:
        """Return the configured path to the test customer CSV."""
        return self.data_dir / config.test_filename

    def transactions_file(self, config: DataConfig) -> Path:
        """Return the configured path to the historical transactions CSV."""
        return self.data_dir / config.transactions_filename

    def raw_feature_file(self, config: DataConfig) -> Path:
        """Return the configured path for the raw engineered feature matrix."""
        return self.features_dir / config.raw_feature_filename

    def clean_feature_file(self, config: DataConfig) -> Path:
        """Return the configured path for the cleaned modeling-ready feature matrix."""
        return self.features_dir / config.clean_feature_filename

    def submission_file(self, config: DataConfig) -> Path:
        """Return the configured path for the champion submission CSV."""
        return self.submissions_dir / config.submission_filename


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest parent folder that looks like the proj1 root."""
    current = (start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "proj1").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


def default_config_path(root: Path | str) -> Path:
    """Return the conventional versioned configuration file for a project root."""
    return Path(root).expanduser().resolve() / "config" / "default.toml"


def load_project_config(
    config_file: Path | str,
    override_file: Path | str | None = None,
) -> ProjectConfig:
    """Load project settings from TOML, optionally merging local overrides."""
    raw = _read_toml(config_file)
    if override_file is not None:
        override_path = Path(override_file).expanduser()
        if override_path.exists():
            raw = _deep_merge(raw, _read_toml(override_path))
    return _build_project_config(raw)


def read_env_file(env_file: Path | str) -> dict[str, str]:
    """Read key-value path overrides from a minimal .env file."""
    path = Path(env_file).expanduser()
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _build_project_config(raw: dict[str, Any]) -> ProjectConfig:
    """Build typed dataclasses from a raw TOML dictionary."""
    data = _table(raw, "data")
    eda = _table(raw, "eda")
    features = _table(raw, "features")
    brand_groups = _table(features, "brand_groups")
    plots = _table(raw, "plots")
    modeling = _table(raw, "modeling")
    workflow = _table(raw, "workflow")

    return ProjectConfig(
        data=DataConfig(
            train_filename=_str(data, "train_filename"),
            test_filename=_str(data, "test_filename"),
            transactions_filename=_str(data, "transactions_filename"),
            raw_feature_filename=_str(data, "raw_feature_filename"),
            clean_feature_filename=_str(data, "clean_feature_filename"),
            submission_filename=_str(data, "submission_filename"),
            customer_id_col=_str(data, "customer_id_col"),
            target_col=_str(data, "target_col"),
            prediction_col=_str(data, "prediction_col"),
            transaction_parse_dates=_str_list(data, "transaction_parse_dates"),
            required_transaction_columns=_str_list(data, "required_transaction_columns"),
            transaction_defaults=dict(_table(data, "transaction_defaults")),
            dropped_feature_date_columns=_str_list(data, "dropped_feature_date_columns"),
        ),
        eda=EDAConfig(
            target_quantiles=_float_list(eda, "target_quantiles"),
            product_quality_columns=_str_list(eda, "product_quality_columns"),
            top_product_quality_values=_int(eda, "top_product_quality_values"),
            categorical_columns=_str_list(eda, "categorical_columns"),
            top_brand_count=_int(eda, "top_brand_count"),
            top_season_count=_int(eda, "top_season_count"),
            quick_feature_columns=_str_list(eda, "quick_feature_columns"),
            positive_target_threshold=_float(eda, "positive_target_threshold"),
        ),
        features=FeatureConfig(
            reference_date=_str(features, "reference_date"),
            rare_quantile=_float(features, "rare_quantile"),
            top_brands=_str_list(brand_groups, "top"),
            premium_brands=_str_list(brand_groups, "premium"),
            sport_brands=_str_list(brand_groups, "sport"),
            classic_brands=_str_list(brand_groups, "classic"),
            dominant_type_values=_str_list(features, "dominant_type_values"),
            one_time_order_count=_int(features, "one_time_order_count"),
            repeat_order_min=_int(features, "repeat_order_min"),
            loyal_order_min=_int(features, "loyal_order_min"),
            high_value_revenue_min=_float(features, "high_value_revenue_min"),
            high_frequency_order_min=_int(features, "high_frequency_order_min"),
            bought_recently_days=_int(features, "bought_recently_days"),
            recent_3m_days=_int(features, "recent_3m_days"),
            recent_6m_days=_int(features, "recent_6m_days"),
            revenue_growth_fallback=_float(features, "revenue_growth_fallback"),
            revenue_growth_cap=_float(features, "revenue_growth_cap"),
            history_years=_int_list(features, "history_years"),
            trend_start_year=_int(features, "trend_start_year"),
            trend_end_year=_int(features, "trend_end_year"),
            returner_history_year=_int(features, "returner_history_year"),
            family_type_min_count=_int(features, "family_type_min_count"),
            purely_women_share=_float(features, "purely_women_share"),
            recent_collection_min_year=_int(features, "recent_collection_min_year"),
            old_stock_max_year=_int(features, "old_stock_max_year"),
            holiday_months=_int_list(features, "holiday_months"),
        ),
        plots=PlotConfig(
            dpi=_int(plots, "dpi"),
            target_hist_bins=_int(plots, "target_hist_bins"),
            target_log_hist_bins=_int(plots, "target_log_hist_bins"),
            customer_order_clip=_int(plots, "customer_order_clip"),
            customer_order_bins=_int(plots, "customer_order_bins"),
            customer_revenue_clip=_float(plots, "customer_revenue_clip"),
            customer_revenue_bins=_int(plots, "customer_revenue_bins"),
            top_type_count=_int(plots, "top_type_count"),
            top_brand_count=_int(plots, "top_brand_count"),
            top_season_count=_int(plots, "top_season_count"),
            recency_quantile_bins=_int(plots, "recency_quantile_bins"),
            monetary_quantile_bins=_int(plots, "monetary_quantile_bins"),
            frequency_bucket_bins=_float_list(plots, "frequency_bucket_bins"),
            frequency_bucket_labels=_str_list(plots, "frequency_bucket_labels"),
        ),
        modeling=ModelConfig(
            random_seed=_int(modeling, "random_seed"),
            validation_size=_float(modeling, "validation_size"),
            ridge_params=dict(_table(modeling, "ridge_params")),
            decision_tree_params=dict(_table(modeling, "decision_tree_params")),
            knn_params=dict(_table(modeling, "knn_params")),
            random_forest_params=dict(_table(modeling, "random_forest_params")),
            xgb_common_params=dict(_table(modeling, "xgb_common_params")),
            xgb_extended_params=dict(_table(modeling, "xgb_extended_params")),
            xgb_cv_params=dict(_table(modeling, "xgb_cv_params")),
            xgb_final_params=dict(_table(modeling, "xgb_final_params")),
            xgb_single_objective=_str(modeling, "xgb_single_objective"),
            xgb_tweedie_objective=_str(modeling, "xgb_tweedie_objective"),
            xgb_tweedie_params=dict(_table(modeling, "xgb_tweedie_params")),
            xgb_hurdle_classifier_objective=_str(modeling, "xgb_hurdle_classifier_objective"),
            xgb_hurdle_classifier_metric=_str(modeling, "xgb_hurdle_classifier_metric"),
            xgb_hurdle_regressor_objective=_str(modeling, "xgb_hurdle_regressor_objective"),
            xgb_mse_objective=_str(modeling, "xgb_mse_objective"),
            xgb_log_objective=_str(modeling, "xgb_log_objective"),
            xgb_pseudo_huber_objective=_str(modeling, "xgb_pseudo_huber_objective"),
            xgb_fallback_objective=_str(modeling, "xgb_fallback_objective"),
            zero_threshold_grid=_float_list(modeling, "zero_threshold_grid"),
            champion_n_splits=_int(modeling, "champion_n_splits"),
            champion_zero_threshold=_float(modeling, "champion_zero_threshold"),
        ),
        workflow=WorkflowConfig(
            show_plots=_bool(workflow, "show_plots"),
            save_plots=_bool(workflow, "save_plots"),
            save_features=_bool(workflow, "save_features"),
            run_course_baselines=_bool(workflow, "run_course_baselines"),
            run_xgboost_models=_bool(workflow, "run_xgboost_models"),
            run_extended_xgboost=_bool(workflow, "run_extended_xgboost"),
            run_cross_validation=_bool(workflow, "run_cross_validation"),
            run_submissions=_bool(workflow, "run_submissions"),
            include_knn=_bool(workflow, "include_knn"),
        ),
    )


def _read_toml(path: Path | str) -> dict[str, Any]:
    """Read a TOML file into a plain dictionary."""
    config_path = Path(path).expanduser()
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a local override dictionary into a base dictionary."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _table(data: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a required nested TOML table."""
    value = data.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Missing required config table: {key}")
    return value


def _str(data: dict[str, Any], key: str) -> str:
    """Return a required string config value."""
    value = data[key]
    if not isinstance(value, str):
        raise TypeError(f"Config value {key!r} must be a string.")
    return value


def _bool(data: dict[str, Any], key: str) -> bool:
    """Return a required boolean config value."""
    value = data[key]
    if not isinstance(value, bool):
        raise TypeError(f"Config value {key!r} must be a boolean.")
    return value


def _int(data: dict[str, Any], key: str) -> int:
    """Return a required integer config value."""
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Config value {key!r} must be an integer.")
    return value


def _float(data: dict[str, Any], key: str) -> float:
    """Return a required numeric config value as a float."""
    value = data[key]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"Config value {key!r} must be numeric.")
    return float(value)


def _str_list(data: dict[str, Any], key: str) -> list[str]:
    """Return a required list of strings."""
    values = data[key]
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise TypeError(f"Config value {key!r} must be a list of strings.")
    return values


def _int_list(data: dict[str, Any], key: str) -> list[int]:
    """Return a required list of integers."""
    values = data[key]
    if not isinstance(values, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in values
    ):
        raise TypeError(f"Config value {key!r} must be a list of integers.")
    return values


def _float_list(data: dict[str, Any], key: str) -> list[float]:
    """Return a required list of numeric values as floats."""
    values = data[key]
    if not isinstance(values, list) or not all(
        isinstance(value, int | float) and not isinstance(value, bool) for value in values
    ):
        raise TypeError(f"Config value {key!r} must be a list of numbers.")
    return [float(value) for value in values]


def _normalize_default_value(value: Any) -> Any:
    """Convert TOML default sentinels to Python values used by pandas."""
    if isinstance(value, str) and value.lower() in {"nan", "none", "null"}:
        return math.nan
    return value


def _resolve_path(path: Path | str, root: Path) -> Path:
    """Resolve absolute paths directly and relative paths against project root."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()

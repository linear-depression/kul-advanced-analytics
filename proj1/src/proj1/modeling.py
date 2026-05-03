"""Model preparation, validation, training, and submission helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from proj1.config import DataConfig, ModelConfig
from proj1.metrics import clip_revenue_predictions, evaluate_predictions, results_frame


@dataclass
class PreprocessedSets:
    """Container for imputed and scaled train/validation matrices."""
    X_train_imp: pd.DataFrame
    X_val_imp: pd.DataFrame
    X_train_scaled: pd.DataFrame
    X_val_scaled: pd.DataFrame
    imputer: SimpleImputer
    scaler: StandardScaler


@dataclass
class ModelRun:
    """Container for model results, fitted artifacts, and validation predictions."""
    results: pd.DataFrame
    artifacts: dict[str, Any] = field(default_factory=dict)
    predictions: dict[str, np.ndarray] = field(default_factory=dict)


def prepare_modeling_data(
    features: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    data_config: DataConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Align engineered features with train targets and test customers."""
    fm = features.copy()
    fm.index.name = data_config.customer_id_col
    X_train_full = fm.loc[train[data_config.customer_id_col]].copy()
    y_train_full = train.set_index(data_config.customer_id_col).loc[
        X_train_full.index,
        data_config.target_col,
    ]
    X_test = fm.loc[test[data_config.customer_id_col]].copy()

    if len(X_train_full) != len(y_train_full) or len(X_train_full) != len(train):
        raise ValueError("Train feature/target alignment failed.")
    if len(X_test) != len(test):
        raise ValueError("Test feature alignment failed.")
    if X_train_full.select_dtypes(include="object").shape[1] > 0:
        raise ValueError("Model matrix still contains object columns.")
    return X_train_full, y_train_full, X_test


def train_validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    config: ModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/validation split by returner status."""
    stratify = (y > 0).astype(int)
    return train_test_split(
        X,
        y,
        test_size=config.validation_size,
        stratify=stratify,
        random_state=config.random_seed,
    )


def make_preprocessed_sets(X_train: pd.DataFrame, X_val: pd.DataFrame) -> PreprocessedSets:
    """Fit median imputation and scaling for models that need them."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )
    X_val_imp = pd.DataFrame(
        imputer.transform(X_val),
        index=X_val.index,
        columns=X_val.columns,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imp),
        index=X_train.index,
        columns=X_train.columns,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val_imp),
        index=X_val.index,
        columns=X_val.columns,
    )
    return PreprocessedSets(X_train_imp, X_val_imp, X_train_scaled, X_val_scaled, imputer, scaler)


def run_sanity_baselines(y_train: pd.Series, y_val: pd.Series) -> pd.DataFrame:
    """Evaluate no-feature baseline predictions."""
    results = [
        evaluate_predictions(
            "A1. Predict 0 for everyone",
            y_val,
            np.zeros(len(y_val)),
            group="A. Sanity",
        ),
        evaluate_predictions(
            "A2. Predict mean",
            y_val,
            np.full(len(y_val), y_train.mean()),
            group="A. Sanity",
        ),
        evaluate_predictions(
            "A3. Predict median",
            y_val,
            np.full(len(y_val), y_train.median()),
            group="A. Sanity",
        ),
    ]
    return results_frame(results)


def run_course_baselines(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    config: ModelConfig,
    include_knn: bool = True,
) -> ModelRun:
    """Train and evaluate course-method baseline models."""
    prep = make_preprocessed_sets(X_train, X_val)
    results: list[dict[str, float | str]] = []
    artifacts: dict[str, Any] = {"preprocessing": prep}
    predictions: dict[str, np.ndarray] = {}

    ridge = Ridge(**_with_random_state(config.ridge_params, config.random_seed))
    ridge.fit(prep.X_train_scaled, y_train)
    pred = ridge.predict(prep.X_val_scaled)
    results.append(evaluate_predictions("B1. Ridge Regression", y_val, pred, "B. Course methods"))
    predictions["ridge"] = clip_revenue_predictions(pred)

    ridge_log = Ridge(**_with_random_state(config.ridge_params, config.random_seed))
    ridge_log.fit(prep.X_train_scaled, np.log1p(y_train))
    pred = np.expm1(ridge_log.predict(prep.X_val_scaled))
    results.append(evaluate_predictions("B1b. Ridge on log1p(y)", y_val, pred, "B. Course methods"))
    predictions["ridge_log"] = clip_revenue_predictions(pred)

    tree = DecisionTreeRegressor(
        **_with_random_state(config.decision_tree_params, config.random_seed)
    )
    tree.fit(prep.X_train_imp, y_train)
    pred = tree.predict(prep.X_val_imp)
    results.append(evaluate_predictions("B2. Decision Tree", y_val, pred, "B. Course methods"))
    predictions["decision_tree"] = clip_revenue_predictions(pred)

    if include_knn:
        knn = KNeighborsRegressor(**config.knn_params)
        knn.fit(prep.X_train_scaled, y_train)
        pred = knn.predict(prep.X_val_scaled)
        results.append(evaluate_predictions("B3. KNN", y_val, pred, "B. Course methods"))
        predictions["knn"] = clip_revenue_predictions(pred)

    forest = RandomForestRegressor(
        **_with_random_state(config.random_forest_params, config.random_seed)
    )
    forest.fit(prep.X_train_imp, y_train)
    pred = forest.predict(prep.X_val_imp)
    results.append(evaluate_predictions("B4. Random Forest", y_val, pred, "B. Course methods"))
    predictions["random_forest"] = clip_revenue_predictions(pred)

    artifacts.update(
        {
            "ridge": ridge,
            "ridge_log": ridge_log,
            "decision_tree": tree,
            "random_forest": forest,
        }
    )
    if include_knn:
        artifacts["knn"] = knn
    return ModelRun(results_frame(results), artifacts=artifacts, predictions=predictions)


def run_xgboost_suite(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    config: ModelConfig,
    include_extended: bool = True,
) -> ModelRun:
    """Train and evaluate the main XGBoost model family."""
    xgb = _import_xgboost()
    results: list[dict[str, float | str]] = []
    artifacts: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}

    common_params = _with_random_state(config.xgb_common_params, config.random_seed)

    xgb_c1 = _fit_xgb_regressor(
        xgb,
        X_train,
        y_train,
        X_val,
        y_val,
        objective=config.xgb_single_objective,
        params=common_params,
        fallback_objective=config.xgb_fallback_objective,
    )
    pred_c1 = clip_revenue_predictions(xgb_c1.predict(X_val))
    results.append(evaluate_predictions("C1. XGBoost single regression", y_val, pred_c1, "C. Main"))
    artifacts["xgb_c1"] = xgb_c1
    predictions["xgb_c1"] = pred_c1

    xgb_c2 = xgb.XGBRegressor(
        objective=config.xgb_tweedie_objective,
        **config.xgb_tweedie_params,
        **common_params,
    )
    xgb_c2.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred_c2 = clip_revenue_predictions(xgb_c2.predict(X_val))
    results.append(evaluate_predictions("C2. XGBoost Tweedie", y_val, pred_c2, "C. Main"))
    artifacts["xgb_c2"] = xgb_c2
    predictions["xgb_c2"] = pred_c2

    hurdle = train_two_stage_hurdle(xgb, X_train, X_val, y_train, y_val, common_params, config)
    results.append(
        evaluate_predictions("C3. Two-stage hurdle", y_val, hurdle["prediction"], "C. Main")
    )
    artifacts["hurdle"] = hurdle
    predictions["hurdle"] = hurdle["prediction"]

    if include_extended:
        extended = run_extended_xgboost_models(xgb, X_train, X_val, y_train, y_val, config)
        results.extend(extended.results.to_dict("records"))
        artifacts.update(extended.artifacts)
        predictions.update(extended.predictions)

    results_df = results_frame(results)
    return ModelRun(results_df, artifacts=artifacts, predictions=predictions)


def run_extended_xgboost_models(
    xgb,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    config: ModelConfig,
) -> ModelRun:
    """Train long-tail-aware XGBoost variants and post-processing checks."""
    results: list[dict[str, float | str]] = []
    artifacts: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    base_params = _with_random_state(config.xgb_extended_params, config.random_seed)

    d1 = xgb.XGBRegressor(objective=config.xgb_mse_objective, **base_params)
    d1.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred_d1 = clip_revenue_predictions(d1.predict(X_val))
    results.append(evaluate_predictions("D1. XGBoost MSE", y_val, pred_d1, "D. Long-tail"))
    artifacts["xgb_d1"] = d1
    predictions["xgb_d1"] = pred_d1

    d2 = xgb.XGBRegressor(objective=config.xgb_log_objective, **base_params)
    d2.fit(X_train, np.log1p(y_train), eval_set=[(X_val, np.log1p(y_val))], verbose=False)
    pred_d2 = clip_revenue_predictions(np.expm1(d2.predict(X_val)))
    results.append(evaluate_predictions("D2. XGBoost log1p MSE", y_val, pred_d2, "D. Long-tail"))
    artifacts["xgb_d2"] = d2
    predictions["xgb_d2"] = pred_d2

    d3 = xgb.XGBRegressor(objective=config.xgb_pseudo_huber_objective, **base_params)
    d3.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred_d3 = clip_revenue_predictions(d3.predict(X_val))
    results.append(evaluate_predictions("D3. XGBoost Pseudo-Huber", y_val, pred_d3, "D. Long-tail"))
    artifacts["xgb_d3"] = d3
    predictions["xgb_d3"] = pred_d3

    pred_c1 = predictions.get("xgb_c1")
    if pred_c1 is not None:
        threshold, pred_pp, mae = find_best_zero_threshold(
            pred_c1,
            y_val,
            config.zero_threshold_grid,
        )
        results.append(
            evaluate_predictions(f"F2. C1 post-processed (zero<{threshold})", y_val, pred_pp, "F. MAE")
        )
        artifacts["best_zero_threshold"] = threshold
        artifacts["best_zero_threshold_mae"] = mae
        predictions["xgb_c1_post_processed"] = pred_pp

    return ModelRun(results_frame(results), artifacts=artifacts, predictions=predictions)


def run_improved_xgboost_trials(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    config: ModelConfig,
    trial_params: dict[str, Any],
    tweedie_powers: list[float],
) -> ModelRun:
    """Train the improved Group C variants explored after the first diagnostics."""
    xgb = _import_xgboost()
    params = _with_random_state(trial_params, config.random_seed)
    results: list[dict[str, float | str]] = []
    artifacts: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}

    c1v2 = _fit_xgb_regressor(
        xgb,
        X_train,
        y_train,
        X_val,
        y_val,
        objective=config.xgb_single_objective,
        params=params,
        fallback_objective=config.xgb_fallback_objective,
    )
    pred = clip_revenue_predictions(c1v2.predict(X_val))
    results.append(evaluate_predictions("C1.v2 XGBoost MAE (deeper)", y_val, pred, "C. Main v2"))
    artifacts["xgb_c1v2"] = c1v2
    predictions["xgb_c1v2"] = pred

    for power in tweedie_powers:
        model = xgb.XGBRegressor(
            objective=config.xgb_tweedie_objective,
            tweedie_variance_power=power,
            eval_metric="mae",
            **params,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred = clip_revenue_predictions(model.predict(X_val))
        key = f"xgb_c2v{str(power).replace('.', '_')}"
        results.append(
            evaluate_predictions(
                f"C2.v Tweedie (power={power})",
                y_val,
                pred,
                "C. Main v2",
            )
        )
        artifacts[key] = model
        predictions[key] = pred

    hurdle = train_two_stage_hurdle(xgb, X_train, X_val, y_train, y_val, params, config)
    results.append(
        evaluate_predictions("C3.v2 Two-stage hurdle", y_val, hurdle["prediction"], "C. Main v2")
    )
    artifacts["hurdle_v2"] = hurdle
    predictions["hurdle_v2"] = hurdle["prediction"]
    return ModelRun(results_frame(results), artifacts=artifacts, predictions=predictions)


def run_mae_aligned_hurdle(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    config: ModelConfig,
    params: dict[str, Any],
    probability_thresholds: list[float],
) -> ModelRun:
    """Train a hurdle model that thresholds return probability for MAE."""
    xgb = _import_xgboost()
    xgb_params = _with_random_state(params, config.random_seed)
    y_train_binary = (y_train > 0).astype(int)
    y_val_binary = (y_val > 0).astype(int)

    clf_params = {**xgb_params, "eval_metric": config.xgb_hurdle_classifier_metric}
    clf = xgb.XGBClassifier(objective=config.xgb_hurdle_classifier_objective, **clf_params)
    clf.fit(X_train, y_train_binary, eval_set=[(X_val, y_val_binary)], verbose=False)
    prob_returner = clf.predict_proba(X_val)[:, 1]

    returner_mask = y_train > 0
    reg = _fit_xgb_regressor(
        xgb,
        X_train[returner_mask],
        y_train[returner_mask],
        X_val[y_val > 0],
        y_val[y_val > 0],
        objective=config.xgb_hurdle_regressor_objective,
        params=xgb_params,
        fallback_objective=config.xgb_fallback_objective,
    )
    pred_if_returner = clip_revenue_predictions(reg.predict(X_val))

    results: list[dict[str, float | str]] = []
    predictions: dict[str, np.ndarray] = {}
    threshold_rows = []
    for threshold in probability_thresholds:
        pred = np.where(prob_returner < threshold, 0, pred_if_returner)
        result = evaluate_predictions(
            f"F1. MAE-aligned hurdle (t={threshold})",
            y_val,
            pred,
            "F. MAE",
        )
        threshold_rows.append(
            {
                "threshold": threshold,
                "MAE": result["MAE"],
                "Spearman": result["Spearman"],
                "pct_zero": float((pred == 0).mean() * 100),
            }
        )
        results.append(result)
        predictions[f"mae_hurdle_t_{threshold}"] = clip_revenue_predictions(pred)

    threshold_table = pd.DataFrame(threshold_rows).sort_values("MAE")
    best_threshold = float(threshold_table.iloc[0]["threshold"])
    best_prediction = predictions[f"mae_hurdle_t_{best_threshold}"]
    return ModelRun(
        results_frame(results),
        artifacts={
            "classifier": clf,
            "regressor": reg,
            "prob_returner": prob_returner,
            "pred_if_returner": pred_if_returner,
            "threshold_table": threshold_table,
            "best_threshold": best_threshold,
            "classifier_auc": float(roc_auc_score(y_val_binary, prob_returner)),
        },
        predictions={"mae_aligned_hurdle": best_prediction, **predictions},
    )


def train_two_stage_hurdle(
    xgb,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    params: dict[str, Any],
    config: ModelConfig,
) -> dict[str, Any]:
    """Train a returner classifier and conditional revenue regressor."""
    clf_params = {**params, "eval_metric": config.xgb_hurdle_classifier_metric}
    clf = xgb.XGBClassifier(objective=config.xgb_hurdle_classifier_objective, **clf_params)
    y_train_binary = (y_train > 0).astype(int)
    y_val_binary = (y_val > 0).astype(int)
    clf.fit(X_train, y_train_binary, eval_set=[(X_val, y_val_binary)], verbose=False)
    prob_returner = clf.predict_proba(X_val)[:, 1]

    returner_mask = y_train > 0
    val_returner_mask = y_val > 0
    reg = _fit_xgb_regressor(
        xgb,
        X_train[returner_mask],
        np.log1p(y_train[returner_mask]),
        X_val[val_returner_mask],
        np.log1p(y_val[val_returner_mask]),
        objective=config.xgb_hurdle_regressor_objective,
        params=params,
        fallback_objective=config.xgb_fallback_objective,
    )
    pred_log = reg.predict(X_val)
    pred_if_returner = clip_revenue_predictions(np.expm1(pred_log))
    prediction = clip_revenue_predictions(prob_returner * pred_if_returner)
    return {
        "classifier": clf,
        "regressor": reg,
        "prob_returner": prob_returner,
        "prediction_if_returner": pred_if_returner,
        "prediction": prediction,
    }


def find_best_zero_threshold(
    predictions,
    y_true,
    thresholds: list[float],
) -> tuple[float, np.ndarray, float]:
    """Find the prediction floor threshold with the lowest validation MAE."""
    from sklearn.metrics import mean_absolute_error

    base = clip_revenue_predictions(predictions)
    best_threshold = 0.0
    best_prediction = base.copy()
    best_mae = float(mean_absolute_error(y_true, base))
    for threshold in thresholds:
        pred = base.copy()
        pred[pred < threshold] = 0
        mae = float(mean_absolute_error(y_true, pred))
        if mae < best_mae:
            best_threshold = float(threshold)
            best_prediction = pred
            best_mae = mae
    return best_threshold, best_prediction, best_mae


def cross_validate_champion(
    X: pd.DataFrame,
    y: pd.Series,
    config: ModelConfig,
    n_splits: int | None = None,
    zero_threshold: float | None = None,
) -> pd.DataFrame:
    """Run stratified cross-validation for the champion XGBoost model."""
    xgb = _import_xgboost()
    params = _with_random_state(config.xgb_cv_params, config.random_seed)
    n_splits = n_splits or config.champion_n_splits
    zero_threshold = zero_threshold if zero_threshold is not None else config.champion_zero_threshold
    strat_label = (y > 0).astype(int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_seed)
    rows = []
    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(X, strat_label), 1):
        started_at = time.time()
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        model = xgb.XGBRegressor(objective=config.xgb_single_objective, **params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred_raw = clip_revenue_predictions(model.predict(X_va))
        pred_pp = pred_raw.copy()
        pred_pp[pred_pp < zero_threshold] = 0
        raw = evaluate_predictions("raw", y_va, pred_raw)
        processed = evaluate_predictions("post_processed", y_va, pred_pp)
        rows.append(
            {
                "fold": fold_i,
                "n_train": len(X_tr),
                "n_val": len(X_va),
                "mae_raw": raw["MAE"],
                "mae_post_processed": processed["MAE"],
                "spearman_post_processed": processed["Spearman"],
                "seconds": time.time() - started_at,
            }
        )
    return pd.DataFrame(rows)


def train_final_champion(
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    X_test: pd.DataFrame,
    config: ModelConfig,
    zero_threshold: float | None = None,
) -> tuple[Any, np.ndarray]:
    """Train the final champion model on all training rows and predict test revenue."""
    xgb = _import_xgboost()
    params = _with_random_state(config.xgb_final_params, config.random_seed)
    zero_threshold = zero_threshold if zero_threshold is not None else config.champion_zero_threshold
    model = xgb.XGBRegressor(objective=config.xgb_single_objective, **params)
    model.fit(X_train_full, y_train_full, verbose=False)
    predictions = clip_revenue_predictions(model.predict(X_test))
    predictions[predictions < zero_threshold] = 0
    return model, predictions


def make_submission(
    test: pd.DataFrame,
    predictions,
    data_config: DataConfig,
) -> pd.DataFrame:
    """Create a leaderboard submission table from test customers and predictions."""
    return pd.DataFrame(
        {
            data_config.customer_id_col: test[data_config.customer_id_col],
            data_config.prediction_col: clip_revenue_predictions(predictions),
        }
    )


def save_submission(submission: pd.DataFrame, path: Path | str) -> Path:
    """Write a submission CSV and return the output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path


def _with_random_state(params: dict[str, Any], random_state: int) -> dict[str, Any]:
    """Return model parameters with the configured random state included."""
    return {**params, "random_state": random_state}


def _fit_xgb_regressor(
    xgb,
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame,
    y_val,
    objective: str,
    params: dict[str, Any],
    fallback_objective: str,
):
    """Fit an XGBoost regressor with a square-error fallback objective."""
    try:
        model = xgb.XGBRegressor(objective=objective, **params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model
    except Exception:
        model = xgb.XGBRegressor(objective=fallback_objective, **params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model


def _import_xgboost():
    """Import xgboost and raise an actionable dependency error if unavailable."""
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for the main models. Install dependencies with `uv sync`."
        ) from exc
    return xgb

"""Reporting diagnostics used by the project notebook."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score

from proj1.metrics import clip_revenue_predictions


def prediction_distribution(
    predictions: Sequence[float],
    cutoffs: Sequence[float],
) -> pd.Series:
    """Summarize the distribution of model predictions."""
    pred = clip_revenue_predictions(predictions)
    values = {
        "min": float(pred.min()),
        "median": float(np.median(pred)),
        "mean": float(pred.mean()),
        "max": float(pred.max()),
        "pct_zero": float((pred == 0).mean() * 100),
    }
    for cutoff in cutoffs:
        values[f"pct_lt_{cutoff:g}"] = float((pred < cutoff).mean() * 100)
    return pd.Series(values)


def prediction_distribution_table(
    predictions: Mapping[str, Sequence[float]],
    cutoffs: Sequence[float],
) -> pd.DataFrame:
    """Build a side-by-side prediction distribution table for several models."""
    rows = []
    for name, values in predictions.items():
        row = prediction_distribution(values, cutoffs)
        row["model"] = name
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def error_decomposition(
    y_true: Sequence[float],
    predictions: Sequence[float],
    positive_threshold: float,
) -> pd.Series:
    """Split absolute error between true zero and positive customers."""
    y = np.asarray(y_true)
    pred = clip_revenue_predictions(predictions)
    abs_err = np.abs(y - pred)
    zero_mask = y <= positive_threshold
    positive_mask = y > positive_threshold
    return pd.Series(
        {
            "mae_true_zero": float(abs_err[zero_mask].mean()) if zero_mask.any() else np.nan,
            "mae_true_positive": float(abs_err[positive_mask].mean())
            if positive_mask.any()
            else np.nan,
            "error_contribution_zero": float(abs_err[zero_mask].sum() / len(y)),
            "error_contribution_positive": float(abs_err[positive_mask].sum() / len(y)),
        }
    )


def error_by_actual_thresholds(
    y_true: Sequence[float],
    predictions: Sequence[float],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    """Measure how much high-actual-revenue customers contribute to MAE."""
    y = np.asarray(y_true)
    pred = clip_revenue_predictions(predictions)
    abs_err = np.abs(y - pred)
    total_error = abs_err.sum()
    rows = []
    for threshold in thresholds:
        mask = y > threshold
        rows.append(
            {
                "threshold": threshold,
                "n_above": int(mask.sum()),
                "pct_above": float(mask.mean() * 100),
                "error_sum": float(abs_err[mask].sum()),
                "pct_total_error": float(abs_err[mask].sum() / total_error * 100)
                if total_error > 0
                else np.nan,
                "mae_if_excluded": float(abs_err[~mask].mean()) if (~mask).any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calibration_by_actual_bins(
    y_true: Sequence[float],
    predictions: Sequence[float],
    bins: Sequence[float],
) -> pd.DataFrame:
    """Compare actual and predicted revenue by actual-revenue bins."""
    y = np.asarray(y_true)
    pred = clip_revenue_predictions(predictions)
    actual_bin = pd.cut(y, bins=bins, include_lowest=True)
    df = pd.DataFrame({"actual_bin": actual_bin, "actual": y, "predicted": pred})
    return df.groupby("actual_bin", observed=False).agg(
        n=("actual", "size"),
        actual_mean=("actual", "mean"),
        pred_mean=("predicted", "mean"),
        pred_max=("predicted", "max"),
    )


def feature_importance_table(model, feature_names: Sequence[str], top_n: int | None = None) -> pd.DataFrame:
    """Return model feature importances sorted from largest to smallest."""
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("model does not expose feature_importances_.")
    table = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    if top_n is not None:
        table = table.head(top_n)
    return table.reset_index(drop=True)


def variance_by_prediction_bin(
    y_true: Sequence[float],
    predictions: Sequence[float],
    positive_threshold: float,
    q: int,
) -> pd.DataFrame:
    """Summarize actual returner revenue within prediction quantile bins."""
    y = np.asarray(y_true)
    pred = clip_revenue_predictions(predictions)
    mask = y > positive_threshold
    bins = pd.qcut(pred[mask], q=q, duplicates="drop")
    df = pd.DataFrame({"pred_bin": bins, "actual": y[mask]})
    return df.groupby("pred_bin", observed=False)["actual"].agg(["mean", "std", "count"])


def high_value_feature_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    quantile: float,
) -> pd.DataFrame:
    """Compare selected feature means for the highest-revenue customers vs the rest."""
    threshold = y.quantile(quantile)
    high = y > threshold
    rows = []
    for feature in features:
        high_mean = X.loc[high, feature].mean()
        rest_mean = X.loc[~high, feature].mean()
        rows.append(
            {
                "feature": feature,
                "top_mean": high_mean,
                "rest_mean": rest_mean,
                "ratio": high_mean / rest_mean if rest_mean != 0 else np.nan,
            }
        )
    return pd.DataFrame(rows), float(threshold)


def high_value_auc_experiment(
    probabilities: Mapping[float, Sequence[float]],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Score precomputed high-value classifier probabilities with AUC."""
    y = np.asarray(y_true)
    rows = []
    for threshold, prob in probabilities.items():
        label = (y > threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "positive_rate": float(label.mean()),
                "auc": float(roc_auc_score(label, prob)),
            }
        )
    return pd.DataFrame(rows)


def zero_threshold_table(
    predictions: Sequence[float],
    y_true: Sequence[float],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    """Evaluate the effect of zeroing out small predictions."""
    base = clip_revenue_predictions(predictions)
    y = np.asarray(y_true)
    rows = []
    for threshold in thresholds:
        pred = base.copy()
        pred[pred < threshold] = 0
        spear, _ = spearmanr(y, pred) if (pred != pred[0]).any() else (np.nan, None)
        rows.append(
            {
                "threshold": threshold,
                "mae": float(mean_absolute_error(y, pred)),
                "spearman": float(spear) if not np.isnan(spear) else np.nan,
                "extra_zeros": int(((base > 0) & (pred == 0)).sum()),
            }
        )
    table = pd.DataFrame(rows)
    table["is_best"] = table["mae"] == table["mae"].min()
    return table


def submission_summary(submissions: Mapping[str, pd.DataFrame], prediction_col: str) -> pd.DataFrame:
    """Summarize prediction distributions for submission data frames."""
    rows = []
    for name, submission in submissions.items():
        pred = clip_revenue_predictions(submission[prediction_col])
        rows.append(
            {
                "submission": name,
                "mean": float(pred.mean()),
                "median": float(np.median(pred)),
                "max": float(pred.max()),
                "pct_zero": float((pred == 0).mean() * 100),
                "n_rows": len(submission),
            }
        )
    return pd.DataFrame(rows).set_index("submission")

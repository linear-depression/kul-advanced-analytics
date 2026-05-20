"""Evaluation metrics used by the revenue prediction workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error


def clip_revenue_predictions(y_pred) -> np.ndarray:
    """Revenue cannot be negative, so clip predictions at zero."""
    return np.clip(np.asarray(y_pred, dtype=float), 0, None)


def evaluate_predictions(
    name: str,
    y_true,
    y_pred,
    group: str = "",
    clip: bool = True,
) -> dict[str, float | str]:
    """Evaluate clipped revenue predictions with MAE and Spearman correlation."""
    predictions = clip_revenue_predictions(y_pred) if clip else np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, predictions))
    spear, _ = spearmanr(y_true, predictions)
    return {
        "group": group,
        "model": name,
        "MAE": mae,
        "Spearman": float(spear) if pd.notna(spear) else np.nan,
    }


def results_frame(results: list[dict[str, float | str]]) -> pd.DataFrame:
    """Return model results sorted by MAE."""
    if not results:
        return pd.DataFrame(columns=["group", "model", "MAE", "Spearman"])
    return pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)


def describe_target(y, name: str) -> dict[str, object]:
    """Summarize a target vector for train/validation split checks."""
    y_series = pd.Series(y)
    return {
        "split": name,
        "n_customers": len(y_series),
        "pct_returners": f"{(y_series > 0).mean() * 100:.2f}%",
        "mean": round(float(y_series.mean()), 2),
        "median": round(float(y_series.median()), 2),
        "std": round(float(y_series.std()), 2),
        "max": round(float(y_series.max()), 2),
        "p90": round(float(y_series.quantile(0.90)), 2),
        "p95": round(float(y_series.quantile(0.95)), 2),
        "p99": round(float(y_series.quantile(0.99)), 2),
    }

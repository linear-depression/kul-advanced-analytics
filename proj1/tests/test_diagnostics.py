from __future__ import annotations

import numpy as np
import pandas as pd

from proj1.diagnostics import (
    calibration_by_actual_bins,
    error_by_actual_thresholds,
    prediction_distribution,
    submission_summary,
    zero_threshold_table,
)


def test_prediction_distribution_uses_explicit_cutoffs():
    """Verify prediction summaries use caller-provided thresholds."""
    summary = prediction_distribution([-1, 0, 4, 10], cutoffs=[5, 20])

    assert summary["min"] == 0
    assert summary["pct_zero"] == 50
    assert summary["pct_lt_5"] == 75
    assert summary["pct_lt_20"] == 100


def test_error_by_actual_thresholds_quantifies_tail_contribution():
    """Verify high-revenue threshold diagnostics report error contribution."""
    table = error_by_actual_thresholds(
        y_true=[0, 100, 250, 500],
        predictions=[0, 80, 100, 200],
        thresholds=[200],
    )

    assert table.loc[0, "n_above"] == 2
    assert np.isclose(table.loc[0, "pct_above"], 50.0)
    assert table.loc[0, "error_sum"] == 450


def test_zero_threshold_table_marks_best_threshold():
    """Verify zero-threshold search returns one best row."""
    table = zero_threshold_table(
        predictions=[0, 1, 20],
        y_true=[0, 0, 20],
        thresholds=[0, 5],
    )

    assert table["is_best"].sum() == 1
    assert table.loc[table["is_best"], "threshold"].iloc[0] == 5


def test_calibration_and_submission_summary_shapes(project_config):
    """Verify calibration and submission summaries produce review tables."""
    calibration = calibration_by_actual_bins(
        y_true=[0, 25, 75, 150],
        predictions=[0, 20, 50, 100],
        bins=[-1, 0, 50, 100, 200],
    )
    submission = pd.DataFrame(
        {
            project_config.data.customer_id_col: [1, 2],
            project_config.data.prediction_col: [0.0, 10.0],
        }
    )
    summary = submission_summary({"v01": submission}, project_config.data.prediction_col)

    assert list(calibration.columns) == ["n", "actual_mean", "pred_mean", "pred_max"]
    assert summary.loc["v01", "n_rows"] == 2

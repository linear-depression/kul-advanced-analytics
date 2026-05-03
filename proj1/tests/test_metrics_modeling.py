from __future__ import annotations

import numpy as np

from proj1.features import build_customer_features
from proj1.metrics import evaluate_predictions
from proj1.modeling import make_preprocessed_sets, prepare_modeling_data, train_validation_split


def test_evaluate_predictions_clips_negative_values():
    """Verify evaluation clips impossible negative revenue predictions."""
    result = evaluate_predictions("toy", [0, 10, 20], [-5, 12, 18])
    assert result["MAE"] == np.mean([0, 2, 2])


def test_modeling_preparation_and_split(make_train_test_transactions, project_config):
    """Verify modeling matrices, split, and preprocessing are aligned."""
    train, test, transactions = make_train_test_transactions
    features = build_customer_features(
        train,
        test,
        transactions,
        data_config=project_config.data,
        feature_config=project_config.features,
        save=False,
    )

    X_train_full, y_train_full, X_test = prepare_modeling_data(
        features,
        train,
        test,
        project_config.data,
    )
    config = project_config.modeling
    X_train, X_val, _, _ = train_validation_split(X_train_full, y_train_full, config)
    prep = make_preprocessed_sets(X_train, X_val)

    assert X_train_full.shape[0] == len(train)
    assert X_test.shape[0] == len(test)
    assert prep.X_train_imp.isna().sum().sum() == 0
    assert prep.X_val_scaled.shape == X_val.shape

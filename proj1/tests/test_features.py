from __future__ import annotations

from proj1.config import ProjectPaths
from proj1.features import build_customer_features, feature_quality_report, parse_season_year


def test_parse_season_year_handles_common_values():
    """Verify product season labels map to collection years."""
    assert parse_season_year("W17") == 2017
    assert parse_season_year("S99") == 1999


def test_build_customer_features_returns_numeric_matrix(
    tmp_path,
    make_train_test_transactions,
    project_config,
):
    """Verify feature engineering returns a numeric customer matrix."""
    train, test, transactions = make_train_test_transactions
    paths = ProjectPaths(
        root=tmp_path,
        data_dir=tmp_path / "data",
        features_dir=tmp_path / "data" / "features",
        plots_dir=tmp_path / "plots",
        submissions_dir=tmp_path / "submissions",
        report_dir=tmp_path / "report",
    )

    features = build_customer_features(
        train,
        test,
        transactions,
        data_config=project_config.data,
        feature_config=project_config.features,
        paths=paths,
        save=True,
    )
    report = feature_quality_report(features, train, test, project_config.data)

    assert set(train["cust_id"]).issubset(features.index)
    assert set(test["cust_id"]).issubset(features.index)
    assert "dominant_type_1_women" in features.columns
    assert "dominant_type_1_unknown" not in features.columns
    assert features.select_dtypes(include="object").empty
    assert report["train_in_fm"] == len(train)
    assert paths.clean_feature_file(project_config.data).exists()

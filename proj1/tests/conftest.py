from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from proj1.config import load_project_config


@pytest.fixture
def project_config():
    """Load the versioned project config used by tests."""
    return load_project_config(Path(__file__).resolve().parents[1] / "config" / "default.toml")


@pytest.fixture
def make_train_test_transactions():
    """Return small synthetic train, test, and transaction tables for tests."""
    train = pd.DataFrame(
        {
            "cust_id": [1, 2, 3, 4, 5, 6],
            "revenue_2018_2019": [0.0, 120.0, 0.0, 80.0, 250.0, 0.0],
        }
    )
    test = pd.DataFrame({"cust_id": [7, 8]})
    rows = []
    for cust_id in range(1, 9):
        for order_offset in range(2):
            sale_id = cust_id * 100 + order_offset
            rows.append(
                {
                    "cust_id": cust_id,
                    "sale_id": sale_id,
                    "prod_id": 1000 + sale_id,
                    "order_date": pd.Timestamp("2017-01-15")
                    + pd.Timedelta(days=30 * order_offset + cust_id),
                    "pack_date": pd.Timestamp("2017-01-16")
                    + pd.Timedelta(days=30 * order_offset + cust_id),
                    "sale_revenue": 50.0 + cust_id * 2 - order_offset,
                    "sale_discount_applied": -5.0 if order_offset == 1 else 0.0,
                    "returned_to_shop_id": None,
                    "prod_size": "38",
                    "prod_brand": ["Gabor", "Nike", "Other"][cust_id % 3],
                    "prod_color": "black",
                    "prod_season": "W17",
                    "prod_type_1": ["women", "men", "boys", "girls"][cust_id % 4],
                    "prod_type_3": "shoe",
                    "prod_type_4": "casual",
                    "prod_type_5": "standard",
                    "prod_heel": "flat",
                    "prod_material": "leather",
                    "prod_print": None,
                    "prod_clasp": None,
                    "prod_comfort_sole": "soft",
                    "prod_comfort_wear": None,
                    "prod_web_only": 0,
                    "prod_outlet": 0,
                }
            )
    transactions = pd.DataFrame(rows)
    return train, test, transactions

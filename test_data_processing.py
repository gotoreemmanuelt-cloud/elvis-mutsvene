import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from modules.sample_data_generator import generate_sample_dataset
from modules.data_processing import compute_features, validate_and_clean


def test_generate_dataset_columns():
    df = generate_sample_dataset(n_months=2, seed=1)
    assert len(df) > 0
    for col in ["date", "direction", "amount", "category", "counterparty", "channel", "balance_after"]:
        assert col in df.columns


def test_compute_features_keys():
    df = generate_sample_dataset(n_months=3, seed=2)
    f = compute_features(df)
    for key in ["months", "total_inflow", "income_cv", "regularity", "repayment_count",
                "cash_out_ratio", "savings_slope", "avg_monthly_inflow"]:
        assert key in f
    assert f["months"] >= 2
    assert 0 <= f["regularity"] <= 1


def test_validate_missing_column():
    try:
        validate_and_clean(pd.DataFrame({"date": []}))
        assert False, "should raise"
    except ValueError:
        pass

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.sample_data_generator import generate_sample_dataset
from modules.fraud_detection import detect_risk_flags, fraud_penalty


def test_detect_returns_list():
    df = generate_sample_dataset(n_months=3, seed=11)
    flags = detect_risk_flags(df)
    assert isinstance(flags, list)
    assert all(isinstance(f, str) for f in flags)


def test_penalty_capped():
    assert fraud_penalty(["a", "b", "c", "d", "e", "f", "g"]) <= 0.4

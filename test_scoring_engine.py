import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.sample_data_generator import generate_sample_dataset
from modules.data_processing import compute_features
from modules.scoring_engine import score_applicant, recommend, WEIGHTS


def test_score_bounds():
    df = generate_sample_dataset(n_months=4, seed=7)
    f = compute_features(df)
    s = score_applicant(f, 0.0)
    assert 0 <= s["score"] <= 1000
    assert set(s["breakdown"].keys()) == set(WEIGHTS.keys())
    assert sum(s["breakdown"].values()) <= 1000


def test_fraud_penalty_lowers_score():
    df = generate_sample_dataset(n_months=4, seed=3)
    f = compute_features(df)
    clean = score_applicant(f, 0.0)["score"]
    penalised = score_applicant(f, 0.3)["score"]
    assert penalised < clean


def test_recommend_contract():
    df = generate_sample_dataset(n_months=4, seed=5)
    f = compute_features(df)
    for decision in ["APPROVE", "REVIEW", "DECLINE"]:
        pass
    rec = recommend(700, [], f)
    assert rec["decision"] == "APPROVE" and rec["loan_amount"] > 0
    rec = recommend(200, ["fraud signal"], f)
    assert rec["decision"] == "DECLINE" and rec["loan_amount"] == 0

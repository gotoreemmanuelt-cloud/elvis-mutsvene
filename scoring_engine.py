"""Explainable 0-1000 Trust Score engine for ScoreYedu."""
from __future__ import annotations

WEIGHTS = {"income_consistency": 250, "cash_flow_health": 250,
           "financial_discipline": 250, "repayment_history": 250}


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _income_consistency(f: dict) -> float:
    cv_score = 1 - _clamp(f["income_cv"], 0, 1)        # low CV = consistent
    return _clamp(0.6 * cv_score + 0.4 * f["regularity"])


def _cash_flow_health(f: dict) -> float:
    if f["months"] == 0:
        return 0.0
    positive = f["net_positive_months"] / f["months"]
    outflow_ratio = _clamp(f["total_outflow"] / f["total_inflow"], 0, 1.5) if f["total_inflow"] > 0 else 1.5
    sustainability = _clamp(1 - abs(1 - outflow_ratio) / 1.5)
    return _clamp(0.6 * positive + 0.4 * sustainability)


def _financial_discipline(f: dict) -> float:
    savings = _clamp(0.5 + f["savings_slope"] / max(abs(f["avg_monthly_inflow"]), 1), 0, 1)
    airtime_penalty = _clamp(1 - f["airtime_ratio"] * 2)
    night_penalty = _clamp(1 - f["night_share"] * 2)
    cash_penalty = _clamp(1 - max(0, f["cash_out_ratio"] - 0.5) * 2)
    return _clamp(0.4 * savings + 0.2 * airtime_penalty + 0.2 * night_penalty + 0.2 * cash_penalty)


def _repayment_history(f: dict) -> float:
    if f["repayment_count"] == 0:
        return 0.35  # no history: neutral-low
    regularity = 1.0 if f["repayment_regular"] else 0.5
    return _clamp(0.5 + 0.5 * regularity)


def score_applicant(features: dict, fraud_penalty: float = 0.0) -> dict:
    parts = {
        "income_consistency": _income_consistency(features),
        "cash_flow_health": _cash_flow_health(features),
        "financial_discipline": _financial_discipline(features),
        "repayment_history": _repayment_history(features),
    }
    raw = sum(parts[k] * WEIGHTS[k] for k in parts)
    final = int(round(_clamp(raw - fraud_penalty * 1000, 0, 1000)))
    return dict(score=final, breakdown={k: int(round(parts[k] * WEIGHTS[k])) for k in parts})


def recommend(score: int, risk_flags: list[str], features: dict) -> dict:
    severe = any("fraud" in f.lower() or "structuring" in f.lower() for f in risk_flags)
    if score >= 650 and not severe:
        decision = "APPROVE"
    elif score >= 450 and not severe:
        decision = "REVIEW"
    else:
        decision = "DECLINE"

    monthly_surplus = max(features["avg_monthly_inflow"] - features["avg_monthly_outflow"], 0)
    base = max(monthly_surplus * 3, features["avg_monthly_inflow"] * 0.5)
    if decision == "APPROVE":
        amount = round(base * 1.0, -1)
        period_weeks = 12
    elif decision == "REVIEW":
        amount = round(base * 0.5, -1)
        period_weeks = 8
    else:
        amount = 0
        period_weeks = 0
    amount = max(0, min(amount, 5000))
    return dict(decision=decision, loan_amount=float(amount),
                repayment_weeks=period_weeks,
                rationale=f"Score {score}/1000 with {len(risk_flags)} risk flag(s).")

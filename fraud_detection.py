"""Risk / fraud indicators: rules + IsolationForest anomaly detection."""
from __future__ import annotations
import pandas as pd


def detect_risk_flags(df: pd.DataFrame) -> list[str]:
    df = df.sort_values("date")
    flags = []
    inflows = df[df["direction"] == "in"]
    if len(inflows) and len(inflows) >= 4:
        mean_in = inflows["amount"].mean()
        big = inflows[inflows["amount"] > 5 * mean_in]
        if len(big) > 0:
            flags.append(f"Sudden large inflows detected ({len(big)} transaction(s))")
    # round-amount structuring
    round_tx = df[(df["amount"] % 50 == 0) & (df["amount"] > 0)]
    if len(round_tx) > max(5, 0.15 * len(df)):
        flags.append("Frequent round-amount transfers (possible structuring)")
    # high velocity
    daily = df.groupby(df["date"].dt.date).size()
    if len(daily) and daily.max() > 25:
        flags.append("Unusually high daily transaction velocity")
    # night activity
    if "night" in df and df["night"].mean() > 0.3:
        flags.append("High share of late-night transactions")
    # duplicate references
    if "reference" in df and df["reference"].duplicated().any():
        flags.append("Duplicate transaction references")
    # concentration to single counterparty on outflows
    outs = df[df["direction"] == "out"]
    if len(outs) and outs["counterparty"].value_counts(normalize=True).iloc[0] > 0.6:
        flags.append("Heavy concentration of outflows to one counterparty")
    # negative balance events
    if (df["balance_after"] < 0).any():
        flags.append("Negative wallet balance events")

    # ML anomaly detection on amounts
    try:
        from sklearn.ensemble import IsolationForest
        if len(df) >= 12:
            X = df[["amount", "balance_after"]].fillna(0).values
            preds = IsolationForest(contamination=0.05, random_state=0).fit_predict(X)
            frac = (preds == -1).mean()
            if frac > 0.08:
                flags.append(f"ML anomaly score: {frac:.0%} of transactions look anomalous")
    except Exception:
        pass
    return flags


def fraud_penalty(flags: list[str]) -> float:
    return min(0.4, 0.08 * len(flags))

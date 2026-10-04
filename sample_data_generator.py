"""Generate a synthetic EcoCash/Paynow-style transaction dataset for demos."""
from __future__ import annotations
import random
import pandas as pd
import numpy as np

MERCHANTS = ["Mbare Produce Market", "Mbare Produce Market", "Cross-border bus fare",
             "Part-time crochet sales", "University dining hall kiosk", "Manicaland tobacco",
             "Chitungwiza carpentry", "Avondale street vending", "Mutare hardware supply",
             "Bulawayo textile market", "Harare CBD hair salon", "Gweru second-hand clothing"]

CUSTOMERS = ["C. Dube", "M. Chikafu", "R. Moyo", "T. Ncube", "K. Sibanda", "J. Mwale",
             "S. Chari", "P. Gondo", "A. Tsvangirai", "N. Khumalo", "N. Dube", "E. Mutasa"]


def generate_sample_dataset(n_months: int = 6, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-05")
    rows = []
    balance = float(np_rng.uniform(150, 400))
    profile = np_rng.choice(["stable", "spiky", "disciplined"], p=[0.5, 0.3, 0.2])
    base_income = float(np_rng.uniform(180, 650))

    for day in pd.date_range(start, periods=n_months * 30, freq="D"):
        # customer payments / sale income
        for _ in range(rng.randint(0, 3)):
            amt = float(np_rng.uniform(5, base_income / 6))
            if profile == "spiky" and rng.random() < 0.08:
                amt *= float(np_rng.uniform(4, 8))  # sudden large inflow (risk flag)
            balance += amt
            rows.append(dict(date=day, direction="in", amount=round(amt, 2),
                             category="sale_income", counterparty=rng.choice(CUSTOMERS),
                             channel="EcoCash", balance_after=round(balance, 2),
                             night=int(day.hour >= 20)))
        # cash withdrawals
        if rng.random() < 0.55:
            amt = round(float(np_rng.uniform(10, 60)), 2)
            balance = max(0, balance - amt)
            rows.append(dict(date=day, direction="out", amount=amt, category="cash_out",
                             counterparty="EcoCash Agent", channel="EcoCash",
                             balance_after=round(balance, 2), night=int(rng.random() < 0.15)))
        # airtime
        if rng.random() < 0.4:
            amt = round(float(np_rng.uniform(1, 12)), 2)
            balance = max(0, balance - amt)
            rows.append(dict(date=day, direction="out", amount=amt, category="airtime_purchase",
                             counterparty="NetOne", channel="EcoCash",
                             balance_after=round(balance, 2), night=0))
        # merchant / supplier payments
        if rng.random() < 0.3:
            amt = round(float(np_rng.uniform(15, 120)), 2)
            balance = max(0, balance - amt)
            rows.append(dict(date=day, direction="out", amount=amt, category="merchant_payment",
                             counterparty=rng.choice(MERCHANTS), channel="Paynow",
                             balance_after=round(balance, 2), night=0))
        # loan repayments (weekly)
        if day.weekday() == 4 and rng.random() < 0.6:
            amt = round(float(np_rng.uniform(8, 25)), 2)
            balance = max(0, balance - amt)
            rows.append(dict(date=day, direction="out", amount=amt, category="loan_repayment",
                             counterparty="MicroLoan ZW", channel="Paynow",
                             balance_after=round(balance, 2), night=0))
        # occasional peer inflow
        if rng.random() < 0.25:
            amt = round(float(np_rng.uniform(5, 80)), 2)
            balance += amt
            rows.append(dict(date=day, direction="in", amount=amt, category="peer_transfer_in",
                             counterparty=rng.choice(CUSTOMERS), channel="EcoCash",
                             balance_after=round(balance, 2), night=int(rng.random() < 0.1)))

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["reference"] = [f"TX{100000 + i}" for i in range(len(df))]
    return df


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "..", "data", "sample_ecocash_transactions.csv")
    df = generate_sample_dataset()
    df.to_csv(os.path.abspath(out), index=False)
    print(f"Wrote {len(df)} rows to {out}")

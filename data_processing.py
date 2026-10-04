"""Data loading, validation and feature engineering for ScoreYedu."""
from __future__ import annotations
import io
import re
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = ["date", "direction", "amount", "category", "counterparty",
                    "channel", "balance_after"]


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["balance_after"] = pd.to_numeric(df["balance_after"], errors="coerce")
    df = df.dropna(subset=["date", "amount", "balance_after"])
    df["direction"] = df["direction"].astype(str).str.lower().str.strip()
    df["category"] = df["category"].astype(str).str.lower().str.strip()
    if "night" not in df.columns:
        df["night"] = 0
    return df.sort_values("date").reset_index(drop=True)


def load_csv(file) -> pd.DataFrame:
    return validate_and_clean(pd.read_csv(file))


def parse_pdf(file) -> pd.DataFrame:
    """Best-effort extraction of transaction lines from a PDF statement.

    Expects lines like: 2026-03-14, in, 45.00, sale_income, C. Dube, EcoCash, 220.10
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("pypdf is required for PDF statements: pip install pypdf") from e

    reader = PdfReader(file)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    rows = []
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2})[,\s]+(in|out)[,\s]+([\d.]+)[,\s]+([\w_]+)[,\s]+([\w\s\.\-]+)[,\s]+([\w]+)[,\s]+([\d.]+)",
        re.IGNORECASE)
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            rows.append(dict(date=m.group(1), direction=m.group(2), amount=float(m.group(3)),
                             category=m.group(4), counterparty=m.group(5).strip(),
                             channel=m.group(6), balance_after=float(m.group(7)), night=0))
    if not rows:
        raise ValueError("No transaction rows could be parsed from the PDF statement.")
    return validate_and_clean(pd.DataFrame(rows))


def compute_features(df: pd.DataFrame) -> dict:
    df = validate_and_clean(df)
    df["month"] = df["date"].dt.to_period("M")
    inflows = df[df["direction"] == "in"]
    outflows = df[df["direction"] == "out"]

    monthly_in = inflows.groupby("month")["amount"].sum()
    monthly_out = outflows.groupby("month")["amount"].sum()
    months = sorted(set(monthly_in.index) | set(monthly_out.index))
    monthly_in = monthly_in.reindex(months, fill_value=0)
    monthly_out = monthly_out.reindex(months, fill_value=0)

    avg_in = float(monthly_in.mean()) if len(monthly_in) else 0.0
    std_in = float(monthly_in.std()) if len(monthly_in) > 1 else 0.0
    income_cv = float(std_in / avg_in) if avg_in > 0 else 1.0
    months_active = int((monthly_in > 0).sum())
    regularity = months_active / max(len(months), 1)

    total_in = float(inflows["amount"].sum())
    total_out = float(outflows["amount"].sum())
    cash_out = float(outflows[outflows["category"] == "cash_out"]["amount"].sum())
    airtime = float(outflows[outflows["category"] == "airtime_purchase"]["amount"].sum())
    repayments = outflows[outflows["category"] == "loan_repayment"]
    repayment_total = float(repayments["amount"].sum())
    repayment_count = int(len(repayments))

    # savings trend: slope of end-of-month balance
    eom = df.groupby("month")["balance_after"].last().reindex(months).ffill().fillna(0)
    if len(eom) > 1:
        x = np.arange(len(eom))
        slope = float(np.polyfit(x, eom.values, 1)[0])
    else:
        slope = 0.0

    night_share = float(df["night"].mean()) if "night" in df else 0.0
    avg_inflow_amt = float(inflows["amount"].mean()) if len(inflows) else 0.0
    max_inflow = float(inflows["amount"].max()) if len(inflows) else 0.0
    spike_ratio = float(max_inflow / avg_inflow_amt) if avg_inflow_amt > 0 else 0.0

    net_positive_months = int(((monthly_in - monthly_out) > 0).sum())

    return dict(
        months=len(months),
        total_inflow=round(total_in, 2),
        total_outflow=round(total_out, 2),
        net=round(total_in - total_out, 2),
        avg_monthly_inflow=round(avg_in, 2),
        avg_monthly_outflow=round(float(monthly_out.mean()) if len(monthly_out) else 0.0, 2),
        income_cv=round(income_cv, 3),
        regularity=round(regularity, 3),
        tx_count=int(len(df)),
        tx_per_month=round(len(df) / max(len(months), 1), 1),
        cash_out_ratio=round(cash_out / total_in, 3) if total_in > 0 else 1.0,
        airtime_ratio=round(airtime / total_out, 3) if total_out > 0 else 0.0,
        repayment_count=repayment_count,
        repayment_total=round(repayment_total, 2),
        repayment_regular=bool(repayment_count >= max(1, len(months) // 2)),
        savings_slope=round(slope, 2),
        night_share=round(night_share, 3),
        spike_ratio=round(spike_ratio, 2),
        unique_counterparties=int(df["counterparty"].nunique()),
        net_positive_months=net_positive_months,
        avg_balance=round(float(df["balance_after"].mean()), 2),
    )

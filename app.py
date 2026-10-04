"""ScoreYedu — AI credit scoring for Zimbabwe's informal sector.

SELF-CONTAINED single-file app: no external 'modules' imports needed.
Just upload this file together with requirements.txt to GitHub and deploy.
"""
from __future__ import annotations

import io
import os
import random
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# 1. SAMPLE DATA GENERATOR (synthetic Zimbabwean informal-trader data)
# --------------------------------------------------------------------------
CUSTOMERS = ["C. Dube", "M. Chikafu", "R. Moyo", "T. Ncube", "K. Sibanda",
             "J. Mwale", "S. Chari", "P. Gondo", "N. Khumalo", "E. Mutasa"]
MERCHANTS = ["Mbare Produce Market", "Chitungwiza carpentry", "Avondale street vending",
             "Harare CBD hair salon", "Bulawayo textile market", "Mutare hardware supply",
             "Gweru second-hand clothing", "University dining kiosk"]


def generate_sample_dataset(n_months: int = 6, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    npr = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-05")
    rows, balance = [], float(npr.uniform(150, 400))
    base = float(npr.uniform(180, 650))
    for day in pd.date_range(start, periods=n_months * 30, freq="D"):
        for _ in range(rng.randint(0, 3)):
            amt = float(npr.uniform(5, base / 6))
            if rng.random() < 0.04:
                amt *= float(npr.uniform(4, 8))
            balance += amt
            rows.append(dict(date=day, direction="in", amount=round(amt, 2),
                             category="sale_income", counterparty=rng.choice(CUSTOMERS),
                             channel="EcoCash", balance_after=round(balance, 2),
                             night=int(rng.random() < 0.05)))
        for category, prob, lo, hi, direction in [
            ("cash_out", 0.55, 10, 60, "out"),
            ("airtime_purchase", 0.4, 1, 12, "out"),
            ("merchant_payment", 0.3, 15, 120, "out"),
        ]:
            if rng.random() < prob:
                amt = round(float(npr.uniform(lo, hi)), 2)
                balance = max(0, balance - amt)
                rows.append(dict(date=day, direction=direction, amount=amt, category=category,
                                 counterparty=rng.choice(MERCHANTS), channel="Paynow",
                                 balance_after=round(balance, 2), night=int(rng.random() < 0.1)))
        if day.weekday() == 4 and rng.random() < 0.6:
            amt = round(float(npr.uniform(8, 25)), 2)
            balance = max(0, balance - amt)
            rows.append(dict(date=day, direction="out", amount=amt, category="loan_repayment",
                             counterparty="MicroLoan ZW", channel="Paynow",
                             balance_after=round(balance, 2), night=0))
        if rng.random() < 0.25:
            amt = round(float(npr.uniform(5, 80)), 2)
            balance += amt
            rows.append(dict(date=day, direction="in", amount=amt, category="peer_transfer_in",
                             counterparty=rng.choice(CUSTOMERS), channel="EcoCash",
                             balance_after=round(balance, 2), night=0))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["reference"] = [f"TX{100000+i}" for i in range(len(df))]
    return df


# --------------------------------------------------------------------------
# 2. DATA PROCESSING
# --------------------------------------------------------------------------
REQUIRED = ["date", "direction", "amount", "category", "counterparty", "channel", "balance_after"]


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["balance_after"] = pd.to_numeric(df["balance_after"], errors="coerce")
    df = df.dropna(subset=["date", "amount", "balance_after"])
    df["direction"] = df["direction"].astype(str).str.lower()
    df["category"] = df["category"].astype(str).str.lower()
    if "night" not in df.columns:
        df["night"] = 0
    return df.sort_values("date").reset_index(drop=True)


def load_csv(file) -> pd.DataFrame:
    return validate_and_clean(pd.read_csv(file))


def parse_pdf(file) -> pd.DataFrame:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("pip install pypdf") from e
    reader = PdfReader(file)
    text = "\n".join(p.extract_text() or "" for p in reader.pages)
    rows = []
    pat = re.compile(r"(\d{4}-\d{2}-\d{2})[,\s]+(in|out)[,\s]+([\d.]+)[,\s]+([\w_]+)[,\s]+([\w\s\.\-]+)[,\s]+([\w]+)[,\s]+([\d.]+)", re.I)
    for line in text.splitlines():
        m = pat.search(line)
        if m:
            rows.append(dict(date=m.group(1), direction=m.group(2), amount=float(m.group(3)),
                             category=m.group(4), counterparty=m.group(5).strip(),
                             channel=m.group(6), balance_after=float(m.group(7)), night=0))
    if not rows:
        raise ValueError("No transaction rows found in the PDF.")
    return validate_and_clean(pd.DataFrame(rows))


def compute_features(df: pd.DataFrame) -> dict:
    df = validate_and_clean(df)
    df["month"] = df["date"].dt.to_period("M")
    in_, out = df[df.direction == "in"], df[df.direction == "out"]
    mi = in_.groupby("month").amount.sum()
    mo = out.groupby("month").amount.sum()
    months = sorted(set(mi.index) | set(mo.index))
    mi, mo = mi.reindex(months, fill_value=0), mo.reindex(months, fill_value=0)
    avg_in = float(mi.mean()) if len(mi) else 0.0
    std_in = float(mi.std()) if len(mi) > 1 else 0.0
    cv = float(std_in / avg_in) if avg_in > 0 else 1.0
    repays = out[out.category == "loan_repayment"]
    eom = df.groupby("month").balance_after.last().reindex(months).ffill().fillna(0)
    slope = float(np.polyfit(np.arange(len(eom)), eom.values, 1)[0]) if len(eom) > 1 else 0.0
    total_in, total_out = float(in_.amount.sum()), float(out.amount.sum())
    cash_out = float(out[out.category == "cash_out"].amount.sum())
    airtime = float(out[out.category == "airtime_purchase"].amount.sum())
    return dict(
        months=len(months), total_inflow=round(total_in, 2), total_outflow=round(total_out, 2),
        net=round(total_in - total_out, 2), avg_monthly_inflow=round(avg_in, 2),
        avg_monthly_outflow=round(float(mo.mean()) if len(mo) else 0, 2),
        income_cv=round(cv, 3), regularity=round(int((mi > 0).sum()) / max(len(months), 1), 3),
        tx_count=int(len(df)), tx_per_month=round(len(df) / max(len(months), 1), 1),
        cash_out_ratio=round(cash_out / total_in, 3) if total_in else 1.0,
        airtime_ratio=round(airtime / total_out, 3) if total_out else 0.0,
        repayment_count=int(len(repays)), repayment_total=round(float(repays.amount.sum()), 2),
        repayment_regular=bool(len(repays) >= max(1, len(months) // 2)),
        savings_slope=round(slope, 2), night_share=round(float(df["night"].mean()), 3),
        spike_ratio=round(float(in_.amount.max() / in_.amount.mean()), 2) if len(in_) else 0.0,
        unique_counterparties=int(df.counterparty.nunique()),
        net_positive_months=int(((mi - mo) > 0).sum()),
        avg_balance=round(float(df.balance_after.mean()), 2),
    )


# --------------------------------------------------------------------------
# 3. FRAUD / RISK FLAGS
# --------------------------------------------------------------------------
def detect_risk_flags(df: pd.DataFrame) -> list:
    df = df.sort_values("date")
    flags = []
    in_ = df[df.direction == "in"]
    if len(in_) >= 4:
        big = in_[in_.amount > 5 * in_.amount.mean()]
        if len(big):
            flags.append(f"Sudden large inflows ({len(big)} tx)")
    if len(df) and (df.amount % 50 == 0).mean() > 0.15 and (df.amount % 50 == 0).sum() > 5:
        flags.append("Frequent round-amount transfers (possible structuring)")
    if df.groupby(df.date.dt.date).size().max() > 25:
        flags.append("Unusually high daily velocity")
    if df["night"].mean() > 0.3:
        flags.append("High late-night transaction share")
    if "reference" in df and df.reference.duplicated().any():
        flags.append("Duplicate references")
    out = df[df.direction == "out"]
    if len(out) and out.counterparty.value_counts(normalize=True).iloc[0] > 0.6:
        flags.append("Outflows concentrated on one counterparty")
    if (df.balance_after < 0).any():
        flags.append("Negative wallet balance events")
    try:
        from sklearn.ensemble import IsolationForest
        if len(df) >= 12:
            X = df[["amount", "balance_after"]].fillna(0).values
            preds = IsolationForest(contamination=0.05, random_state=0).fit_predict(X)
            if (preds == -1).mean() > 0.08:
                flags.append(f"ML anomaly score: {(preds == -1).mean():.0%} anomalous")
    except Exception:
        pass
    return flags


def fraud_penalty(flags) -> float:
    return min(0.4, 0.08 * len(flags))


# --------------------------------------------------------------------------
# 4. SCORING ENGINE
# --------------------------------------------------------------------------
WEIGHTS = {"income_consistency": 250, "cash_flow_health": 250,
           "financial_discipline": 250, "repayment_history": 250}


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def score_applicant(f: dict, penalty: float = 0.0) -> dict:
    ic = _clamp(0.6 * (1 - _clamp(f["income_cv"], 0, 1)) + 0.4 * f["regularity"])
    positive = f["net_positive_months"] / max(f["months"], 1)
    ratio = _clamp(f["total_outflow"] / f["total_inflow"], 0, 1.5) if f["total_inflow"] > 0 else 1.5
    cf = _clamp(0.6 * positive + 0.4 * (1 - abs(1 - ratio) / 1.5))
    savings = _clamp(0.5 + f["savings_slope"] / max(abs(f["avg_monthly_inflow"]), 1))
    disc = _clamp(0.4 * savings + 0.2 * _clamp(1 - f["airtime_ratio"] * 2)
                  + 0.2 * _clamp(1 - f["night_share"] * 2)
                  + 0.2 * _clamp(1 - max(0, f["cash_out_ratio"] - 0.5) * 2))
    rep = 0.35 if f["repayment_count"] == 0 else _clamp(0.5 + 0.5 * (1.0 if f["repayment_regular"] else 0.5))
    parts = {"income_consistency": ic, "cash_flow_health": cf,
             "financial_discipline": disc, "repayment_history": rep}
    raw = sum(parts[k] * WEIGHTS[k] for k in parts)
    final = int(round(_clamp(raw - penalty * 1000, 0, 1000)))
    return dict(score=final, breakdown={k: int(round(parts[k] * WEIGHTS[k])) for k in parts})


def recommend(score: int, flags: list, f: dict) -> dict:
    severe = any("fraud" in x.lower() or "structuring" in x.lower() for x in flags)
    decision = "DECLINE" if (score < 450 or severe) else ("REVIEW" if score < 650 else "APPROVE")
    surplus = max(f["avg_monthly_inflow"] - f["avg_monthly_outflow"], 0)
    base = max(surplus * 3, f["avg_monthly_inflow"] * 0.5)
    amount = round(base * (1.0 if decision == "APPROVE" else 0.5), -1) if decision != "DECLINE" else 0
    return dict(decision=decision, loan_amount=float(max(0, min(amount, 5000))),
                repayment_weeks=12 if decision == "APPROVE" else (8 if decision == "REVIEW" else 0),
                rationale=f"Score {score}/1000 with {len(flags)} risk flag(s).")


# --------------------------------------------------------------------------
# 5. LLM CLIENT (optional) + OFFLINE FALLBACKS
# --------------------------------------------------------------------------
def llm_chat(prompt: str, system: str) -> str | None:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL") or None)
        r = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.4, max_tokens=900)
        return r.choices[0].message.content.strip()
    except Exception:
        return None


def explain_score(name, score, breakdown, flags, f) -> str:
    prompt = (f"Explain this Trust Score to a lender in 3 short paragraphs. Name: {name}, "
              f"Score {score}/1000, breakdown {breakdown}, flags {flags}, "
              f"avg inflow ${f['avg_monthly_inflow']}, CV {f['income_cv']}, repayments {f['repayment_count']}.")
    text = llm_chat(prompt, "You are ScoreYedu's credit analyst for Zimbabwean informal traders.")
    if text:
        return text
    parts = ", ".join(f"{k.replace('_', ' ')}: {v}/250" for k, v in breakdown.items())
    risk = "No major risk flags." if not flags else "Risk indicators: " + "; ".join(flags) + "."
    return (f"**{name}** scored **{score}/1000**. Breakdown — {parts}.\n\n"
            f"Average monthly inflow is ${f['avg_monthly_inflow']} over {f['months']} months; "
            f"{f['repayment_count']} repayments recorded. {risk}\n\n"
            "_Offline mode — set OPENAI_API_KEY for a richer LLM explanation._")


def credit_report(name, phone, f, score, breakdown, flags, rec) -> str:
    prompt = (f"Write a one-page credit report in Markdown. Applicant {name} ({phone}), "
              f"score {score}/1000, breakdown {breakdown}, flags {flags}, rec {rec}, features {f}.")
    text = llm_chat(prompt, "You are ScoreYedu's credit analyst.")
    if text:
        return text
    return "\n".join([
        "# ScoreYedu Credit Assessment Report",
        f"**Applicant:** {name} | **Phone:** {phone} | **Score:** {score}/1000",
        "", "## Score Breakdown",
        *[f"- {k.replace('_',' ').title()}: {v}/250" for k, v in breakdown.items()],
        "", "## Behaviour",
        f"- {f['tx_count']} tx over {f['months']} months (~{f['tx_per_month']}/mo), {f['unique_counterparties']} counterparties",
        f"- Cash-out share {f['cash_out_ratio']:.0%}, airtime share {f['airtime_ratio']:.0%}",
        "", "## Income & Cash Flow",
        f"- Avg monthly inflow ${f['avg_monthly_inflow']}, CV {f['income_cv']}, net ${f['net']}, positive months {f['net_positive_months']}/{f['months']}",
        "", "## Discipline & Repayment",
        f"- Savings slope ${f['savings_slope']}/mo, night share {f['night_share']:.0%}, {f['repayment_count']} repayments (${f['repayment_total']}), regular: {f['repayment_regular']}",
        "", "## Risk Flags",
        *([f"- {x}" for x in flags] if flags else ["- None"]),
        "", "## Recommendation",
        f"**{rec['decision']}** — suggested loan ${rec['loan_amount']} over {rec['repayment_weeks']} weeks. {rec['rationale']}",
        "", "_Offline rule-based report. Set OPENAI_API_KEY for narrative reports._",
    ])


def chat_answer(q: str, ctx: dict | None) -> str:
    ctxp = ""
    if ctx:
        ctxp = f"Applicant {ctx['name']}: score {ctx['score']}/1000, decision {ctx['decision']}, inflow ${ctx['avg_inflow']}. "
    text = llm_chat(ctxp + "Question: " + q, "You are ScoreYedu's friendly borrower assistant.")
    if text:
        return text
    ql = q.lower()
    if "improve" in ql:
        return ("Keep regular customer payments, pay repayments on time, avoid sudden unexplained "
                "large inflows, and reduce late-night transactions to improve your score.")
    if "score" in ql:
        return (f"Your Trust Score is {ctx['score']}/1000. " if ctx else "") + \
               "It blends income consistency, cash flow, discipline and repayment history."
    if "loan" in ql:
        return f"You may qualify for about ${ctx.get('loan_amount', 0)}." if ctx else "Build consistent inflows first."
    if "decline" in ql or "reject" in ql:
        return "Declines usually come from unstable income, fraud flags or no repayment history. Clean up 3 months of records and re-apply."
    return "Ask me about your score, your loan, or how to improve your profile."


# --------------------------------------------------------------------------
# 6. MOCK THIRD-PARTY APIS (labelled demo integrations)
# --------------------------------------------------------------------------
def mock_ecocash_fetch(phone: str) -> pd.DataFrame:
    import hashlib
    seed = int(hashlib.md5(phone.encode()).hexdigest()[:8], 16) % 1000
    return generate_sample_dataset(seed=seed)


def mock_paynow_verify(ref: str) -> dict:
    return {"reference": ref, "verified": ref[-1] in "05", "status": "PAID" if ref[-1] in "05" else "PENDING", "mock": True}


# --------------------------------------------------------------------------
# 7. STREAMLIT UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="ScoreYedu", page_icon="📊", layout="wide")
st.title("📊 ScoreYedu")
st.caption("AI-powered alternative credit scoring for Zimbabwe's informal-sector workers")

for k, v in [("applicants", []), ("current", None), ("chat", [])]:
    if k not in st.session_state:
        st.session_state[k] = v

page = st.sidebar.radio("Navigate", ["🏠 Home", "📤 Applicant Analysis",
                                     "🤖 AI Credit Analyst", "💬 Borrower Chatbot", "🛠️ Admin Dashboard"])
st.sidebar.info("EcoCash/Paynow are **mock APIs** for demo. Set OPENAI_API_KEY for full AI mode.")


def run_analysis(df, name, phone):
    f = compute_features(df)
    flags = detect_risk_flags(df)
    s = score_applicant(f, fraud_penalty(flags))
    rec = recommend(s["score"], flags, f)
    r = dict(name=name, phone=phone, features=f, flags=flags,
             score=s["score"], breakdown=s["breakdown"], recommendation=rec)
    st.session_state.current = r
    st.session_state.applicants.append(r)
    return r


if page == "🏠 Home":
    st.write("ScoreYedu analyses mobile-money transaction behaviour to build an explainable "
             "**Trust Score (0–1000)** and outputs APPROVE / REVIEW / DECLINE with a loan amount "
             "and repayment period.")
    st.markdown("1. Upload EcoCash CSV/PDF or use demo data → 2. Scoring engine analyses it → "
                "3. AI Credit Analyst explains → 4. Borrower chatbot helps → 5. Admin reviews.")

elif page == "📤 Applicant Analysis":
    c1, c2 = st.columns(2)
    name = c1.text_input("Applicant name", "Tapiwa Moyo")
    phone = c2.text_input("Phone", "+263771234567")
    src = st.radio("Data source", ["Upload CSV", "Upload PDF", "Demo sample data", "Mock EcoCash API"], horizontal=True)
    df = None
    try:
        if src == "Upload CSV":
            up = st.file_uploader("CSV", type=["csv"])
            if up:
                df = load_csv(up)
        elif src == "Upload PDF":
            up = st.file_uploader("PDF", type=["pdf"])
            if up:
                df = parse_pdf(up)
        elif src == "Demo sample data":
            df = generate_sample_dataset()
        else:
            if st.button("Fetch (mock)"):
                df = mock_ecocash_fetch(phone)
                st.success("Mock EcoCash API data loaded.")
    except Exception as e:
        st.error(str(e))
    if df is not None:
        st.dataframe(df.head(20), use_container_width=True)
        if st.button("🔎 Analyse", type="primary"):
            r = run_analysis(df, name, phone)
            m1, m2, m3 = st.columns(3)
            m1.metric("Trust Score", f"{r['score']}/1000")
            m2.metric("Decision", r["recommendation"]["decision"])
            m3.metric("Loan", f"${r['recommendation']['loan_amount']:.0f}")
            st.write("Breakdown:", r["breakdown"])
            st.write("Flags:", r["flags"] or "None")

elif page == "🤖 AI Credit Analyst":
    if st.session_state.current is None:
        st.warning("Analyse an applicant first.")
    else:
        r = st.session_state.current
        st.write(f"**{r['name']}** | Score {r['score']} | {r['recommendation']['decision']}")
        if st.button("Explain score"):
            st.markdown(explain_score(r["name"], r["score"], r["breakdown"], r["flags"], r["features"]))
        if st.button("Full credit report"):
            rep = credit_report(r["name"], r["phone"], r["features"], r["score"], r["breakdown"], r["flags"], r["recommendation"])
            st.markdown(rep)
            st.download_button("⬇️ Download", rep, file_name="scoreyedu_report.md")
        ref = st.text_input("Paynow reference (mock)", "P20260510")
        if st.button("Verify mock"):
            st.json(mock_paynow_verify(ref))

elif page == "💬 Borrower Chatbot":
    ctx = None
    if st.session_state.current:
        r = st.session_state.current
        ctx = dict(name=r["name"], score=r["score"], decision=r["recommendation"]["decision"],
                   avg_inflow=r["features"]["avg_monthly_inflow"],
                   loan_amount=r["recommendation"]["loan_amount"])
    for role, msg in st.session_state.chat:
        st.chat_message(role).write(msg)
    if q := st.chat_input("Ask about your score, loan, or how to improve..."):
        st.session_state.chat.append(("user", q))
        st.chat_message("user").write(q)
        reply = chat_answer(q, ctx)
        st.session_state.chat.append(("assistant", reply))
        st.chat_message("assistant").write(reply)

else:
    if not st.session_state.applicants:
        st.info("No applicants yet.")
    else:
        rows = [dict(name=a["name"], score=a["score"], decision=a["recommendation"]["decision"],
                     loan=a["recommendation"]["loan_amount"], flags=len(a["flags"]),
                     avg_inflow=a["features"]["avg_monthly_inflow"]) for a in st.session_state.applicants]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.bar_chart(df.set_index("name")["score"])
        c2.bar_chart(df.decision.value_counts())
        st.download_button("⬇️ Export CSV", df.to_csv(index=False), file_name="applicants.csv")

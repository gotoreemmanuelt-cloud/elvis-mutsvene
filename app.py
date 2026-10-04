"""ScoreYedu — AI credit scoring for Zimbabwe's informal sector."""
from __future__ import annotations
import os
import sys
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_processing import load_csv, parse_pdf, compute_features
from modules.scoring_engine import score_applicant, recommend
from modules.fraud_detection import detect_risk_flags, fraud_penalty
from modules.ai_analyst import explain_score, credit_report
from modules.chatbot import answer
from modules.mock_apis import mock_ecocash_fetch, mock_paynow_verify
from modules.sample_data_generator import generate_sample_dataset

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_ecocash_transactions.csv")

st.set_page_config(page_title="ScoreYedu", page_icon="📊", layout="wide")
st.title("📊 ScoreYedu")
st.caption("AI-powered alternative credit scoring for Zimbabwe's informal-sector workers")

if "applicants" not in st.session_state:
    st.session_state.applicants = []
if "current" not in st.session_state:
    st.session_state.current = None

page = st.sidebar.radio("Navigate", ["🏠 Home", "📤 Applicant Analysis", "🤖 AI Credit Analyst",
                                     "💬 Borrower Chatbot", "🛠️ Admin Dashboard"])
st.sidebar.markdown("---")
st.sidebar.info("EcoCash/Paynow integrations are **mock APIs** for demo purposes.")


def run_analysis(df: pd.DataFrame, name: str, phone: str) -> dict:
    features = compute_features(df)
    flags = detect_risk_flags(df)
    penalty = fraud_penalty(flags)
    scored = score_applicant(features, penalty)
    rec = recommend(scored["score"], flags, features)
    result = dict(name=name, phone=phone, features=features, flags=flags,
                  score=scored["score"], breakdown=scored["breakdown"], recommendation=rec)
    st.session_state.current = result
    st.session_state.applicants.append(result)
    return result


if page == "🏠 Home":
    st.subheader("Welcome to ScoreYedu")
    st.write("ScoreYedu analyses EcoCash/Paynow transaction behaviour to build an explainable "
             "**Trust Score (0–1000)** for informal-sector workers, and outputs an APPROVE / REVIEW / "
             "DECLINE recommendation with a suggested loan amount and repayment period.")
    st.markdown("""
**How it works**
1. Upload EcoCash transactions (CSV or PDF statement) — or use the built-in sample/demo data.
2. The scoring engine measures income consistency, cash flow, financial discipline and repayment history.
3. Risk & fraud indicators are detected (rules + IsolationForest anomaly detection).
4. An AI Credit Analyst explains the score and writes a credit assessment report.
5. A borrower chatbot helps applicants understand and improve their score.
""")
    st.warning("All EcoCash / Paynow integrations in this demo are clearly labelled **mock APIs**.")

elif page == "📤 Applicant Analysis":
    st.subheader("Applicant Analysis")
    col1, col2 = st.columns(2)
    name = col1.text_input("Applicant name", "Tapiwa Moyo")
    phone = col2.text_input("EcoCash phone number", "+263771234567")

    source = st.radio("Data source", ["Upload CSV", "Upload PDF statement", "Use sample CSV",
                                      "Mock EcoCash API (demo)"], horizontal=True)
    df = None
    if source == "Upload CSV":
        up = st.file_uploader("CSV file", type=["csv"])
        if up:
            try:
                df = load_csv(up)
            except Exception as e:
                st.error(str(e))
    elif source == "Upload PDF statement":
        up = st.file_uploader("PDF file", type=["pdf"])
        if up:
            try:
                df = parse_pdf(up)
            except Exception as e:
                st.error(str(e))
    elif source == "Use sample CSV":
        if os.path.exists(SAMPLE_PATH):
            df = pd.read_csv(SAMPLE_PATH)
            df["date"] = pd.to_datetime(df["date"])
        else:
            st.info("Sample file missing — generating on the fly.")
            df = generate_sample_dataset()
    else:
        if st.button("Fetch via mock EcoCash API"):
            df = mock_ecocash_fetch(phone)
            st.success("MOCK EcoCash API returned synthetic transactions.")

    if df is not None:
        st.dataframe(df.head(20), use_container_width=True)
        if st.button("🔎 Analyse applicant", type="primary"):
            with st.spinner("Scoring..."):
                result = run_analysis(df, name, phone)
            c1, c2, c3 = st.columns(3)
            c1.metric("Trust Score", f"{result['score']}/1000")
            c2.metric("Decision", result["recommendation"]["decision"])
            c3.metric("Suggested loan", f"${result['recommendation']['loan_amount']:.0f}")
            st.write("**Breakdown:**", result["breakdown"])
            if result["flags"]:
                st.warning("Risk flags: " + "; ".join(result["flags"]))
            st.info(f"Repayment period: {result['recommendation']['repayment_weeks']} weeks")

elif page == "🤖 AI Credit Analyst":
    st.subheader("AI Credit Analyst")
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("LLM_API_KEY"):
        st.info("No LLM API key set — running in offline rule-based mode. Set `OPENAI_API_KEY` to enable the LLM.")
    if st.session_state.current is None:
        st.warning("Analyse an applicant first (Applicant Analysis page).")
    else:
        r = st.session_state.current
        st.write(f"**Applicant:** {r['name']} | **Score:** {r['score']}/1000 | "
                 f"**Decision:** {r['recommendation']['decision']}")
        if st.button("Explain this score"):
            st.markdown(explain_score(r["name"], r["score"], r["breakdown"], r["flags"], r["features"]))
        if st.button("Generate full credit report"):
            report = credit_report(r["name"], r["phone"], r["features"], r["score"],
                                   r["breakdown"], r["flags"], r["recommendation"])
            st.markdown(report)
            st.download_button("⬇️ Download report (Markdown)", report,
                               file_name=f"scoreyedu_report_{r['name'].replace(' ', '_')}.md")
        st.markdown("---")
        st.write("**Mock Paynow check** for the applicant's latest loan reference:")
        ref = st.text_input("Reference", "P20260510")
        if st.button("Verify (mock)"):
            st.json(mock_paynow_verify(ref))

elif page == "💬 Borrower Chatbot":
    st.subheader("Borrower Chatbot")
    ctx = None
    if st.session_state.current:
        r = st.session_state.current
        ctx = dict(name=r["name"], score=r["score"], decision=r["recommendation"]["decision"],
                   avg_inflow=r["features"]["avg_monthly_inflow"],
                   loan_amount=r["recommendation"]["loan_amount"])
        st.caption(f"Chatting about {r['name']} (score {r['score']}/1000)")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for role, msg in st.session_state.chat:
        st.chat_message(role).write(msg)
    if q := st.chat_input("Ask about your score, loans, or how to improve..."):
        st.session_state.chat.append(("user", q))
        st.chat_message("user").write(q)
        reply = answer(q, ctx)
        st.session_state.chat.append(("assistant", reply))
        st.chat_message("assistant").write(reply)

else:
    st.subheader("Admin Dashboard")
    if not st.session_state.applicants:
        st.info("No applicants analysed yet. Run an analysis first.")
    else:
        rows = []
        for a in st.session_state.applicants:
            rows.append(dict(name=a["name"], phone=a["phone"], score=a["score"],
                             decision=a["recommendation"]["decision"],
                             loan=a["recommendation"]["loan_amount"],
                             weeks=a["recommendation"]["repayment_weeks"],
                             flags=len(a["flags"]), months=a["features"]["months"],
                             avg_inflow=a["features"]["avg_monthly_inflow"]))
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.bar_chart(df.set_index("name")["score"])
        c2.bar_chart(df["decision"].value_counts())
        st.write("**All risk factors**")
        for a in st.session_state.applicants:
            if a["flags"]:
                st.write(f"- **{a['name']}**: " + "; ".join(a["flags"]))
        st.download_button("⬇️ Export applicants CSV", df.to_csv(index=False),
                           file_name="scoreyedu_applicants.csv")

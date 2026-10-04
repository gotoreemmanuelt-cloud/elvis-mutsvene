"""Borrower-facing chatbot with friendly fallback answers."""
from __future__ import annotations
from .llm_client import chat

SYSTEM = ("You are ScoreYedu's friendly borrower assistant. Help Zimbabwean informal "
          "traders understand their credit score and how to improve it. Be concise.")


def answer(question: str, context: dict | None = None) -> str:
    ctx = ""
    if context:
        ctx = (f"Applicant {context.get('name')} has score {context.get('score')}/1000, "
               f"recommendation {context.get('decision')}, avg monthly inflow "
               f"${context.get('avg_inflow')}. ")
    text = chat(ctx + "Question: " + question, SYSTEM)
    if text:
        return text
    q = question.lower()
    if "score" in q and "improve" in q or "increase" in q:
        return ("To improve your Trust Score: keep regular customer payments, avoid sudden huge "
                "inflows without explanation, pay any loan repayments on time, reduce airtime and "
                "late-night transactions, and build a positive balance trend.")
    if "score" in q:
        base = f"Your Trust Score is currently **{context.get('score')}/1000**. " if context else ""
        return base + "It blends income consistency, cash flow, discipline and repayment history."
    if "loan" in q and ("amount" in q or "get" in q):
        if context and context.get('decision') == "APPROVE":
            return "Based on your profile you may qualify for around $" + str(context.get('loan_amount')) + "."
        return "Build a few months of consistent EcoCash inflows first, then re-apply."
    if "decline" in q or "reject" in q:
        return ("Applications may be declined due to very unstable income, many fraud flags, or "
                "missing repayment history. Keep a cleaner record for 3 months and try again.")
    return ("I can help with your Trust Score, loan eligibility, and tips to improve your profile. "
            "Ask me about your score, how to improve it, or why a decision was made.")

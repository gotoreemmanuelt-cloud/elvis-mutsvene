"""LLM client with graceful offline fallback (rule-based) behaviour."""
from __future__ import annotations
import os

MODEL_DEFAULT = "gpt-4o-mini"


def is_llm_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"))


def chat(prompt: str, system: str = "You are a helpful credit analyst for Zimbabwe.") -> str | None:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL") or None)
        resp = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", MODEL_DEFAULT),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=0.4, max_tokens=900)
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

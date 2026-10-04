"""MOCK third-party integrations.

These functions simulate EcoCash / Paynow APIs for the ScoreYedu demo.
They are NOT real integrations and must not be used with real money data.
"""
from __future__ import annotations
import hashlib
import pandas as pd
from .sample_data_generator import generate_sample_dataset


def mock_ecocash_fetch(phone_number: str) -> pd.DataFrame:
    """MOCK EcoCash API: returns synthetic transactions for a phone number."""
    seed = int(hashlib.md5(phone_number.encode()).hexdigest()[:8], 16) % 1000
    return generate_sample_dataset(n_months=6, seed=seed)


def mock_paynow_verify(reference: str) -> dict:
    """MOCK Paynow verification of a transaction reference."""
    ok = reference.endswith("0") or reference.endswith("5")
    return {"reference": reference, "verified": ok,
            "status": "PAID" if ok else "PENDING", "mock": True}


def mock_mobile_money_consent(phone_number: str) -> bool:
    """MOCK: simulate user consent to share mobile-money data."""
    return True

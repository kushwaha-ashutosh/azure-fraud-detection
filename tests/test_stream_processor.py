import json
import pytest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.abspath("functions/stream_processor"))

from function_app import enrich_transaction, NUMERICAL_FEATURES

def make_txn():
    return {
        "transaction_id": "txn_test123",
        "timestamp": "2026-04-10T12:00:00Z",
        "amount": 1500.0,
        "currency": "INR",
        "entry_mode": "Online",
        "merchant_id": "merch_abc",
        "merchant_category": 5411,
        "user_id": "usr_xyz",
        "user_account_age_days": 365,
        "card_id": "tok_def",
        "card_type": "Credit",
        "card_brand": "Visa",
        "card_country": "IN",
        "ip_address": "1.2.3.4",
        "device_id": "dev_ghi",
        "was_3ds_successful": True,
        "is_fraud": False
    }

def test_enrich_adds_numerical_features():
    txn = make_txn()
    enriched = enrich_transaction(txn)
    for f in NUMERICAL_FEATURES:
        assert f in enriched, f"Missing feature: {f}"

def test_enrich_amount_mapped():
    txn = make_txn()
    enriched = enrich_transaction(txn)
    assert enriched["Amount"] == 1500.0

def test_enrich_time_mapped():
    txn = make_txn()
    enriched = enrich_transaction(txn)
    assert enriched["Time"] == 365 * 24.0

def test_enrich_preserves_original_fields():
    txn = make_txn()
    enriched = enrich_transaction(txn)
    assert enriched["transaction_id"] == "txn_test123"
    assert enriched["currency"] == "INR"

def test_numerical_features_count():
    assert len(NUMERICAL_FEATURES) == 30
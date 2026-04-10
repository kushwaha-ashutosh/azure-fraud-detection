import pytest
import sys, os
sys.path.insert(0, os.path.abspath("."))
from batch_etl.etl_job import transform

def make_raw_txn(fraud_score=0.1, is_fraud=False):
    return {
        "transaction_id": "txn_test001",
        "timestamp": "2026-04-10T12:00:00Z",
        "amount": 1500.0,
        "currency": "INR",
        "user_id": "usr_abc",
        "merchant_id": "merch_xyz",
        "merchant_category": 5411,
        "card_type": "Credit",
        "card_brand": "Visa",
        "entry_mode": "Online",
        "fraud_score": fraud_score,
        "is_fraud_predicted": fraud_score > 0.5,
        "is_fraud": is_fraud,
        "scored_at": "2026-04-10T12:00:01Z"
    }

def test_transform_basic():
    records = transform([make_raw_txn()])
    assert len(records) == 1

def test_transform_fields_present():
    records = transform([make_raw_txn()])
    r = records[0]
    for field in ["transaction_id","amount","fraud_score","is_fraud_predicted","etl_processed_at"]:
        assert field in r

def test_transform_fraud_flag():
    records = transform([make_raw_txn(fraud_score=0.9)])
    assert records[0]["is_fraud_predicted"] == True

def test_transform_amount_type():
    records = transform([make_raw_txn()])
    assert isinstance(records[0]["amount"], float)

def test_transform_skips_malformed():
    bad = {"broken": "record", "amount": "not_a_number_xyz"}
    good = make_raw_txn()
    results = transform([bad, good])
    assert len(results) == 1
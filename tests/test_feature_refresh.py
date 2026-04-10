import pytest
import sys, os
sys.path.insert(0, os.path.abspath("."))
from orchestration.feature_refresh import compute_user_features

def make_records(user_id="usr_001", amount=1000.0, is_fraud=False, count=3):
    return [
        {
            "user_id": user_id,
            "amount": amount,
            "is_fraud_predicted": is_fraud,
            "transaction_id": f"txn_{i}"
        }
        for i in range(count)
    ]

def test_compute_basic():
    records = make_records()
    features = compute_user_features(records)
    assert len(features) == 1

def test_transaction_count():
    records = make_records(count=5)
    features = compute_user_features(records)
    assert features[0]["transaction_count"] == 5

def test_avg_amount():
    records = make_records(amount=2000.0, count=4)
    features = compute_user_features(records)
    assert features[0]["avg_transaction_amount"] == 2000.0

def test_fraud_rate():
    legit = make_records("usr_a", is_fraud=False, count=8)
    fraud = make_records("usr_a", is_fraud=True, count=2)
    features = compute_user_features(legit + fraud)
    assert features[0]["fraud_rate"] == 0.2

def test_multiple_users():
    r1 = make_records("usr_a", count=3)
    r2 = make_records("usr_b", count=5)
    features = compute_user_features(r1 + r2)
    assert len(features) == 2

def test_feature_has_id():
    records = make_records()
    features = compute_user_features(records)
    assert "id" in features[0]
    assert features[0]["id"] == features[0]["user_id"]
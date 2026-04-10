import json
import pytest
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.abspath("functions/realtime_action"))

from function_app import send_alert, FRAUD_THRESHOLD

def make_scored_txn(fraud_score=0.9):
    return {
        "transaction_id": "txn_test999",
        "amount": 5000.0,
        "currency": "INR",
        "user_id": "usr_abc",
        "timestamp": "2026-04-10T12:00:00Z",
        "fraud_score": fraud_score,
        "is_fraud_predicted": fraud_score > 0.5
    }

def test_fraud_threshold_default():
    assert FRAUD_THRESHOLD == 0.5

def test_high_score_is_fraud():
    txn = make_scored_txn(fraud_score=0.95)
    assert txn["fraud_score"] > FRAUD_THRESHOLD

def test_low_score_is_legit():
    txn = make_scored_txn(fraud_score=0.1)
    assert txn["fraud_score"] <= FRAUD_THRESHOLD

def test_send_alert_skips_without_url():
    with patch.dict(os.environ, {"LOGIC_APP_URL": ""}):
        txn = make_scored_txn()
        send_alert(txn, 0.95)

def test_fraud_score_range():
    for score in [0.0, 0.5, 0.99, 1.0]:
        txn = make_scored_txn(fraud_score=score)
        assert 0.0 <= txn["fraud_score"] <= 1.0
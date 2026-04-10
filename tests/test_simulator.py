
import json
import pytest
from simulator.generator import generate_transaction, generate_batch
from simulator.schema import Transaction

def test_transaction_has_all_fields():
    txn = generate_transaction()
    assert isinstance(txn, Transaction)
    d = txn.to_dict()
    for field in ["transaction_id","timestamp","amount","currency","entry_mode",
                  "merchant_id","user_id","card_id","card_type","is_fraud"]:
        assert field in d

def test_transaction_id_format():
    txn = generate_transaction()
    assert txn.transaction_id.startswith("txn_")

def test_amount_is_positive():
    for _ in range(50):
        assert generate_transaction().amount > 0

def test_fraud_rate_accuracy():
    batch = generate_batch(count=1000, fraud_rate=0.02)
    fraud_pct = sum(1 for t in batch if t.is_fraud) / len(batch)
    assert 0.01 <= fraud_pct <= 0.04, f"Fraud rate {fraud_pct} out of range"

def test_batch_count():
    assert len(generate_batch(count=50)) == 50

def test_to_dict_serializable():
    assert len(json.dumps(generate_transaction().to_dict())) > 0

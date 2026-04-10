files = {
'simulator/__init__.py': '',
'tests/__init__.py': '',
'simulator/schema.py': """
from dataclasses import dataclass, asdict
import uuid

@dataclass
class Transaction:
    transaction_id: str
    timestamp: str
    amount: float
    currency: str
    entry_mode: str
    merchant_id: str
    merchant_category: int
    user_id: str
    user_account_age_days: int
    card_id: str
    card_type: str
    card_brand: str
    card_country: str
    ip_address: str
    device_id: str
    was_3ds_successful: bool
    is_fraud: bool

    def to_dict(self):
        return asdict(self)
""",
'simulator/generator.py': """
import random
import uuid
from datetime import datetime, timezone
from faker import Faker
from simulator.schema import Transaction

fake = Faker()

CURRENCIES = ["INR", "USD", "EUR", "GBP"]
ENTRY_MODES = ["Online", "Chip", "Contactless", "Swipe"]
CARD_TYPES = ["Credit", "Debit"]
CARD_BRANDS = ["Visa", "Mastercard", "Rupay", "Amex"]
CARD_COUNTRIES = ["IN", "US", "GB", "SG", "AE"]
MERCHANT_CATEGORIES = [5411, 5541, 5812, 5999, 7011, 4111, 6011]

def generate_transaction(fraud_rate=0.02):
    is_fraud = random.random() < fraud_rate
    amount = round(random.uniform(500, 50000) if is_fraud else random.uniform(50, 10000), 2)
    return Transaction(
        transaction_id=f"txn_{uuid.uuid4().hex[:12]}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        amount=amount,
        currency=random.choice(CURRENCIES),
        entry_mode=random.choice(ENTRY_MODES),
        merchant_id=f"merch_{uuid.uuid4().hex[:8]}",
        merchant_category=random.choice(MERCHANT_CATEGORIES),
        user_id=f"usr_{uuid.uuid4().hex[:8]}",
        user_account_age_days=random.randint(1, 3650),
        card_id=f"tok_{uuid.uuid4().hex[:8]}",
        card_type=random.choice(CARD_TYPES),
        card_brand=random.choice(CARD_BRANDS),
        card_country=random.choice(CARD_COUNTRIES),
        ip_address=fake.ipv4(),
        device_id=f"dev_{uuid.uuid4().hex[:8]}",
        was_3ds_successful=random.choice([True, False]),
        is_fraud=is_fraud,
    )

def generate_batch(count=100, fraud_rate=0.02):
    return [generate_transaction(fraud_rate) for _ in range(count)]
""",
'simulator/main.py': """
import json
import time
import argparse
from simulator.generator import generate_transaction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--fraud-rate", type=float, default=0.02)
    parser.add_argument("--output", choices=["stdout", "servicebus"], default="stdout")
    args = parser.parse_args()
    print(f"Starting: {args.count} transactions at {args.rate}/sec")
    for i in range(args.count):
        txn = generate_transaction(args.fraud_rate)
        if args.output == "stdout":
            print(json.dumps(txn.to_dict()))
        time.sleep(1.0 / args.rate)
    print(f"Done. Generated {args.count} transactions.")

if __name__ == "__main__":
    main()
""",
'tests/test_simulator.py': """
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
"""
}

for path, content in files.items():
    with open(path, 'w') as f:
        f.write(content)
    print(f"wrote {path}")

print("All files written OK")
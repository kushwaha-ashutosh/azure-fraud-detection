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

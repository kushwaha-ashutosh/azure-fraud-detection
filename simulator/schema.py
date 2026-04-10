
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

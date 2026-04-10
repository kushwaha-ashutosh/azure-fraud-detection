import json
import os
from datetime import datetime, timezone
from azure.storage.blob import BlobServiceClient

STORAGE_CONN = os.environ.get("STORAGE_CONNECTION_STRING", "")
SOURCE_CONTAINER = "scored-transactions"
OUTPUT_CONTAINER = "etl-output"

def read_scored_transactions(client, prefix="transactions/"):
    container = client.get_container_client(SOURCE_CONTAINER)
    blobs = container.list_blobs(name_starts_with=prefix)
    transactions = []
    for blob in blobs:
        if blob.name.endswith(".json"):
            bc = client.get_blob_client(SOURCE_CONTAINER, blob.name)
            data = json.loads(bc.download_blob().readall())
            transactions.append(data)
    return transactions

def transform(transactions):
    cleaned = []
    for t in transactions:
        try:
            cleaned.append({
                "transaction_id": t.get("transaction_id", ""),
                "timestamp": t.get("timestamp", ""),
                "amount": float(t.get("amount", 0)),
                "currency": t.get("currency", ""),
                "user_id": t.get("user_id", ""),
                "merchant_id": t.get("merchant_id", ""),
                "merchant_category": int(t.get("merchant_category", 0)),
                "card_type": t.get("card_type", ""),
                "card_brand": t.get("card_brand", ""),
                "entry_mode": t.get("entry_mode", ""),
                "fraud_score": float(t.get("fraud_score", 0)),
                "is_fraud_predicted": bool(t.get("is_fraud_predicted", False)),
                "is_fraud_actual": bool(t.get("is_fraud", False)),
                "scored_at": t.get("scored_at", ""),
                "etl_processed_at": datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            print(f"Skipping malformed record: {e}")
    return cleaned

def write_output(client, records):
    if not records:
        print("No records to write")
        return
    now = datetime.now(timezone.utc)
    output_path = f"processed/{now.year}/{now.month:02d}/{now.day:02d}/transactions.json"
    container = client.get_container_client(OUTPUT_CONTAINER)
    container.upload_blob(
        output_path,
        json.dumps(records, indent=2),
        overwrite=True
    )
    print(f"Written {len(records)} records to {output_path}")
    return output_path

def run_etl():
    print("=== Starting ETL Job ===")
    client = BlobServiceClient.from_connection_string(STORAGE_CONN)
    print("Reading scored transactions from Blob...")
    transactions = read_scored_transactions(client)
    print(f"Read {len(transactions)} transactions")
    cleaned = transform(transactions)
    print(f"Transformed {len(cleaned)} records")
    output_path = write_output(client, cleaned)
    fraud_count = sum(1 for r in cleaned if r["is_fraud_predicted"])
    print(f"Fraud detected: {fraud_count}/{len(cleaned)} ({100*fraud_count/max(len(cleaned),1):.1f}%)")
    print("=== ETL Job Complete ===")
    return cleaned

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_etl()
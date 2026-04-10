import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

STORAGE_CONN = os.environ.get("STORAGE_CONNECTION_STRING", "")
COSMOS_ENDPOINT = os.environ.get("COSMOS_ENDPOINT", "")
COSMOS_KEY = os.environ.get("COSMOS_KEY", "")
COSMOS_DATABASE = os.environ.get("COSMOS_DATABASE", "frauddetection")
COSMOS_CONTAINER = os.environ.get("COSMOS_CONTAINER", "features")

def read_etl_output():
    from azure.storage.blob import BlobServiceClient
    client = BlobServiceClient.from_connection_string(STORAGE_CONN)
    container = client.get_container_client("etl-output")
    blobs = list(container.list_blobs(name_starts_with="processed/"))
    all_records = []
    for blob in blobs:
        if blob.name.endswith(".json"):
            bc = client.get_blob_client("etl-output", blob.name)
            records = json.loads(bc.download_blob().readall())
            all_records.extend(records)
    return all_records

def compute_user_features(records):
    from collections import defaultdict
    user_stats = defaultdict(lambda: {
        "transaction_count": 0,
        "total_amount": 0.0,
        "fraud_count": 0,
        "amounts": []
    })
    for r in records:
        uid = r.get("user_id", "unknown")
        user_stats[uid]["transaction_count"] += 1
        user_stats[uid]["total_amount"] += r.get("amount", 0.0)
        user_stats[uid]["amounts"].append(r.get("amount", 0.0))
        if r.get("is_fraud_predicted"):
            user_stats[uid]["fraud_count"] += 1

    features = []
    for uid, stats in user_stats.items():
        amounts = stats["amounts"]
        avg_amount = stats["total_amount"] / max(len(amounts), 1)
        features.append({
            "id": uid,
            "user_id": uid,
            "transaction_count": stats["transaction_count"],
            "avg_transaction_amount": round(avg_amount, 2),
            "total_amount": round(stats["total_amount"], 2),
            "fraud_count": stats["fraud_count"],
            "fraud_rate": round(stats["fraud_count"] / max(stats["transaction_count"], 1), 4),
            "updated_at": datetime.now(timezone.utc).isoformat()
        })
    return features

def upsert_to_cosmos(features):
    from azure.cosmos import CosmosClient
    client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    db = client.get_database_client(COSMOS_DATABASE)
    container = db.get_container_client(COSMOS_CONTAINER)
    for f in features:
        container.upsert_item(f)
    print(f"Upserted {len(features)} user feature records to Cosmos DB")

def run_feature_refresh():
    print("=== Starting Feature Refresh ===")
    records = read_etl_output()
    print(f"Read {len(records)} ETL records")
    features = compute_user_features(records)
    print(f"Computed features for {len(features)} users")
    upsert_to_cosmos(features)
    print("=== Feature Refresh Complete ===")
    return features

if __name__ == "__main__":
    run_feature_refresh()
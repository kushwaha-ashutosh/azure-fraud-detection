import azure.functions as func
import json
import logging
import os
import io
import uuid
import tempfile
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

app = func.FunctionApp()

SERVICEBUS_CONN = os.environ.get("SERVICEBUS_CONNECTION_STRING", "")
COSMOS_ENDPOINT = os.environ.get("COSMOS_ENDPOINT", "")
COSMOS_KEY = os.environ.get("COSMOS_KEY", "")
STORAGE_CONN = os.environ.get("STORAGE_CONNECTION_STRING", "")
ML_INFERENCE_URL = os.environ.get("ML_INFERENCE_URL", "http://localhost:7071/api/predict")
FRAUD_THRESHOLD = float(os.environ.get("FRAUD_SCORE_THRESHOLD", "0.5"))

NUMERICAL_FEATURES = [
    "Time", "Amount", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"
]

def enrich_transaction(txn: dict) -> dict:
    """Add ML-ready features to raw transaction."""
    enriched = txn.copy()
    enriched["Time"] = enriched.get("user_account_age_days", 0) * 24.0
    enriched["Amount"] = enriched.get("amount", 0.0)
    for i in range(1, 29):
        enriched[f"V{i}"] = 0.0
    enriched["entry_mode_encoded"] = hash(enriched.get("entry_mode", "")) % 100
    return enriched

def call_ml_inference(enriched: dict) -> dict:
    """Call the ML inference Function."""
    import urllib.request
    features = {f: enriched.get(f, 0.0) for f in NUMERICAL_FEATURES}
    payload = json.dumps({"features": features}).encode()
    req = urllib.request.Request(
        ML_INFERENCE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())

def write_to_blob(scored: dict):
    """Write scored transaction to Blob Storage as JSON."""
    from azure.storage.blob import BlobServiceClient
    client = BlobServiceClient.from_connection_string(STORAGE_CONN)
    now = datetime.now(timezone.utc)
    blob_name = f"transactions/{now.year}/{now.month:02d}/{now.day:02d}/{scored['transaction_id']}.json"
    blob_client = client.get_blob_client(container="scored-transactions", blob=blob_name)
    blob_client.upload_blob(json.dumps(scored), overwrite=True)

def send_to_scored_queue(scored: dict):
    """Forward scored transaction to scored-transactions Service Bus queue."""
    from azure.servicebus import ServiceBusClient, ServiceBusMessage
    client = ServiceBusClient.from_connection_string(SERVICEBUS_CONN)
    sender = client.get_queue_sender(queue_name="scored-transactions")
    sender.send_messages(ServiceBusMessage(json.dumps(scored)))
    sender.close()
    client.close()

@app.service_bus_queue_trigger(
    arg_name="msg",
    queue_name="raw-transactions",
    connection="SERVICEBUS_CONNECTION_STRING"
)
def stream_processor(msg: func.ServiceBusMessage):
    try:
        body = msg.get_body().decode("utf-8")
        txn = json.loads(body)
        logging.info(f"Processing transaction: {txn.get('transaction_id')}")

        enriched = enrich_transaction(txn)
        ml_result = call_ml_inference(enriched)

        scored = {
            **txn,
            "fraud_score": ml_result.get("fraud_score", 0.0),
            "is_fraud_predicted": ml_result.get("is_fraud", False),
            "scored_at": datetime.now(timezone.utc).isoformat()
        }

        write_to_blob(scored)
        send_to_scored_queue(scored)

        logging.info(f"Scored: {txn.get('transaction_id')} -> {scored['fraud_score']}")

    except Exception as e:
        logging.error(f"Stream processor error: {e}")
        raise
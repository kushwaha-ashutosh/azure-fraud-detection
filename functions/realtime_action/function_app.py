import azure.functions as func
import json
import logging
import os
from datetime import datetime, timezone

app = func.FunctionApp()

FRAUD_THRESHOLD = float(os.environ.get("FRAUD_SCORE_THRESHOLD", "0.5"))
LOGIC_APP_URL = os.environ.get("LOGIC_APP_URL", "")

def send_alert(txn: dict, fraud_score: float):
    """Send fraud alert via Logic App webhook."""
    import urllib.request
    if not LOGIC_APP_URL:
        logging.warning("LOGIC_APP_URL not set - skipping alert")
        return
    payload = json.dumps({
        "transaction_id": txn.get("transaction_id"),
        "fraud_score": fraud_score,
        "amount": txn.get("amount"),
        "currency": txn.get("currency"),
        "user_id": txn.get("user_id"),
        "timestamp": txn.get("timestamp"),
        "alert_time": datetime.now(timezone.utc).isoformat()
    }).encode()
    req = urllib.request.Request(
        LOGIC_APP_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            logging.info(f"Alert sent, status: {resp.status}")
    except Exception as e:
        logging.error(f"Alert failed: {e}")

@app.service_bus_queue_trigger(
    arg_name="msg",
    queue_name="scored-transactions",
    connection="SERVICEBUS_CONNECTION_STRING"
)
def realtime_action(msg: func.ServiceBusMessage):
    try:
        body = msg.get_body().decode("utf-8")
        txn = json.loads(body)
        fraud_score = txn.get("fraud_score", 0.0)
        txn_id = txn.get("transaction_id")

        logging.info(f"Action check: {txn_id} score={fraud_score}")

        if fraud_score > FRAUD_THRESHOLD:
            logging.warning(f"FRAUD DETECTED: {txn_id} score={fraud_score}")
            send_alert(txn, fraud_score)
        else:
            logging.info(f"Transaction {txn_id} is legitimate")

    except Exception as e:
        logging.error(f"Realtime action error: {e}")
        raise
import azure.functions as func
import json
import joblib
import numpy as np
import pandas as pd
import logging
import os
import tempfile
from azure.storage.blob import BlobServiceClient

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

MODEL = None
PREPROCESSOR = None
TEMP_DIR = tempfile.gettempdir()

NUMERICAL_FEATURES = [
    "Time", "Amount", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"
]

def load_model():
    global MODEL, PREPROCESSOR
    if MODEL is not None:
        return
    logging.info("Loading model from Blob Storage...")
    conn_str = os.environ["STORAGE_CONNECTION_STRING"]
    client = BlobServiceClient.from_connection_string(conn_str)
    for name in ["model.joblib", "preprocessor.joblib"]:
        blob = client.get_blob_client(container="ml-models", blob=name)
        path = os.path.join(TEMP_DIR, name)
        with open(path, "wb") as f:
            f.write(blob.download_blob().readall())
        logging.info(f"Downloaded {name} to {path}")
    MODEL = joblib.load(os.path.join(TEMP_DIR, "model.joblib"))
    PREPROCESSOR = joblib.load(os.path.join(TEMP_DIR, "preprocessor.joblib"))
    logging.info("Model loaded successfully")

@app.route(route="predict", methods=["POST"])
def predict(req: func.HttpRequest) -> func.HttpResponse:
    try:
        load_model()
        body = req.get_json()
        features = body.get("features", {})
        values = [features.get(f, 0.0) for f in NUMERICAL_FEATURES]
        df = pd.DataFrame([values], columns=NUMERICAL_FEATURES)
        scaled = PREPROCESSOR.transform(df)
        score = float(MODEL.predict_proba(scaled)[0][1])
        return func.HttpResponse(
            json.dumps({"fraud_score": round(score, 4), "is_fraud": score > 0.5}),
            mimetype="application/json", status_code=200
        )
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json", status_code=500
        )

@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps({"status": "healthy"}),
        mimetype="application/json", status_code=200
    )
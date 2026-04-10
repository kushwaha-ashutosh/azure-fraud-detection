import os

files = {

'functions/ml_inference/requirements.txt': """azure-functions
scikit-learn==1.7.2
xgboost==3.2.0
joblib
numpy
pandas
azure-storage-blob
azure-identity
""",

'functions/ml_inference/function_app.py': """
import azure.functions as func
import json
import joblib
import numpy as np
import pandas as pd
import logging
import os
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

MODEL = None
PREPROCESSOR = None

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
    for name, var in [("model.joblib", "model"), ("preprocessor.joblib", "pre")]:
        blob = client.get_blob_client(container="ml-models", blob=name)
        path = f"/tmp/{name}"
        with open(path, "wb") as f:
            f.write(blob.download_blob().readall())
    MODEL = joblib.load("/tmp/model.joblib")
    PREPROCESSOR = joblib.load("/tmp/preprocessor.joblib")
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
            json.dumps({"fraud_score": score, "is_fraud": score > 0.5}),
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
""",

'functions/ml_inference/local.settings.json': """{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "STORAGE_CONNECTION_STRING": "REPLACE_WITH_YOUR_CONNECTION_STRING"
  }
}
""",

'tests/test_ml_function.py': """
import json
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import joblib
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

NUMERICAL_FEATURES = [
    "Time", "Amount", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"
]

def test_model_predict_shape():
    model = joblib.load("ml_training/models/model.joblib")
    pre = joblib.load("ml_training/models/preprocessor.joblib")
    values = [0.0] * 30
    df = pd.DataFrame([values], columns=NUMERICAL_FEATURES)
    scaled = pre.transform(df)
    result = model.predict_proba(scaled)
    assert result.shape == (1, 2)

def test_fraud_score_range():
    model = joblib.load("ml_training/models/model.joblib")
    pre = joblib.load("ml_training/models/preprocessor.joblib")
    for _ in range(10):
        values = [float(x) for x in np.random.rand(30)]
        df = pd.DataFrame([values], columns=NUMERICAL_FEATURES)
        scaled = pre.transform(df)
        score = float(model.predict_proba(scaled)[0][1])
        assert 0.0 <= score <= 1.0

def test_feature_count():
    assert len(NUMERICAL_FEATURES) == 30

def test_model_loads():
    model = joblib.load("ml_training/models/model.joblib")
    assert model is not None
"""
}

for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"wrote {path}")

print("All function files written OK")
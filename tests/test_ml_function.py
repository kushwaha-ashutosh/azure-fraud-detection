import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.abspath("."))

NUMERICAL_FEATURES = [
    "Time", "Amount", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"
]

def get_model():
    """Load model from Blob if available, otherwise skip."""
    import joblib
    model_path = "ml_training/models/model.joblib"
    pre_path = "ml_training/models/preprocessor.joblib"
    if os.path.exists(model_path) and os.path.exists(pre_path):
        return joblib.load(model_path), joblib.load(pre_path)
    conn = os.environ.get("STORAGE_CONNECTION_STRING")
    if not conn:
        return None, None
    import tempfile
    from azure.storage.blob import BlobServiceClient
    client = BlobServiceClient.from_connection_string(conn)
    tmp = tempfile.gettempdir()
    for name in ["model.joblib", "preprocessor.joblib"]:
        blob = client.get_blob_client("ml-models", name)
        path = os.path.join(tmp, name)
        with open(path, "wb") as f:
            f.write(blob.download_blob().readall())
    return joblib.load(os.path.join(tmp, "model.joblib")), joblib.load(os.path.join(tmp, "preprocessor.joblib"))

def test_feature_count():
    assert len(NUMERICAL_FEATURES) == 30

def test_model_predict_shape():
    import joblib
    model, pre = get_model()
    if model is None:
        pytest.skip("Model not available in CI - tested via Blob download")
    values = [0.0] * 30
    df = pd.DataFrame([values], columns=NUMERICAL_FEATURES)
    scaled = pre.transform(df)
    result = model.predict_proba(scaled)
    assert result.shape == (1, 2)

def test_fraud_score_range():
    model, pre = get_model()
    if model is None:
        pytest.skip("Model not available in CI - tested via Blob download")
    for _ in range(10):
        values = [float(x) for x in np.random.rand(30)]
        df = pd.DataFrame([values], columns=NUMERICAL_FEATURES)
        scaled = pre.transform(df)
        score = float(model.predict_proba(scaled)[0][1])
        assert 0.0 <= score <= 1.0

def test_model_loads():
    model, pre = get_model()
    if model is None:
        pytest.skip("Model not available in CI - tested via Blob download")
    assert model is not None
    assert pre is not None
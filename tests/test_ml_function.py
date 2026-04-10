import pytest
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath("."))

NUMERICAL_FEATURES = [
    "Time", "Amount", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"
]

MODEL_PATH = "ml_training/models/model.joblib"
PRE_PATH = "ml_training/models/preprocessor.joblib"
MODEL_AVAILABLE = os.path.exists(MODEL_PATH) and os.path.exists(PRE_PATH)

def test_feature_count():
    assert len(NUMERICAL_FEATURES) == 30

@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model files not in CI - tested via deployed endpoint")
def test_model_predict_shape():
    import joblib
    model = joblib.load(MODEL_PATH)
    pre = joblib.load(PRE_PATH)
    values = [0.0] * 30
    df = pd.DataFrame([values], columns=NUMERICAL_FEATURES)
    scaled = pre.transform(df)
    result = model.predict_proba(scaled)
    assert result.shape == (1, 2)

@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model files not in CI - tested via deployed endpoint")
def test_fraud_score_range():
    import joblib
    model = joblib.load(MODEL_PATH)
    pre = joblib.load(PRE_PATH)
    for _ in range(10):
        values = [float(x) for x in np.random.rand(30)]
        df = pd.DataFrame([values], columns=NUMERICAL_FEATURES)
        scaled = pre.transform(df)
        score = float(model.predict_proba(scaled)[0][1])
        assert 0.0 <= score <= 1.0

@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model files not in CI - tested via deployed endpoint")
def test_model_loads():
    import joblib
    model = joblib.load(MODEL_PATH)
    pre = joblib.load(PRE_PATH)
    assert model is not None
    assert pre is not None

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

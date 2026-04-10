
import os
import joblib
from ml_training.fraud_detection.config import MODEL_OUTPUT_DIR, MODEL_PATH, PREPROCESSOR_PATH

def save_model(model, preprocessor):
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")

def load_model():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("Model and preprocessor loaded successfully")
    return model, preprocessor

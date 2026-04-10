files = {
'ml_training/__init__.py': '',
'ml_training/fraud_detection/__init__.py': '',

'ml_training/fraud_detection/config.py': """
import os

DATA_PATH = "ml_training/data/creditcard.csv"
MODEL_OUTPUT_DIR = "ml_training/models"
MODEL_PATH = "ml_training/models/model.joblib"
PREPROCESSOR_PATH = "ml_training/models/preprocessor.joblib"

TARGET_COLUMN = "Class"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_TRIALS = 20
N_SPLITS_CV = 3

NUMERICAL_FEATURES = [
    "Time", "Amount", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"
]
""",

'ml_training/fraud_detection/data_preparation.py': """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from ml_training.fraud_detection.config import (
    DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, NUMERICAL_FEATURES
)

def load_data():
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df[TARGET_COLUMN].mean():.4f}")
    return df

def prepare_data(df):
    X = df[NUMERICAL_FEATURES]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train fraud count: {y_train.sum()}, Test fraud count: {y_test.sum()}")
    return X_train, X_test, y_train, y_test

def build_preprocessor():
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return preprocessor
""",

'ml_training/fraud_detection/training.py': """
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from ml_training.fraud_detection.config import N_TRIALS, N_SPLITS_CV, RANDOM_STATE

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, X_train, y_train, preprocessor):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 100),
        "random_state": RANDOM_STATE,
        "eval_metric": "auc",
        "use_label_encoder": False,
    }

    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        X_tr_scaled = preprocessor.fit_transform(X_tr)
        X_val_scaled = preprocessor.transform(X_val)

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr_scaled, y_tr, verbose=False)
        preds = model.predict_proba(X_val_scaled)[:, 1]
        auc_scores.append(roc_auc_score(y_val, preds))

    return np.mean(auc_scores)

def run_training(X_train, y_train, preprocessor):
    print(f"Running Optuna hyperparameter search ({N_TRIALS} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, preprocessor),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )
    print(f"Best AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_params = {**study.best_params, "random_state": RANDOM_STATE, "eval_metric": "auc"}
    X_train_scaled = preprocessor.fit_transform(X_train)
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train_scaled, y_train, verbose=False)
    print("Final model trained on full training set")
    return final_model
""",

'ml_training/fraud_detection/evaluation.py': """
import numpy as np
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve, average_precision_score
)

def evaluate_model(model, preprocessor, X_test, y_test):
    X_test_scaled = preprocessor.transform(X_test)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["legit", "fraud"])

    print(f"AUC-ROC:          {auc:.4f}")
    print(f"Avg Precision:    {ap:.4f}")
    print(f"Confusion Matrix:\\n{cm}")
    print(f"Classification Report:\\n{report}")

    return {"auc_roc": auc, "avg_precision": ap, "confusion_matrix": cm.tolist()}
""",

'ml_training/fraud_detection/model_exporter.py': """
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
""",

'ml_training/fraud_detection/main.py': """
from ml_training.fraud_detection.data_preparation import load_data, prepare_data, build_preprocessor
from ml_training.fraud_detection.training import run_training
from ml_training.fraud_detection.evaluation import evaluate_model
from ml_training.fraud_detection.model_exporter import save_model

def main():
    print("=== Fraud Detection ML Training Pipeline ===")
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    preprocessor = build_preprocessor()
    model = run_training(X_train, y_train, preprocessor)
    metrics = evaluate_model(model, preprocessor, X_test, y_test)
    save_model(model, preprocessor)
    print("=== Training complete ===")
    return metrics

if __name__ == "__main__":
    main()
"""
}

for path, content in files.items():
    with open(path, 'w') as f:
        f.write(content)
    print(f"wrote {path}")

print("All training files written OK")

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

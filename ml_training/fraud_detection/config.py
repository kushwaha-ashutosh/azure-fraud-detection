
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

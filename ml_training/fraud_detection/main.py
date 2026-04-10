
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

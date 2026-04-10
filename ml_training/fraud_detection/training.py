
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


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
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{report}")

    return {"auc_roc": auc, "avg_precision": ap, "confusion_matrix": cm.tolist()}

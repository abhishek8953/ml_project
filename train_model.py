"""
Credit Card Fraud Detection - Model Training Script

Dataset:
Download creditcard.csv from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Target:
Class = 0 means Genuine transaction
Class = 1 means Fraud transaction
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DATASET_PATH = "creditcard.csv"
MODEL_PATH = "fraud_model.pkl"
SCALER_PATH = "scaler.pkl"
METRICS_PATH = "model_metrics.pkl"


def load_dataset(path):
    """Load and validate dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download creditcard.csv and place it in this folder."
        )

    data = pd.read_csv(path)

    required_columns = ["Time", "Amount", "Class"]
    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Missing required column: {column}")

    return data


def train_model():
    """Train fraud detection model and save model files."""
    print("Loading dataset...")
    data = load_dataset(DATASET_PATH)

    print("Dataset shape:", data.shape)
    print("\nClass distribution:")
    print(data["Class"].value_counts())

    X = data.drop("Class", axis=1)
    y = data["Class"]

    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTraining model...")

    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("\nEvaluating model...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": classification_report(y_test, y_pred),
        "feature_columns": list(X.columns),
    }

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(metrics["classification_report"])

    print("\nScores:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(metrics, METRICS_PATH)

    print("\nModel saved successfully.")
    print(f"Saved: {MODEL_PATH}")
    print(f"Saved: {SCALER_PATH}")
    print(f"Saved: {METRICS_PATH}")


if __name__ == "__main__":
    train_model()

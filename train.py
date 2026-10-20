"""Training script for the SENG 264 ML assignment.

This script trains the best-performing allowed model on the provided training
data, using the same preprocessing pipeline as inference.

Output:
- model.ubj
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from preprocess import preprocess


def compute_sample_weights(y: pd.Series) -> np.ndarray:
    """Inverse-frequency weights for class imbalance."""
    counts = y.value_counts().sort_index()
    class_weights = {
        cls: len(y) / (len(counts) * count)
        for cls, count in counts.items()
    }
    return y.map(class_weights).to_numpy(dtype=float)


def main() -> None:
    x_path = "X_train.csv"
    y_path = "y_train.csv"
    model_path = "model.ubj"

    print(f"Loading {x_path} and {y_path} ...")
    X_raw = pd.read_csv(x_path)
    y = pd.read_csv(y_path).iloc[:, 0]

    print("Preprocessing training features ...")
    X = preprocess(X_raw)

    print("Splitting data into train and test sets ...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,      # 20% testing
        random_state=42,
        stratify=y           # keeps class proportions balanced
    )

    # =========================
    # SAMPLE WEIGHTS
    # =========================
    sample_weight = compute_sample_weights(y_train)


    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    print("Training XGBoost model ...")
    model.fit(X_train, y_train, sample_weight=sample_weight)

    #predıctıon

    print("Making predictions on test set ...")

    y_pred = model.predict(X_test)

    # probability predictions
    y_prob = model.predict_proba(X_test)

    # =========================
    # EVALUATION
    # =========================
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    composite_score = (0.5 * accuracy+ 0.5 * macro_f1)

    print(f"Macro F1 Score: {macro_f1:.4f}")

    print("\n========== RESULTS ==========")

    print(f"Accuracy: {accuracy:.4f}")

    print(f"Composite Score: {composite_score:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"Saving model to {model_path} ...")
    model.save_model(model_path)

    summary = {
        "framework": "xgboost",
        "model_file": model_path,
    }
    print("Training complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

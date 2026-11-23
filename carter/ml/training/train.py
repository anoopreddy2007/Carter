"""
Offline training pipeline for telemetry anomaly detection.
Supports IsolationForest and baseline evaluation metrics (Precision, Recall, F1, Confusion Matrix).
Saves model artifacts, config, dataset hash, seed, and metrics (FR-27, FR-28, FR-29, FR-31).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
import joblib

from carter.ml.dataset import (
    generate_synthetic_telemetry,
    save_dataset_csv,
    load_dataset_csv,
    FEATURE_NAMES,
)
from carter.utils.logging import get_logger

logger = get_logger("ml.training")


def train_anomaly_model(
    data_path: Path | str | None = None,
    output_dir: Path | str = "artifacts/models",
    n_estimators: int = 100,
    contamination: float = 0.05,
    seed: int = 42,
    num_samples: int = 1500,
) -> dict[str, Any]:
    """
    Train an Isolation Forest anomaly detection model on telemetry data.

    Steps:
    1. Load existing data or generate reproducible synthetic telemetry.
    2. Split into Train (70%), Val (15%), and Test (15%) splits deterministically.
    3. Fit IsolationForest on Train set.
    4. Evaluate on Test set (Precision, Recall, F1, Confusion Matrix).
    5. Save model (.joblib), metadata (.json), and test metrics.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Acquisition
    if data_path and Path(data_path).exists():
        X, y, feature_names = load_dataset_csv(data_path)
        dataset_meta = {"path": str(data_path), "sha256": "pre-existing"}
    else:
        logger.info("Generating synthetic telemetry dataset (%d samples)...", num_samples)
        X, y = generate_synthetic_telemetry(num_samples=num_samples, anomaly_ratio=contamination, seed=seed)
        csv_path = out_dir / "telemetry_data.csv"
        dataset_meta = save_dataset_csv(X, y, csv_path)
        feature_names = FEATURE_NAMES

    # 2. Deterministic Split: 70% Train, 15% Val, 15% Test
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X))
    n_train = int(len(X) * 0.70)
    n_val = int(len(X) * 0.15)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    logger.info("Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))

    # 3. Model Training
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train)

    # 4. Evaluation on held-out test set
    y_test_pred = model.predict(X_test)  # returns 1 for inliers, -1 for outliers

    # Compute binary metrics (treating -1 anomaly as positive class for evaluation)
    # Convert: normal=0, anomaly=1
    y_true_binary = (y_test == -1).astype(int)
    y_pred_binary = (y_test_pred == -1).astype(int)

    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "test_samples": len(X_test),
        "anomalies_in_test": int(y_true_binary.sum()),
    }

    # 5. Save Artifacts
    model_artifact_path = out_dir / "isolation_forest.joblib"
    joblib.dump(model, model_artifact_path)

    metadata = {
        "model_type": "sklearn",
        "algorithm": "IsolationForest",
        "hyperparameters": {
            "n_estimators": n_estimators,
            "contamination": contamination,
            "random_state": seed,
        },
        "feature_names": feature_names,
        "dataset_hash": dataset_meta.get("sha256"),
        "seed": seed,
        "metrics": metrics,
        "model_artifact": str(model_artifact_path),
    }

    meta_path = out_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model saved to %s (F1 Score: %.4f)", model_artifact_path, f1)

    return metadata


if __name__ == "__main__":
    import sys
    print("Training offline Carter anomaly detection model...")
    meta = train_anomaly_model()
    print(json.dumps(meta, indent=2))


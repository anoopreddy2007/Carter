"""
Offline model inference runner.
Loads trained scikit-learn anomaly detection model artifact and executes predictions (FR-30).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import joblib

from carter.utils.logging import get_logger

logger = get_logger("ml.inference")


class AnomalyDetector:
    """
    Dedicated local inference runner for Carter anomaly models.
    """

    def __init__(self, model_path: Path | str, metadata_path: Path | str | None = None) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")

        self.model = joblib.load(self.model_path)

        self.metadata = {}
        if metadata_path:
            meta_p = Path(metadata_path)
            if meta_p.exists():
                with open(meta_p, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

        logger.info("AnomalyDetector loaded model from %s", self.model_path)

    def predict(self, features: list[float] | np.ndarray) -> dict[str, Any]:
        """
        Run inference on feature input.

        Args:
            features: 1D sequence of float features matching model input size.

        Returns:
            Dict containing:
            - is_anomaly (bool): True if sample is an outlier/anomaly
            - raw_prediction (int): 1 (normal) or -1 (anomaly)
            - anomaly_score (float): Anomaly score from decision_function
        """
        arr = np.asarray(features, dtype=float).reshape(1, -1)

        raw_pred = int(self.model.predict(arr)[0])
        is_anomaly = raw_pred == -1

        score = None
        if hasattr(self.model, "decision_function"):
            score = float(self.model.decision_function(arr)[0])

        return {
            "is_anomaly": is_anomaly,
            "raw_prediction": raw_pred,
            "anomaly_score": round(score, 4) if score is not None else None,
            "status": "anomaly" if is_anomaly else "normal",
        }


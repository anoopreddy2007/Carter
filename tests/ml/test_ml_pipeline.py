"""
Tests for Carter offline ML module: dataset generation, training pipeline, and inference runner.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import pytest
import numpy as np

from carter.ml.dataset import (
    generate_synthetic_telemetry,
    save_dataset_csv,
    load_dataset_csv,
    FEATURE_NAMES,
)
from carter.ml.inference.predict import AnomalyDetector


class TestMLDataset:
    def test_generate_synthetic_telemetry(self):
        X, y = generate_synthetic_telemetry(num_samples=200, anomaly_ratio=0.1, seed=42)
        assert X.shape == (200, 5)
        assert len(y) == 200
        # 10% anomalies -> 20 outliers (-1) and 180 normal (1)
        assert (y == -1).sum() == 20
        assert (y == 1).sum() == 180

    def test_dataset_csv_roundtrip(self, tmp_path: Path):
        X, y = generate_synthetic_telemetry(num_samples=50, anomaly_ratio=0.1, seed=123)
        csv_file = tmp_path / "telemetry.csv"

        meta = save_dataset_csv(X, y, csv_file)
        assert csv_file.exists()
        assert meta["num_rows"] == 50
        assert meta["num_features"] == 5
        assert len(meta["sha256"]) == 64

        X_loaded, y_loaded, names = load_dataset_csv(csv_file)
        assert names == FEATURE_NAMES
        assert X_loaded.shape == (50, 5)
        assert len(y_loaded) == 50
        np.testing.assert_allclose(X_loaded, np.round(X, 4))
        np.testing.assert_array_equal(y_loaded, y)


class TestMLInference:
    def test_detector_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Model artifact not found"):
            AnomalyDetector(tmp_path / "nonexistent.joblib")

    def test_detector_predict_with_mock(self, tmp_path: Path, monkeypatch):
        dummy_file = tmp_path / "model.joblib"
        dummy_file.write_bytes(b"dummy")

        mock_model = MagicMock()
        mock_model.predict.return_value = [-1]
        mock_model.decision_function.return_value = [-0.1523]

        monkeypatch.setattr("joblib.load", lambda p: mock_model)

        detector = AnomalyDetector(dummy_file)
        res = detector.predict([95.0, 80.0, 50.0, 300.0, 500.0])

        assert res["is_anomaly"] is True
        assert res["raw_prediction"] == -1
        assert res["status"] == "anomaly"
        assert res["anomaly_score"] == -0.1523


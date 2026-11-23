"""
Tests for predict_anomaly tool (sklearn pipeline, validation, and error cases).
Uses mock objects to avoid sandbox Application Control restrictions on C DLLs (scipy/numpy).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from carter.tools.predict_anomaly import predict_anomaly


class TestPredictAnomaly:
    @pytest.fixture
    def dummy_model_file(self, tmp_path: Path) -> Path:
        """Create a placeholder model file on disk."""
        model_path = tmp_path / "iso_forest.joblib"
        model_path.write_bytes(b"dummy_model_artifact")
        return model_path

    def test_predict_anomaly_success_with_mock(self, dummy_model_file: Path, monkeypatch):
        mock_model = MagicMock(spec=["predict", "decision_function"])
        mock_model.predict.return_value = [1]
        mock_model.decision_function.return_value = [0.1234]

        monkeypatch.setattr("joblib.load", lambda path: mock_model)

        result = predict_anomaly([1.0, 2.0], str(dummy_model_file), model_type="sklearn")
        assert result["model_type"] == "sklearn"
        assert result["prediction"] == 1
        assert "confidence" in result
        assert result["confidence"] == 0.1234

    def test_invalid_features_not_a_list(self, dummy_model_file: Path):
        with pytest.raises(ValueError, match="features must be a list of numbers"):
            predict_anomaly("not-a-list", str(dummy_model_file))  # type: ignore

    def test_empty_features_list(self, dummy_model_file: Path):
        with pytest.raises(ValueError, match="features list is empty"):
            predict_anomaly([], str(dummy_model_file))

    def test_too_many_features(self, dummy_model_file: Path):
        with pytest.raises(ValueError, match="Too many features"):
            predict_anomaly([1.0, 2.0, 3.0], str(dummy_model_file), max_features=2)

    def test_non_numeric_feature_element(self, dummy_model_file: Path):
        with pytest.raises(ValueError, match="Feature at index 1 is not a number"):
            predict_anomaly([1.0, "abc"], str(dummy_model_file))  # type: ignore

    def test_model_file_not_found(self, tmp_path: Path):
        missing = tmp_path / "missing.joblib"
        with pytest.raises(FileNotFoundError, match="ML model artifact not found"):
            predict_anomaly([1.0, 2.0], str(missing))

    def test_unknown_model_type(self, dummy_model_file: Path):
        with pytest.raises(ValueError, match="Unknown model_type"):
            predict_anomaly([1.0, 2.0], str(dummy_model_file), model_type="xgboost")

    def test_pytorch_unimplemented_stub(self, tmp_path: Path):
        dummy_pt = tmp_path / "model.pt"
        dummy_pt.write_bytes(b"dummy")

        with pytest.raises(Exception):
            predict_anomaly([1.0, 2.0], str(dummy_pt), model_type="pytorch")

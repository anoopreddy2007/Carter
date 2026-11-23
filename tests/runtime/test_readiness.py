"""
Tests for runtime readiness checks.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from carter.runtime.readiness import (
    check_llm,
    check_ml_model,
    check_memory,
    check_workspace,
    run_readiness_check,
)


class TestReadiness:
    def test_check_llm_missing_config(self):
        status = check_llm({})
        assert status.ready is False
        assert "No model_path configured" in status.message

    def test_check_llm_file_not_found(self, tmp_path: Path):
        cfg = {"llm": {"model_path": str(tmp_path / "model.gguf")}}
        status = check_llm(cfg)
        assert status.ready is False
        assert "Model file not found" in status.message

    def test_check_llm_file_too_small(self, tmp_path: Path):
        tiny = tmp_path / "tiny.gguf"
        tiny.write_bytes(b"0" * 1024)  # 1 KB < 10 MB
        cfg = {"llm": {"model_path": str(tiny)}}
        status = check_llm(cfg)
        assert status.ready is False
        assert "looks too small" in status.message

    def test_check_llm_valid(self, tmp_path: Path):
        valid = tmp_path / "valid.gguf"
        # Write 11 MB
        valid.write_bytes(b"\0" * (11 * 1024 * 1024))
        cfg = {"llm": {"model_path": str(valid)}}
        status = check_llm(cfg)
        assert status.ready is True
        assert "Model file found" in status.message

    def test_check_ml_model_missing_config(self):
        status = check_ml_model({})
        assert status.ready is False
        assert "No ml.model_path configured" in status.message

    def test_check_ml_model_file_not_found(self, tmp_path: Path):
        cfg = {"ml": {"model_path": str(tmp_path / "missing.joblib")}}
        status = check_ml_model(cfg)
        assert status.ready is False
        assert "ML model artifact not found" in status.message

    def test_check_ml_model_valid(self, tmp_path: Path):
        model_file = tmp_path / "model.joblib"
        model_file.write_bytes(b"artifact")
        cfg = {"ml": {"model_path": str(model_file)}}
        status = check_ml_model(cfg)
        assert status.ready is True
        assert "ML model artifact found" in status.message

    def test_check_memory_creatable_and_writable(self, tmp_path: Path):
        db_path = tmp_path / "artifacts" / "carter.db"
        cfg = {"memory": {"db_path": str(db_path)}}
        status = check_memory(cfg)
        assert status.ready is True
        assert db_path.parent.exists()

    def test_check_workspace_creates_directory(self, tmp_path: Path):
        ws_dir = tmp_path / "new_workspace"
        cfg = {"workspace": {"root": str(ws_dir)}}
        assert not ws_dir.exists()
        status = check_workspace(cfg)
        assert status.ready is True
        assert ws_dir.exists()

    def test_run_readiness_check_all_ready(self, tmp_path: Path):
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"\0" * (11 * 1024 * 1024))

        cfg = {
            "llm": {"model_path": str(model_file)},
            "memory": {"db_path": str(tmp_path / "art" / "carter.db")},
            "workspace": {"root": str(tmp_path / "ws")},
            "ml": {"model_path": ""},
        }
        ready = run_readiness_check(cfg)
        assert ready is True

    def test_run_readiness_check_fails_on_missing_required(self, tmp_path: Path):
        cfg = {
            "llm": {"model_path": str(tmp_path / "missing.gguf")},
            "memory": {"db_path": str(tmp_path / "art" / "carter.db")},
            "workspace": {"root": str(tmp_path / "ws")},
        }
        ready = run_readiness_check(cfg)
        assert ready is False


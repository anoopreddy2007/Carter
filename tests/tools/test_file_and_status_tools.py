"""
Tests for read_file, list_files, and get_system_status tools.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from carter.tools.read_file import read_file
from carter.tools.list_files import list_files, MAX_ENTRIES
from carter.tools.system_status import get_system_status


class TestReadFile:
    def test_read_existing_text_file(self, tmp_path: Path):
        file_path = tmp_path / "hello.txt"
        file_path.write_text("Hello, Carter!\nSecond line.\n", encoding="utf-8")

        result = read_file("hello.txt", str(tmp_path))
        assert result["filepath"] == "hello.txt"
        assert result["content"] == "Hello, Carter!\nSecond line.\n"
        assert result["lines"] == 2
        assert result["size_bytes"] == len(file_path.read_bytes())

    def test_read_file_without_trailing_newline(self, tmp_path: Path):
        file_path = tmp_path / "single.txt"
        file_path.write_text("No newline", encoding="utf-8")

        result = read_file("single.txt", str(tmp_path))
        assert result["lines"] == 1
        assert result["content"] == "No newline"

    def test_read_empty_file(self, tmp_path: Path):
        file_path = tmp_path / "empty.txt"
        file_path.write_text("", encoding="utf-8")

        result = read_file("empty.txt", str(tmp_path))
        assert result["content"] == ""
        assert result["lines"] == 0
        assert result["size_bytes"] == 0

    def test_read_nonexistent_file_raises_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="File not found in workspace"):
            read_file("does_not_exist.txt", str(tmp_path))

    def test_read_directory_raises_value_error(self, tmp_path: Path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        with pytest.raises(ValueError, match="is a directory, not a file"):
            read_file("subdir", str(tmp_path))

    def test_read_file_exceeds_max_size_raises(self, tmp_path: Path):
        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * 100, encoding="utf-8")

        with pytest.raises(ValueError, match="too large"):
            read_file("large.txt", str(tmp_path), max_size_bytes=50)

    def test_path_traversal_raises_value_error(self, tmp_path: Path):
        with pytest.raises(ValueError, match="traversal|outside permitted workspace"):
            read_file("../secret.txt", str(tmp_path))

    def test_non_utf8_file_raises_value_error(self, tmp_path: Path):
        binary_file = tmp_path / "bad.bin"
        binary_file.write_bytes(b"\x80\x81\xFF\xFE")

        with pytest.raises(ValueError, match="not valid UTF-8"):
            read_file("bad.bin", str(tmp_path))


class TestListFiles:
    def test_list_files_root(self, tmp_path: Path):
        (tmp_path / "file1.txt").write_text("1", encoding="utf-8")
        (tmp_path / "file2.txt").write_text("22", encoding="utf-8")
        (tmp_path / "sub").mkdir()

        result = list_files(str(tmp_path))
        assert result["path"] == "."
        assert result["truncated"] is False
        names = {e["name"] for e in result["entries"]}
        assert names == {"file1.txt", "file2.txt", "sub"}

        file2_entry = next(e for e in result["entries"] if e["name"] == "file2.txt")
        assert file2_entry["type"] == "file"
        assert file2_entry["size_bytes"] == 2

        sub_entry = next(e for e in result["entries"] if e["name"] == "sub")
        assert sub_entry["type"] == "directory"

    def test_list_files_subdirectory(self, tmp_path: Path):
        sub = tmp_path / "child"
        sub.mkdir()
        (sub / "inner.txt").write_text("inner", encoding="utf-8")

        result = list_files(str(tmp_path), "child")
        assert result["path"] == "child"
        assert len(result["entries"]) == 1
        assert result["entries"][0]["name"] == "inner.txt"

    def test_list_nonexistent_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            list_files(str(tmp_path), "missing_dir")

    def test_list_file_as_directory_raises(self, tmp_path: Path):
        file_path = tmp_path / "regular.txt"
        file_path.write_text("content", encoding="utf-8")
        with pytest.raises(ValueError, match="Not a directory"):
            list_files(str(tmp_path), "regular.txt")

    def test_list_traversal_raises_value_error(self, tmp_path: Path):
        with pytest.raises(ValueError, match="traversal|outside permitted workspace"):
            list_files(str(tmp_path), "../")

    def test_list_files_truncation_at_max_entries(self, tmp_path: Path, monkeypatch):
        for i in range(10):
            (tmp_path / f"file_{i:02d}.txt").write_text("a", encoding="utf-8")

        import carter.tools.list_files as lf_module
        monkeypatch.setattr(lf_module, "MAX_ENTRIES", 5)

        result = list_files(str(tmp_path))
        assert len(result["entries"]) == 5
        assert result["truncated"] is True


class TestSystemStatus:
    def test_system_status_reports_correct_structure(self, tmp_path: Path):
        dummy_llm = tmp_path / "model.gguf"
        dummy_llm.write_bytes(b"0" * 1024)

        dummy_ml = tmp_path / "model.joblib"
        dummy_ml.write_bytes(b"0" * 512)

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        db_path = artifacts_dir / "carter.db"
        db_path.write_bytes(b"")

        ws_dir = tmp_path / "workspace"
        ws_dir.mkdir()

        cfg = {
            "llm": {
                "model_path": str(dummy_llm),
                "model_id": "test-1b",
                "quantization": "Q4_K_M",
                "context_length": 2048,
            },
            "ml": {
                "model_path": str(dummy_ml),
                "model_type": "sklearn",
            },
            "memory": {
                "db_path": str(db_path),
            },
            "workspace": {
                "root": str(ws_dir),
            },
            "agent": {
                "max_steps": 7,
            },
        }

        status = get_system_status(cfg)
        assert status["llm"]["ready"] is True
        assert status["llm"]["model_id"] == "test-1b"
        assert status["ml_model"]["ready"] is True
        assert status["memory"]["ready"] is True
        assert status["memory"]["db_exists"] is True
        assert status["workspace"]["ready"] is True
        assert status["agent"]["max_steps"] == 7

    def test_system_status_missing_components(self, tmp_path: Path):
        cfg = {
            "llm": {"model_path": str(tmp_path / "nonexistent.gguf")},
            "ml": {"model_path": ""},
            "memory": {"db_path": str(tmp_path / "no_parent" / "carter.db")},
            "workspace": {"root": str(tmp_path / "nonexistent_ws")},
        }

        status = get_system_status(cfg)
        assert status["llm"]["ready"] is False
        assert status["ml_model"]["ready"] is False
        assert status["memory"]["ready"] is False
        assert status["memory"]["db_exists"] is False
        assert status["workspace"]["ready"] is False


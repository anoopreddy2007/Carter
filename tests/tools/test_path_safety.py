"""
Tests for path safety utilities.
These are the most security-critical tests in the project.
"""

import os
import tempfile
from pathlib import Path

import pytest

from carter.utils.path_safety import safe_resolve, is_safe_path, PathTraversalError


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory with some files."""
    (tmp_path / "notes.txt").write_text("hello world")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "data.txt").write_text("data")
    return tmp_path


class TestSafeResolve:
    def test_simple_filename(self, workspace: Path) -> None:
        result = safe_resolve("notes.txt", workspace)
        assert result == workspace / "notes.txt"

    def test_subdirectory_access(self, workspace: Path) -> None:
        result = safe_resolve("subdir/data.txt", workspace)
        assert result == workspace / "subdir" / "data.txt"

    def test_workspace_root_itself(self, workspace: Path) -> None:
        result = safe_resolve("", workspace)
        # Empty string resolves to workspace root
        assert result == workspace.resolve()

    def test_path_traversal_dotdot(self, workspace: Path) -> None:
        with pytest.raises(PathTraversalError):
            safe_resolve("../secret.txt", workspace)

    def test_path_traversal_deep(self, workspace: Path) -> None:
        with pytest.raises(PathTraversalError):
            safe_resolve("subdir/../../etc/passwd", workspace)

    def test_absolute_path_outside_workspace(self, workspace: Path) -> None:
        with pytest.raises(PathTraversalError):
            safe_resolve("/etc/passwd", workspace)

    def test_windows_style_traversal(self, workspace: Path) -> None:
        with pytest.raises((PathTraversalError, ValueError)):
            safe_resolve("..\\..\\windows\\system32", workspace)

    def test_nonexistent_workspace(self, tmp_path: Path) -> None:
        bad_root = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            safe_resolve("file.txt", bad_root)


class TestIsSafePath:
    def test_safe_path_returns_true(self, workspace: Path) -> None:
        assert is_safe_path("notes.txt", workspace) is True

    def test_traversal_returns_false(self, workspace: Path) -> None:
        assert is_safe_path("../outside.txt", workspace) is False

    def test_nonexistent_workspace_returns_false(self, tmp_path: Path) -> None:
        assert is_safe_path("file.txt", tmp_path / "missing") is False

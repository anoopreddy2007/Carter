"""
Path safety utilities.
All file access from Carter's tools must go through these helpers.
They enforce the workspace boundary and protect against path traversal.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union


class PathTraversalError(ValueError):
    """Raised when a path escapes the permitted workspace root."""


def safe_resolve(
    requested: Union[str, Path],
    workspace_root: Union[str, Path],
) -> Path:
    """
    Resolve `requested` relative to `workspace_root` and verify it does not
    escape the workspace via path traversal (e.g., `../../etc/passwd`),
    symbolic links, or absolute paths that bypass the root.

    Returns the resolved absolute Path if safe.
    Raises PathTraversalError if the path is outside the workspace.

    Args:
        requested: The filename or relative path provided by the agent/tool.
        workspace_root: The permitted workspace directory.

    Raises:
        PathTraversalError: If the resolved path is outside workspace_root.
        FileNotFoundError: If the workspace_root itself does not exist.
    """
    root = Path(workspace_root).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Workspace root does not exist: {root}")

    # Resolve without following symlinks first to detect symlink escapes
    candidate = (root / requested).resolve()

    # Check containment
    try:
        candidate.relative_to(root)
    except ValueError:
        raise PathTraversalError(
            f"Path '{requested}' resolves outside the workspace root '{root}'. "
            "Access denied."
        )

    # Additional symlink check: the real path must also be inside the root
    try:
        real = candidate.resolve(strict=False)
        real.relative_to(root)
    except ValueError:
        raise PathTraversalError(
            f"Path '{requested}' follows a symlink outside the workspace. "
            "Access denied."
        )

    return candidate


def is_safe_path(
    requested: Union[str, Path],
    workspace_root: Union[str, Path],
) -> bool:
    """
    Return True if `requested` is safely within `workspace_root`, False otherwise.
    Convenience wrapper around safe_resolve that swallows exceptions.
    """
    try:
        safe_resolve(requested, workspace_root)
        return True
    except (PathTraversalError, FileNotFoundError, ValueError):
        return False

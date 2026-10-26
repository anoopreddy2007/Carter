"""
Tool: read_file
Reads a small UTF-8 text file from within the workspace.
Enforces path safety and file size limits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carter.utils.path_safety import safe_resolve, PathTraversalError
from carter.utils.logging import get_logger

logger = get_logger("tools.read_file")


def read_file(
    filepath: str,
    workspace_root: str,
    max_size_bytes: int = 524288,  # 512 KB default
) -> dict[str, Any]:
    """
    Read a UTF-8 text file from within the workspace.

    Args:
        filepath: Relative path to the file within the workspace.
        workspace_root: The permitted workspace root (from config).
        max_size_bytes: Maximum file size to read. Larger files are rejected.

    Returns:
        A dict with keys:
          - "filepath": the relative path that was read
          - "content": the file content as a string
          - "size_bytes": the file size
          - "lines": the number of lines

    Raises:
        ValueError: on path traversal or size limit exceeded.
        FileNotFoundError: if the file does not exist.
        UnicodeDecodeError: if the file is not valid UTF-8.
    """
    try:
        resolved = safe_resolve(filepath, workspace_root)
    except PathTraversalError as e:
        raise ValueError(str(e))

    if not resolved.exists():
        raise FileNotFoundError(f"File not found in workspace: {filepath}")

    if resolved.is_dir():
        raise ValueError(f"'{filepath}' is a directory, not a file.")

    size = resolved.stat().st_size
    if size > max_size_bytes:
        raise ValueError(
            f"File '{filepath}' is too large ({size:,} bytes). "
            f"Maximum allowed: {max_size_bytes:,} bytes."
        )

    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValueError(
            f"File '{filepath}' is not valid UTF-8. "
            "Carter can only read plain text files."
        )

    line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

    logger.info("read_file: read %d bytes, %d lines from '%s'", size, line_count, filepath)

    return {
        "filepath": filepath,
        "content": content,
        "size_bytes": size,
        "lines": line_count,
    }

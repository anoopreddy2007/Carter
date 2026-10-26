"""
Tool: list_files
Lists files within the configured workspace directory.
No path traversal permitted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from carter.utils.path_safety import safe_resolve, PathTraversalError
from carter.utils.logging import get_logger

logger = get_logger("tools.list_files")

# Maximum number of file entries to return
MAX_ENTRIES = 200


def list_files(workspace_root: str, subpath: str = "") -> dict[str, Any]:
    """
    List files in the workspace directory (or a subdirectory of it).

    Args:
        workspace_root: The permitted workspace root (from config).
        subpath: Optional relative subdirectory within the workspace.

    Returns:
        A dict with keys:
          - "entries": list of {"name", "type", "size_bytes"} dicts
          - "path": the directory that was listed (relative to workspace_root)
          - "truncated": True if MAX_ENTRIES was hit
    """
    try:
        if subpath:
            target = safe_resolve(subpath, workspace_root)
        else:
            target = Path(workspace_root).resolve()
    except PathTraversalError as e:
        raise ValueError(str(e))

    if not target.exists():
        raise FileNotFoundError(f"Directory not found: {subpath or '.'}")

    if not target.is_dir():
        raise ValueError(f"Not a directory: {subpath or '.'}")

    entries = []
    truncated = False

    for i, item in enumerate(sorted(target.iterdir())):
        if i >= MAX_ENTRIES:
            truncated = True
            break
        entry = {
            "name": item.name,
            "type": "directory" if item.is_dir() else "file",
        }
        if item.is_file():
            try:
                entry["size_bytes"] = item.stat().st_size
            except OSError:
                entry["size_bytes"] = None
        entries.append(entry)

    rel_path = str(target.relative_to(Path(workspace_root).resolve()))
    logger.info("list_files: listed %d entries in '%s'", len(entries), rel_path)

    return {
        "path": rel_path,
        "entries": entries,
        "truncated": truncated,
    }

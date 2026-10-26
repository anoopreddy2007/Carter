"""
Carter configuration loader.
Loads YAML config from a local file. Never fetches remote config.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit(
        "[Carter] ERROR: PyYAML is not installed. "
        "Run: pip install --no-index --find-links=<wheelhouse> pyyaml"
    )


def load_config(config_path: str = "configs/carter.yaml") -> dict[str, Any]:
    """
    Load and return Carter's configuration from a local YAML file.

    Raises SystemExit with a clear message if the file is missing or malformed.
    Does NOT attempt to download or fetch any remote resource.
    """
    path = Path(config_path)

    if not path.exists():
        example = path.parent / (path.stem + ".example" + path.suffix)
        sys.exit(
            f"[Carter] ERROR: Configuration file not found: {path}\n"
            f"  Copy the example and edit it:\n"
            f"    copy {example} {path}\n"
            f"  Then set your model_path and other settings."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        sys.exit(f"[Carter] ERROR: Failed to parse configuration file {path}:\n  {e}")

    if cfg is None:
        sys.exit(f"[Carter] ERROR: Configuration file is empty: {path}")

    return cfg


def get(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely retrieve a nested config value.

    Example:
        get(cfg, "llm", "model_path") -> cfg["llm"]["model_path"]
    """
    value = cfg
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
        if value is None:
            return default
    return value

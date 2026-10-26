"""
Carter structured logging setup.
Writes to stdout and to a rotating local log file. No remote logging.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Any


def setup_logging(cfg: dict[str, Any]) -> None:
    """
    Configure the root logger based on the [logging] section of carter.yaml.
    Creates the log directory if it does not exist.
    """
    log_cfg = cfg.get("logging", {})
    level_name = log_cfg.get("level", "INFO").upper()
    log_dir = Path(log_cfg.get("log_dir", "artifacts/logs"))
    max_mb = int(log_cfg.get("max_log_size_mb", 50))

    numeric_level = getattr(logging, level_name, logging.INFO)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "carter.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_mb * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger("carter").info(
        "Logging initialized. Level=%s, File=%s", level_name, log_file
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger under the 'carter' namespace."""
    return logging.getLogger(f"carter.{name}")

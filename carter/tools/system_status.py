"""
Tool: get_system_status
Reports the readiness of Carter's local components.
Does not expose secrets, credentials, or internal paths beyond what is safe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carter.utils.config import get
from carter.utils.logging import get_logger

logger = get_logger("tools.system_status")


def get_system_status(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Report Carter's component readiness status.

    Returns a safe, structured summary suitable for display in a chat response.
    Does NOT expose secrets, API keys, full filesystem paths, or internal configs.

    Args:
        cfg: The loaded Carter configuration dict.

    Returns:
        A dict with status information for each component.
    """
    # LLM status
    model_path = get(cfg, "llm", "model_path", default="")
    llm_ready = bool(model_path and Path(model_path).exists())
    llm_status = {
        "ready": llm_ready,
        "model_id": get(cfg, "llm", "model_id", default="unknown"),
        "quantization": get(cfg, "llm", "quantization", default="unknown"),
        "context_length": get(cfg, "llm", "context_length", default="unknown"),
    }

    # ML model status
    ml_path = get(cfg, "ml", "model_path", default="")
    ml_ready = bool(ml_path and Path(ml_path).exists())
    ml_status = {
        "ready": ml_ready,
        "model_type": get(cfg, "ml", "model_type", default="unknown"),
    }

    # Memory status
    db_path = Path(get(cfg, "memory", "db_path", default="artifacts/carter.db"))
    memory_status = {
        "ready": db_path.parent.exists(),
        "db_exists": db_path.exists(),
    }

    # Workspace status
    workspace_root = get(cfg, "workspace", "root", default="workspace/files")
    workspace_status = {
        "ready": Path(workspace_root).exists(),
    }

    logger.info("get_system_status called")

    return {
        "llm": llm_status,
        "ml_model": ml_status,
        "memory": memory_status,
        "workspace": workspace_status,
        "agent": {
            "max_steps": get(cfg, "agent", "max_steps", default=5),
        },
    }

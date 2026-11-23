"""
Local audit logging for Carter agent actions.
Records every executed tool step, input arguments, execution duration, and results to a local JSONL audit trail (FR-15).
Ensures air-gap compliance and traceability without leaking external telemetry.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from carter.utils.config import get
from carter.utils.logging import get_logger

logger = get_logger("agent.audit")


class AuditLogger:
    """
    Appends structured step execution records to a local JSONL audit trail file.
    """

    def __init__(self, log_path: Path | str = "artifacts/audit.jsonl", enabled: bool = True) -> None:
        self.log_path = Path(log_path)
        self.enabled = enabled
        if self.enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_step(
        self,
        task_id: str,
        step_num: int,
        tool_name: str,
        args: dict[str, Any],
        result: Any,
        error: str | None,
        duration_ms: float,
    ) -> None:
        """Record a single tool action step."""
        if not self.enabled:
            return

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "step_num": step_num,
            "tool_name": tool_name,
            "args": args,
            "error": error,
            "duration_ms": duration_ms,
            "result_summary": str(result)[:200] if result is not None else None,
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            logger.error("Failed to write to audit log %s: %s", self.log_path, e)

    def log_task_completion(
        self,
        task_id: str,
        user_request: str,
        success: bool,
        steps_used: int,
        duration_ms: float,
    ) -> None:
        """Record overall task completion."""
        if not self.enabled:
            return

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "event": "task_completed",
            "user_request": user_request[:100],
            "success": success,
            "steps_used": steps_used,
            "total_duration_ms": duration_ms,
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            logger.error("Failed to write task completion to audit log: %s", e)

    def read_records(self, limit: int = 50) -> list[dict[str, Any]]:
        """Read recent audit records from disk."""
        if not self.log_path.exists():
            return []

        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records[-limit:]


def create_audit_logger(cfg: dict[str, Any]) -> AuditLogger:
    """Instantiate audit logger from config settings."""
    enabled = get(cfg, "agent", "audit_log", default=True)
    log_path = get(cfg, "agent", "audit_log_path", default="artifacts/audit.jsonl")
    return AuditLogger(log_path=log_path, enabled=enabled)

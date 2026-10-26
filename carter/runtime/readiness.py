"""
Carter system readiness check.
Verifies all required local resources before the agent starts.
Never attempts to download anything — fails clearly if a resource is missing.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carter.utils.config import get
from carter.utils.logging import get_logger

logger = get_logger("readiness")


@dataclass
class ComponentStatus:
    name: str
    ready: bool
    message: str
    details: list[str] = field(default_factory=list)


def check_llm(cfg: dict[str, Any]) -> ComponentStatus:
    """Verify that the configured LLM model file exists and is non-empty."""
    model_path = get(cfg, "llm", "model_path", default="")

    if not model_path:
        return ComponentStatus(
            name="LLM",
            ready=False,
            message="No model_path configured in carter.yaml",
        )

    path = Path(model_path)

    if not path.exists():
        return ComponentStatus(
            name="LLM",
            ready=False,
            message=f"Model file not found: {path}",
            details=[
                "Place your GGUF model file at the configured path.",
                "Do NOT run Carter with network access to download it.",
                f"Expected: {path.resolve()}",
            ],
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb < 10:
        return ComponentStatus(
            name="LLM",
            ready=False,
            message=f"Model file looks too small ({size_mb:.1f} MB) — may be corrupt.",
            details=["Verify the model file against its expected checksum."],
        )

    return ComponentStatus(
        name="LLM",
        ready=True,
        message=f"Model file found ({size_mb:.0f} MB): {path}",
    )


def check_ml_model(cfg: dict[str, Any]) -> ComponentStatus:
    """Verify that the trained ML model artifact exists."""
    model_path = get(cfg, "ml", "model_path", default="")

    if not model_path:
        return ComponentStatus(
            name="ML Model",
            ready=False,
            message="No ml.model_path configured. Train the model first (Week 4).",
        )

    path = Path(model_path)

    if not path.exists():
        return ComponentStatus(
            name="ML Model",
            ready=False,
            message=f"ML model artifact not found: {path}",
            details=[
                "Run the training script to produce this artifact:",
                "  python -m carter.ml.training.train",
            ],
        )

    return ComponentStatus(
        name="ML Model",
        ready=True,
        message=f"ML model artifact found: {path}",
    )


def check_memory(cfg: dict[str, Any]) -> ComponentStatus:
    """Verify that the SQLite database directory is writable."""
    db_path_str = get(cfg, "memory", "db_path", default="artifacts/carter.db")
    db_path = Path(db_path_str)

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Check writability via a probe
        probe = db_path.parent / ".write_probe"
        probe.touch()
        probe.unlink()
        return ComponentStatus(
            name="Memory (SQLite)",
            ready=True,
            message=f"Database directory writable: {db_path.parent}",
        )
    except OSError as e:
        return ComponentStatus(
            name="Memory (SQLite)",
            ready=False,
            message=f"Cannot write to database directory: {e}",
        )


def check_workspace(cfg: dict[str, Any]) -> ComponentStatus:
    """Verify that the configured workspace directory exists."""
    root = get(cfg, "workspace", "root", default="workspace/files")
    path = Path(root)

    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
            return ComponentStatus(
                name="Workspace",
                ready=True,
                message=f"Workspace directory created: {path}",
            )
        except OSError as e:
            return ComponentStatus(
                name="Workspace",
                ready=False,
                message=f"Cannot create workspace directory: {e}",
            )

    return ComponentStatus(
        name="Workspace",
        ready=True,
        message=f"Workspace directory exists: {path}",
    )


def run_readiness_check(cfg: dict[str, Any]) -> bool:
    """
    Run all component readiness checks and print a summary.

    Returns True if all REQUIRED components (LLM, Memory, Workspace) are ready.
    ML model is reported but not required to start the agent.
    """
    start = time.monotonic()

    checks = [
        check_llm(cfg),
        check_memory(cfg),
        check_workspace(cfg),
        check_ml_model(cfg),
    ]

    elapsed = time.monotonic() - start

    print("\n" + "=" * 60)
    print("  CARTER READINESS CHECK")
    print("=" * 60)

    required_ready = True
    for status in checks:
        icon = "✓" if status.ready else "✗"
        label = status.name.ljust(20)
        print(f"  {icon} {label}  {status.message}")
        if status.details:
            for detail in status.details:
                print(f"       → {detail}")

        # LLM, Memory, and Workspace are required; ML model is optional at start
        if not status.ready and status.name != "ML Model":
            required_ready = False

    print("-" * 60)
    overall = "READY" if required_ready else "NOT READY"
    print(f"  Overall: {overall}  (checked in {elapsed:.2f}s)")
    print("=" * 60 + "\n")

    if not required_ready:
        logger.error("Readiness check failed. Resolve the issues above before starting Carter.")
    else:
        logger.info("Readiness check passed in %.2fs", elapsed)

    return required_ready


if __name__ == "__main__":
    from carter.utils.config import load_config
    from carter.utils.logging import setup_logging

    cfg = load_config()
    setup_logging(cfg)
    ok = run_readiness_check(cfg)
    sys.exit(0 if ok else 1)

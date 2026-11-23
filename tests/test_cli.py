"""
Tests for Carter Click CLI and local audit logging.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from click.testing import CliRunner
import yaml

from carter import cli
from carter.agent.audit import AuditLogger


@pytest.fixture
def cli_config_file(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    cfg = {
        "llm": {
            "backend": "mock",
        },
        "workspace": {
            "root": str(ws),
        },
        "memory": {
            "db_path": str(artifacts / "carter.db"),
        },
        "agent": {
            "audit_log": True,
            "audit_log_path": str(artifacts / "audit.jsonl"),
        },
    }

    config_path = tmp_path / "test_carter.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    return config_path


class TestAuditLogger:
    def test_audit_log_step_and_read(self, tmp_path: Path):
        log_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(log_file, enabled=True)

        audit.log_step(
            task_id="t123",
            step_num=1,
            tool_name="read_file",
            args={"filepath": "test.txt"},
            result={"content": "hello"},
            error=None,
            duration_ms=12.5,
        )

        records = audit.read_records()
        assert len(records) == 1
        assert records[0]["task_id"] == "t123"
        assert records[0]["tool_name"] == "read_file"
        assert records[0]["duration_ms"] == 12.5
        assert records[0]["error"] is None

    def test_audit_log_disabled(self, tmp_path: Path):
        log_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(log_file, enabled=False)

        audit.log_step("t1", 1, "test", {}, None, None, 1.0)
        assert not log_file.exists()


class TestCLI:
    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Carter: your fully offline AI agent." in result.output
        assert "check" in result.output
        assert "run" in result.output
        assert "ask" in result.output
        assert "train-ml" in result.output
        assert "audit" in result.output

    def test_cli_ask_command(self, cli_config_file: Path):
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cli_config_file), "ask", "Who are you?"])
        assert result.exit_code == 0
        assert "I have completed the task." in result.output

    def test_cli_audit_empty(self, cli_config_file: Path):
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cli_config_file), "audit"])
        assert result.exit_code == 0
        assert "No audit records found." in result.output

    def test_cli_audit_shows_records_after_ask(self, cli_config_file: Path):
        runner = CliRunner()
        # Trigger an ask command which logs task completion
        runner.invoke(cli, ["--config", str(cli_config_file), "ask", "Sample query"])
        result = runner.invoke(cli, ["--config", str(cli_config_file), "audit"])
        assert result.exit_code == 0
        assert "Recent Carter Audit Trail" in result.output
        assert "TASK COMPLETED" in result.output

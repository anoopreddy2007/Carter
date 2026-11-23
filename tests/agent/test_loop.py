"""
Tests for the bounded AgentLoop and action execution.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from carter.agent.loop import AgentLoop, AgentResult


@pytest.fixture
def agent_config(tmp_path: Path) -> dict:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "sample.txt").write_text("Carter local workspace text.", encoding="utf-8")

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    return {
        "llm": {
            "backend": "mock",
        },
        "workspace": {
            "root": str(ws),
            "max_file_size_bytes": 1024,
        },
        "memory": {
            "db_path": str(artifacts / "test.db"),
            "max_note_length": 500,
        },
        "ml": {
            "model_path": "",
            "model_type": "sklearn",
            "max_features": 10,
        },
        "agent": {
            "max_steps": 5,
            "tool_timeout_seconds": 10,
        },
    }


class TestAgentLoop:
    def test_loop_initialization(self, agent_config: dict):
        agent = AgentLoop(agent_config)
        tools = agent.registry.list_tools()
        assert "list_files" in tools
        assert "read_file" in tools
        assert "save_note" in tools
        assert "search_notes" in tools
        assert "predict_anomaly" in tools
        assert "get_system_status" in tools

    def test_parse_action_valid_json(self, agent_config: dict):
        agent = AgentLoop(agent_config)
        output = 'Thought: let me read files.\n{"tool": "list_files", "args": {"subpath": ""}}'
        action = agent._parse_action(output)
        assert action is not None
        assert action["tool"] == "list_files"
        assert action["args"] == {"subpath": ""}

    def test_parse_action_no_tool_returns_none(self, agent_config: dict):
        agent = AgentLoop(agent_config)
        output = "I have finished the task. Here is the final summary."
        action = agent._parse_action(output)
        assert action is None

    def test_parse_action_malformed_json_raises(self, agent_config: dict):
        agent = AgentLoop(agent_config)
        output = '{"tool": "read_file", "args": {broken}}'
        with pytest.raises(ValueError, match="Malformed JSON in action"):
            agent._parse_action(output)

    def test_parse_action_invalid_tool_type_raises(self, agent_config: dict):
        agent = AgentLoop(agent_config)
        output = '{"tool": 123, "args": {}}'
        with pytest.raises(ValueError, match="Action 'tool' field must be a string"):
            agent._parse_action(output)

    def test_parse_action_invalid_args_type_raises(self, agent_config: dict):
        agent = AgentLoop(agent_config)
        output = '{"tool": "read_file", "args": "invalid"}'
        with pytest.raises(ValueError, match="Action 'args' field must be a JSON object"):
            agent._parse_action(output)

    def test_execute_action_success(self, agent_config: dict):
        agent = AgentLoop(agent_config)
        action = {
            "tool": "read_file",
            "args": {"filepath": "sample.txt"},
        }
        result, error = agent.execute_action(action)
        assert error is None
        assert result is not None
        assert result["content"] == "Carter local workspace text."

    def test_execute_action_unknown_tool(self, agent_config: dict):
        agent = AgentLoop(agent_config)
        action = {
            "tool": "unauthorized_bash_exec",
            "args": {"cmd": "ls"},
        }
        result, error = agent.execute_action(action)
        assert result is None
        assert error is not None
        assert "Unknown tool" in error

    def test_execute_action_tool_runtime_failure(self, agent_config: dict):
        agent = AgentLoop(agent_config)
        action = {
            "tool": "read_file",
            "args": {"filepath": "non_existent.txt"},
        }
        result, error = agent.execute_action(action)
        assert result is None
        assert error is not None
        assert "File not found" in error

    def test_run_task_mock_execution(self, agent_config: dict):
        agent = AgentLoop(agent_config)
        res = agent.run_task("Check system files")
        assert isinstance(res, AgentResult)
        assert res.success is True
        assert res.steps_used == 0


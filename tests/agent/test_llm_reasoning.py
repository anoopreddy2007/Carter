"""
Tests for LLM prompts, LLM client interfaces, and multi-step agent reasoning loops.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from carter.agent.prompts import build_system_prompt, build_agent_turn_prompt
from carter.agent.llm import MockLLMClient, LlamaServerClient
from carter.agent.loop import AgentLoop, AgentResult


@pytest.fixture
def agent_config(tmp_path: Path) -> dict:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "notes.txt").write_text("Offline AI project notes.", encoding="utf-8")

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
            "max_steps": 3,
            "tool_timeout_seconds": 10,
        },
    }


class TestLLMPromptsAndClients:
    def test_llama_server_rejects_non_local_host(self):
        with pytest.raises(ValueError, match="Air-gap safety violation"):
            LlamaServerClient(host="api.openai.com")

    def test_mock_llm_client_returns_canned_responses(self):
        client = MockLLMClient(["response 1", "response 2"])
        assert client.generate([]) == "response 1"
        assert client.generate([]) == "response 2"
        assert client.generate([]) == "I have completed the task."


class TestAgentReasoningLoop:
    def test_direct_answer_without_tools(self, agent_config: dict):
        mock_client = MockLLMClient(["Hello! I am Carter, your local offline assistant."])
        agent = AgentLoop(agent_config, llm_client=mock_client)

        result = agent.run_task("Who are you?")
        assert result.success is True
        assert result.steps_used == 0
        assert "Carter, your local offline assistant" in result.answer

    def test_single_tool_execution_and_final_answer(self, agent_config: dict):
        tool_call = json.dumps({
            "thought": "I need to read notes.txt",
            "tool": "read_file",
            "args": {"filepath": "notes.txt"},
        })
        final_answer = "The notes say: Offline AI project notes."

        mock_client = MockLLMClient([tool_call, final_answer])
        agent = AgentLoop(agent_config, llm_client=mock_client)

        result = agent.run_task("What is inside notes.txt?")
        assert result.success is True
        assert result.steps_used == 1
        assert len(result.steps) == 1
        assert result.steps[0].tool_name == "read_file"
        assert result.steps[0].result["content"] == "Offline AI project notes."
        assert result.answer == final_answer

    def test_step_budget_enforcement(self, agent_config: dict):
        # Config has max_steps = 3. Feed infinite tool calls.
        infinite_tool = json.dumps({
            "tool": "list_files",
            "args": {},
        })
        mock_client = MockLLMClient([infinite_tool, infinite_tool, infinite_tool, infinite_tool])
        agent = AgentLoop(agent_config, llm_client=mock_client)

        result = agent.run_task("Keep listing files forever")
        assert result.success is False
        assert result.steps_used == 3
        assert result.stopped_early is True
        assert "Reached maximum execution limit of 3 steps" in result.answer

    def test_llm_connection_error_handling(self, agent_config: dict):
        class FailingClient(MockLLMClient):
            def generate(self, messages, **kwargs):
                raise ConnectionError("Server connection refused")

        agent = AgentLoop(agent_config, llm_client=FailingClient())
        result = agent.run_task("Hello")
        assert result.success is False
        assert "Error communicating with local LLM" in result.answer

"""
Carter agent loop.
Converts user requests into bounded, validated tool calls.
Maximum 5 steps per task (configurable). No arbitrary shell execution.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from carter.tools.registry import ToolRegistry, ToolSchema
from carter.tools.list_files import list_files
from carter.tools.read_file import read_file
from carter.tools.memory import save_note, search_notes
from carter.tools.system_status import get_system_status
from carter.tools.predict_anomaly import predict_anomaly
from carter.utils.config import get
from carter.utils.logging import get_logger

logger = get_logger("agent.loop")


@dataclass
class AgentStep:
    """Record of a single agent step for the audit log."""
    step_num: int
    tool_name: str
    args: dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class AgentResult:
    """Final result of an agent task."""
    success: bool
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    steps_used: int = 0
    stopped_early: bool = False


class AgentLoop:
    """
    The bounded tool-using agent.
    
    Responsibilities:
    - Parse and validate the LLM's structured action requests
    - Call only registered tools with validated arguments
    - Enforce step budget and time limits
    - Record all actions in the local audit log
    - Return clear answers (including failure reasons)
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.max_steps: int = get(cfg, "agent", "max_steps", default=5)
        self.tool_timeout: int = get(cfg, "agent", "tool_timeout_seconds", default=30)
        self.registry = self._build_registry()
        logger.info(
            "AgentLoop initialized. max_steps=%d, tools=%s",
            self.max_steps,
            self.registry.list_tools(),
        )

    def _build_registry(self) -> ToolRegistry:
        """Register all allowlisted MVP tools."""
        reg = ToolRegistry()
        ws_root = get(self.cfg, "workspace", "root", default="workspace/files")
        max_file_size = get(self.cfg, "workspace", "max_file_size_bytes", default=524288)
        db_path = get(self.cfg, "memory", "db_path", default="artifacts/carter.db")
        max_note_len = get(self.cfg, "memory", "max_note_length", default=4096)
        ml_path = get(self.cfg, "ml", "model_path", default="")
        ml_type = get(self.cfg, "ml", "model_type", default="sklearn")
        max_features = get(self.cfg, "ml", "max_features", default=50)

        # list_files
        reg.register(ToolSchema(
            name="list_files",
            description="List files and directories within the permitted workspace.",
            args={
                "subpath": {
                    "type": "string",
                    "required": False,
                    "description": "Relative subdirectory path within the workspace.",
                }
            },
            handler=lambda subpath="": list_files(ws_root, subpath),
        ))

        # read_file
        reg.register(ToolSchema(
            name="read_file",
            description="Read the content of a UTF-8 text file from the workspace.",
            args={
                "filepath": {
                    "type": "string",
                    "required": True,
                    "description": "Relative path to the file within the workspace.",
                }
            },
            handler=lambda filepath: read_file(filepath, ws_root, max_file_size),
        ))

        # save_note
        reg.register(ToolSchema(
            name="save_note",
            description="Save a note to local persistent storage (SQLite).",
            args={
                "content": {
                    "type": "string",
                    "required": True,
                    "description": "The note body text.",
                },
                "title": {
                    "type": "string",
                    "required": False,
                    "description": "Optional note title.",
                },
                "tags": {
                    "type": "string",
                    "required": False,
                    "description": "Optional comma-separated tags.",
                },
            },
            handler=lambda content, title="", tags="": save_note(
                content, db_path, title, tags, max_note_len
            ),
        ))

        # search_notes
        reg.register(ToolSchema(
            name="search_notes",
            description="Search previously saved notes by keyword.",
            args={
                "query": {
                    "type": "string",
                    "required": True,
                    "description": "The search keyword or phrase.",
                },
                "limit": {
                    "type": "integer",
                    "required": False,
                    "description": "Maximum number of results to return (default 10).",
                },
            },
            handler=lambda query, limit=10: search_notes(query, db_path, limit),
        ))

        # predict_anomaly
        reg.register(ToolSchema(
            name="predict_anomaly",
            description="Run the local ML model on a feature vector and get a prediction.",
            args={
                "features": {
                    "type": "array",
                    "required": True,
                    "description": "List of numeric feature values.",
                }
            },
            handler=lambda features: predict_anomaly(
                features, ml_path, ml_type, max_features
            ),
        ))

        # get_system_status
        reg.register(ToolSchema(
            name="get_system_status",
            description="Report the readiness status of Carter's local components.",
            args={},
            handler=lambda: get_system_status(self.cfg),
        ))

        return reg

    def _parse_action(self, llm_output: str) -> Optional[dict[str, Any]]:
        """
        Parse a structured action from LLM output.
        Expected format: JSON object with "tool" and "args" keys.
        Returns None if no valid action is found (treat as final answer).
        """
        llm_output = llm_output.strip()

        # Look for JSON block
        start = llm_output.find("{")
        end = llm_output.rfind("}") + 1
        if start == -1 or end == 0:
            return None

        json_str = llm_output[start:end]
        try:
            action = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse action JSON: %s | Input: %s", e, json_str[:200])
            raise ValueError(f"Malformed JSON in action: {e}")

        if not isinstance(action, dict):
            raise ValueError("Action must be a JSON object.")

        if "tool" not in action:
            return None  # No tool call — treat as final answer

        if not isinstance(action.get("tool"), str):
            raise ValueError("Action 'tool' field must be a string.")

        if "args" not in action:
            action["args"] = {}

        if not isinstance(action.get("args"), dict):
            raise ValueError("Action 'args' field must be a JSON object.")

        return action

    def execute_action(self, action: dict[str, Any]) -> tuple[Any, Optional[str]]:
        """
        Execute a single parsed action.
        Returns (result, error_message). error_message is None on success.
        """
        tool_name = action.get("tool", "")
        args = action.get("args", {})

        if not self.registry.is_registered(tool_name):
            return None, f"Unknown tool: '{tool_name}'. Allowed: {self.registry.list_tools()}"

        try:
            result = self.registry.call(tool_name, args)
            return result, None
        except (ValueError, FileNotFoundError, PermissionError) as e:
            return None, str(e)
        except RuntimeError as e:
            return None, str(e)

    def start(self) -> None:
        """Start the interactive REPL loop."""
        print("\nCarter is ready. Type your request (or 'quit' to exit).\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n[Carter] Session ended.")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                print("[Carter] Goodbye.")
                break

            if not user_input:
                continue

            response = self.run_task(user_input)
            print(f"\nCarter: {response.answer}\n")

    def run_task(self, user_request: str) -> AgentResult:
        """
        Run a single task through the agent loop.
        This is the main entry point for programmatic use and evaluation.
        
        NOTE: In MVP, this is a simplified loop that demonstrates the structure.
        The actual LLM integration will be completed in Week 2-3.
        """
        logger.info("run_task: '%s'", user_request[:100])
        steps: list[AgentStep] = []

        # TODO (Week 2-3): Replace this stub with actual LLM planning.
        # For now, return a placeholder response explaining the system is ready.
        answer = (
            f"[MVP Stub] Carter received your request: '{user_request}'\n"
            f"LLM integration is pending (Week 2). "
            f"Agent loop structure is in place. "
            f"Registered tools: {self.registry.list_tools()}"
        )

        return AgentResult(
            success=True,
            answer=answer,
            steps=steps,
            steps_used=0,
        )

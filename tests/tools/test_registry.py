"""
Tests for the tool registry — allowlist enforcement and argument validation.
"""

import pytest
from carter.tools.registry import ToolRegistry, ToolSchema


def make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSchema(
        name="echo",
        description="Echo a message.",
        args={
            "message": {"type": "string", "required": True, "description": "The message."}
        },
        handler=lambda message: {"echoed": message},
    ))
    return reg


class TestToolRegistry:
    def test_registered_tool_is_callable(self) -> None:
        reg = make_registry()
        result = reg.call("echo", {"message": "hello"})
        assert result == {"echoed": "hello"}

    def test_unknown_tool_raises_value_error(self) -> None:
        reg = make_registry()
        with pytest.raises(ValueError, match="Unknown tool"):
            reg.call("run_shell", {"cmd": "rm -rf /"})

    def test_missing_required_arg_raises_value_error(self) -> None:
        reg = make_registry()
        with pytest.raises(ValueError, match="missing required argument"):
            reg.call("echo", {})

    def test_unexpected_arg_raises_value_error(self) -> None:
        reg = make_registry()
        with pytest.raises(ValueError, match="unexpected argument"):
            reg.call("echo", {"message": "hi", "injected": "bad"})

    def test_is_registered(self) -> None:
        reg = make_registry()
        assert reg.is_registered("echo") is True
        assert reg.is_registered("unknown") is False

    def test_list_tools(self) -> None:
        reg = make_registry()
        assert "echo" in reg.list_tools()

    def test_duplicate_registration_raises(self) -> None:
        reg = make_registry()
        with pytest.raises(ValueError, match="already registered"):
            reg.register(ToolSchema(
                name="echo",
                description="Duplicate.",
                args={},
                handler=lambda: None,
            ))

    def test_register_without_handler_raises(self) -> None:
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="no handler"):
            reg.register(ToolSchema(
                name="no_handler",
                description="Missing handler.",
                args={},
                handler=None,
            ))

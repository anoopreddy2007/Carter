"""
Carter tool registry.
Only explicitly registered tools may be called by the agent.
Unknown tool names are rejected before any execution occurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from carter.utils.logging import get_logger

logger = get_logger("tools.registry")


@dataclass
class ToolSchema:
    """Describes a registered tool: its name, description, and argument schema."""
    name: str
    description: str
    # JSON Schema-style argument definitions: {arg_name: {type, required, description}}
    args: dict[str, dict[str, Any]] = field(default_factory=dict)
    handler: Optional[Callable[..., Any]] = field(default=None, repr=False)


class ToolRegistry:
    """
    Allowlist-based tool registry.
    Only tools explicitly registered here may be called by the agent.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSchema] = {}

    def register(self, schema: ToolSchema) -> None:
        """Register a tool. Raises ValueError if the name is already registered."""
        if schema.name in self._tools:
            raise ValueError(f"Tool already registered: {schema.name}")
        if schema.handler is None:
            raise ValueError(f"Tool '{schema.name}' has no handler.")
        self._tools[schema.name] = schema
        logger.debug("Registered tool: %s", schema.name)

    def get(self, name: str) -> Optional[ToolSchema]:
        """Return the tool schema for the given name, or None if not registered."""
        return self._tools.get(name)

    def is_registered(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def call(self, name: str, args: dict[str, Any]) -> Any:
        """
        Execute a registered tool with the given arguments.

        Raises:
            ValueError: if the tool is not registered or args are invalid.
            RuntimeError: if the tool handler raises an unexpected error.
        """
        schema = self.get(name)
        if schema is None:
            logger.warning("Attempted to call unknown tool: %s", name)
            raise ValueError(
                f"Unknown tool: '{name}'. "
                f"Registered tools: {self.list_tools()}"
            )

        # Validate required arguments
        for arg_name, arg_spec in schema.args.items():
            if arg_spec.get("required", False) and arg_name not in args:
                raise ValueError(
                    f"Tool '{name}' missing required argument: '{arg_name}'"
                )

        # Reject unexpected arguments
        for arg_name in args:
            if arg_name not in schema.args:
                raise ValueError(
                    f"Tool '{name}' received unexpected argument: '{arg_name}'"
                )

        logger.info("Calling tool: %s with args: %s", name, list(args.keys()))
        try:
            return schema.handler(**args)
        except (ValueError, PermissionError, FileNotFoundError) as e:
            logger.warning("Tool '%s' raised expected error: %s", name, e)
            raise
        except Exception as e:
            logger.error("Tool '%s' raised unexpected error: %s", name, e)
            raise RuntimeError(f"Tool '{name}' failed unexpectedly: {e}") from e

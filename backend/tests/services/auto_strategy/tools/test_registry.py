from typing import Any, Dict
from unittest.mock import patch

import pytest

from app.services.auto_strategy.tools.base import (
    BaseTool,
    ToolContext,
    ToolDefinition,
)
from app.services.auto_strategy.tools.registry import ToolRegistry


def build_tool(name: str) -> BaseTool:
    class _Tool(BaseTool):
        tool_definition = ToolDefinition(
            name=name,
            description=f"{name} description",
            default_params={"enabled": True, "tags": []},
        )

        def should_skip_entry(
            self,
            context: ToolContext,
            params: Dict[str, Any],
        ) -> bool:
            return False

    return _Tool()


class TestToolRegistry:
    @pytest.fixture(autouse=True)
    def clear_registry(self):
        registry = ToolRegistry()
        registry.clear()
        yield
        registry.clear()

    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = build_tool("registry_test_tool")

        registry.register(tool)

        assert registry.get("registry_test_tool") == tool
        assert tool in registry.get_all()
        assert any(
            definition.name == "registry_test_tool"
            for definition in registry.get_definitions()
        )

    def test_get_definitions_returns_independent_copies(self):
        registry = ToolRegistry()
        tool = build_tool("registry_copy_tool")

        registry.register(tool)

        copied_definition = next(
            definition
            for definition in registry.get_definitions()
            if definition.name == "registry_copy_tool"
        )
        copied_definition.default_params["tags"].append("x")

        assert tool.definition.default_params["tags"] == []
        assert tool.get_default_params()["tags"] == []

    def test_get_not_found(self):
        registry = ToolRegistry()
        assert registry.get("non_existent") is None

    def test_register_duplicate_warning(self):
        registry = ToolRegistry()
        tool1 = build_tool("dup_registry_tool")
        tool2 = build_tool("dup_registry_tool")

        with patch("app.services.auto_strategy.tools.registry.logger.warning") as warning_mock:
            registry.register(tool1)
            registry.register(tool2)

        assert warning_mock.called
        assert registry.get("dup_registry_tool") == tool2

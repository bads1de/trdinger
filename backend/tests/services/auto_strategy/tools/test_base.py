from typing import Any, Dict

import pytest

from app.services.auto_strategy.tools.base import (
    BaseTool,
    ToolContext,
    ToolDefinition,
)


class MockTool(BaseTool):
    """テスト用の具象ツールクラス"""

    tool_definition = ToolDefinition(
        name="mock_tool",
        description="Mock tool for testing",
        default_params={"should_skip": False, "tags": []},
    )

    def should_skip_entry(self, context: ToolContext, params: Dict[str, Any]) -> bool:
        return params.get("should_skip", False)


class TestBaseTool:
    def test_tool_context_init(self):
        context = ToolContext(current_price=50000.0)
        assert context.current_price == 50000.0
        assert context.extra_data == {}

    def test_base_tool_abstract_methods(self):
        # 抽象クラスなのでインスタンス化できないことを確認
        with pytest.raises(TypeError):
            BaseTool()

    def test_mock_tool_implementation(self):
        tool = MockTool()
        assert tool.name == "mock_tool"
        assert tool.description == "Mock tool for testing"
        assert tool.definition == MockTool.tool_definition

        context = ToolContext()
        assert tool.should_skip_entry(context, {"should_skip": True}) is True
        assert tool.should_skip_entry(context, {"should_skip": False}) is False
        assert tool.get_default_params() == {
            "should_skip": False,
            "tags": [],
        }

    def test_mock_tool_default_params_are_copied(self):
        tool = MockTool()
        params = tool.get_default_params()
        params["tags"].append("x")

        assert tool.get_default_params()["tags"] == []

    def test_base_tool_default_methods(self):
        import random

        random.seed(42)  # Ensure deterministic behavior for mutate_params
        tool = MockTool()
        # デフォルト実装の確認
        assert tool.validate_params({}) is True

        params = {"key": "value"}
        mutated = tool.mutate_params(params)
        assert mutated == params
        assert mutated is not params  # コピーされていること

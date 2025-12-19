import pytest
from typing import Any, Dict
from app.services.auto_strategy.tools.base import BaseTool, ToolContext

class MockTool(BaseTool):
    """テスト用の具象ツールクラス"""
    @property
    def name(self) -> str:
        return "mock_tool"
    
    @property
    def description(self) -> str:
        return "Mock tool for testing"

    def should_skip_entry(self, context: ToolContext, params: Dict[str, Any]) -> bool:
        return params.get("should_skip", False)

    def get_default_params(self) -> Dict[str, Any]:
        return {"should_skip": False}

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
        
        context = ToolContext()
        assert tool.should_skip_entry(context, {"should_skip": True}) is True
        assert tool.should_skip_entry(context, {"should_skip": False}) is False
        assert tool.get_default_params() == {"should_skip": False}

    def test_base_tool_default_methods(self):
        tool = MockTool()
        # デフォルト実装の確認
        assert tool.validate_params({}) is True
        
        params = {"key": "value"}
        mutated = tool.mutate_params(params)
        assert mutated == params
        assert mutated is not params # コピーされていること

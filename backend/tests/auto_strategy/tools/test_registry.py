import pytest
from unittest.mock import MagicMock
from app.services.auto_strategy.tools.registry import ToolRegistry
from app.services.auto_strategy.tools.base import BaseTool

class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = MagicMock(spec=BaseTool)
        tool.name = "my_tool"
        
        registry.register(tool)
        assert registry.get("my_tool") == tool
        assert tool in registry.get_all()

    def test_get_not_found(self):
        registry = ToolRegistry()
        assert registry.get("non_existent") is None

    def test_register_duplicate_warning(self, caplog):
        # ログレベルを設定
        import logging
        registry = ToolRegistry()
        t1 = MagicMock(spec=BaseTool)
        t1.name = "dup"
        t2 = MagicMock(spec=BaseTool)
        t2.name = "dup"
        
        registry.register(t1)
        # 2回目は警告が出るはず
        with caplog.at_level(logging.WARNING):
            registry.register(t2)
            assert "既に登録されています" in caplog.text or len(caplog.records) > 0
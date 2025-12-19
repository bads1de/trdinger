"""
ツールのユニットテスト
"""

import pytest
from datetime import datetime
import pandas as pd

from app.services.auto_strategy.tools.weekend_filter import WeekendFilter
from app.services.auto_strategy.tools.base import ToolContext
from app.services.auto_strategy.tools.registry import ToolRegistry


class TestWeekendFilter:
    """WeekendFilterのテストクラス"""

    @pytest.fixture
    def filter_tool(self):
        return WeekendFilter()

    def test_should_skip_entry_on_weekends(self, filter_tool):
        """週末にスキップされるかテスト"""
        # 土曜日: 2024-01-06
        sat_ctx = ToolContext(timestamp=pd.Timestamp("2024-01-06"))
        assert filter_tool.should_skip_entry(sat_ctx, {"enabled": True}) is True
        
        # 日曜日: 2024-01-07
        sun_ctx = ToolContext(timestamp=pd.Timestamp("2024-01-07"))
        assert filter_tool.should_skip_entry(sun_ctx, {"enabled": True}) is True

    def test_should_not_skip_on_weekdays(self, filter_tool):
        """平日にスキップされないかテスト"""
        # 月曜日: 2024-01-08
        mon_ctx = ToolContext(timestamp=pd.Timestamp("2024-01-08"))
        assert filter_tool.should_skip_entry(mon_ctx, {"enabled": True}) is False

    def test_respects_enabled_param(self, filter_tool):
        """enabledパラメータを尊重するかテスト"""
        sat_ctx = ToolContext(timestamp=pd.Timestamp("2024-01-06"))
        # 無効化されている場合は週末でもFalse（スキップしない）
        assert filter_tool.should_skip_entry(sat_ctx, {"enabled": False}) is False


class TestToolRegistry:
    """ツールのレジストリのテスト"""

    def test_registry_basic_operations(self):
        """レジストリの基本操作を独立したインスタンスでテスト"""
        registry = ToolRegistry()
        filter_tool = WeekendFilter()
        registry.register(filter_tool)
        
        assert registry.get("weekend_filter") is not None
        assert any(t.name == "weekend_filter" for t in registry.get_all())

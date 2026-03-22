import pytest
from unittest.mock import MagicMock
from datetime import datetime
import pandas as pd
from app.services.auto_strategy.tools.weekend_filter import WeekendFilter
from app.services.auto_strategy.tools.base import ToolContext

class TestWeekendFilter:
    @pytest.fixture
    def filter_tool(self):
        return WeekendFilter()

    @pytest.fixture
    def context(self):
        # ToolContextのモック
        ctx = MagicMock(spec=ToolContext)
        return ctx

    def test_should_skip_entry_weekend(self, filter_tool, context):
        # 土曜日 (2023-01-07)
        context.timestamp = pd.Timestamp("2023-01-07 12:00:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True

        # 日曜日 (2023-01-08)
        context.timestamp = pd.Timestamp("2023-01-08 12:00:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True

    def test_should_skip_entry_weekday(self, filter_tool, context):
        # 月曜日 (2023-01-09)
        context.timestamp = pd.Timestamp("2023-01-09 12:00:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False

        # 金曜日 (2023-01-06)
        context.timestamp = pd.Timestamp("2023-01-06 12:00:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False

    def test_should_skip_entry_disabled(self, filter_tool, context):
        # 土曜日だが無効化されている場合
        context.timestamp = pd.Timestamp("2023-01-07 12:00:00")
        assert filter_tool.should_skip_entry(context, {"enabled": False}) is False

    def test_should_skip_entry_none_timestamp(self, filter_tool, context):
        context.timestamp = None
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False

    def test_should_skip_entry_invalid_timestamp(self, filter_tool, context):
        # weekdayメソッドを持たないオブジェクト
        context.timestamp = "2023-01-07"
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False

    def test_get_default_params(self, filter_tool):
        params = filter_tool.get_default_params()
        assert params["enabled"] is True

    def test_mutate_params(self, filter_tool):
        # 乱数を固定して変異を確認
        import random
        random.seed(42)
        
        # 乱数の出方によっては変わらないこともあるので、何度か試行して変化を確認する
        # あるいはrandom.randomをモックする
        
        from unittest.mock import patch
        with patch('random.random') as mock_random:
            # 0.2未満 -> 変異する
            mock_random.return_value = 0.1
            params = {"enabled": True}
            new_params = filter_tool.mutate_params(params)
            assert new_params["enabled"] is False
            
            # 0.2以上 -> 変異しない
            mock_random.return_value = 0.5
            params = {"enabled": True}
            new_params = filter_tool.mutate_params(params)
            assert new_params["enabled"] is True

    def test_metadata(self, filter_tool):
        assert filter_tool.name == "weekend_filter"
        assert "土曜日" in filter_tool.description

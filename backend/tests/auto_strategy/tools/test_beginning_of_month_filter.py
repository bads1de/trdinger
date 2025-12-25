import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from app.services.auto_strategy.tools.beginning_of_month_filter import BeginningOfMonthFilter
from app.services.auto_strategy.tools.base import ToolContext

class TestBeginningOfMonthFilter:
    @pytest.fixture
    def filter_tool(self):
        return BeginningOfMonthFilter()

    @pytest.fixture
    def context(self):
        ctx = MagicMock(spec=ToolContext)
        return ctx

    def test_should_skip_entry_first_days(self, filter_tool, context):
        # デフォルト: days_from_start=2 (1日と2日)
        
        # 1日 -> スキップ
        context.timestamp = pd.Timestamp("2023-01-01")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True
        
        # 2日 -> スキップ
        context.timestamp = pd.Timestamp("2023-01-02")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True
        
        # 3日 -> スキップしない
        context.timestamp = pd.Timestamp("2023-01-03")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False

    def test_should_skip_entry_custom_days(self, filter_tool, context):
        # 3日までスキップ
        params = {"enabled": True, "days_from_start": 3}
        
        context.timestamp = pd.Timestamp("2023-01-03")
        assert filter_tool.should_skip_entry(context, params) is True
        
        context.timestamp = pd.Timestamp("2023-01-04")
        assert filter_tool.should_skip_entry(context, params) is False

    def test_mutate_params(self, filter_tool):
        with patch('random.random', side_effect=[0.5, 0.1, 0.5, 0.1]), \
             patch('random.randint', return_value=1):
            
            # 1回目: enabled変化なし, days変化あり (+1)
            params = {"enabled": True, "days_from_start": 2}
            new_params = filter_tool.mutate_params(params)
            assert new_params["enabled"] is True
            assert new_params["days_from_start"] == 3
            
            # 2回目: 境界値テスト (上限5)
            params = {"enabled": True, "days_from_start": 5}
            new_params = filter_tool.mutate_params(params)
            assert new_params["days_from_start"] == 5  # max(..., min(5, 6)) -> 5


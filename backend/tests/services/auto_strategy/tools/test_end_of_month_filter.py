import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from app.services.auto_strategy.tools.end_of_month_filter import EndOfMonthFilter
from app.services.auto_strategy.tools.base import ToolContext

class TestEndOfMonthFilter:
    @pytest.fixture
    def filter_tool(self):
        return EndOfMonthFilter()

    @pytest.fixture
    def context(self):
        ctx = MagicMock(spec=ToolContext)
        return ctx

    def test_should_skip_entry_last_day(self, filter_tool, context):
        # 1月31日 (最終日)
        context.timestamp = pd.Timestamp("2023-01-31 12:00:00")
        
        # デフォルト (days_before_end=0) -> 最終日のみスキップ
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True
        
        # 1月30日 -> スキップしない
        context.timestamp = pd.Timestamp("2023-01-30 12:00:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False

    def test_should_skip_entry_range(self, filter_tool, context):
        # days_before_end=2 (最終日、前日、前々日)
        params = {"enabled": True, "days_before_end": 2}
        
        # 1月31日 (残り0日) -> True
        context.timestamp = pd.Timestamp("2023-01-31")
        assert filter_tool.should_skip_entry(context, params) is True
        
        # 1月30日 (残り1日) -> True
        context.timestamp = pd.Timestamp("2023-01-30")
        assert filter_tool.should_skip_entry(context, params) is True
        
        # 1月29日 (残り2日) -> True
        context.timestamp = pd.Timestamp("2023-01-29")
        assert filter_tool.should_skip_entry(context, params) is True
        
        # 1月28日 (残り3日) -> False
        context.timestamp = pd.Timestamp("2023-01-28")
        assert filter_tool.should_skip_entry(context, params) is False

    def test_should_skip_entry_leap_year(self, filter_tool, context):
        # 閏年 (2024年2月)
        # 2月29日が最終日
        
        context.timestamp = pd.Timestamp("2024-02-29")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True
        
        context.timestamp = pd.Timestamp("2024-02-28")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False

    def test_mutate_params(self, filter_tool):
        with patch('random.random', side_effect=[0.5, 0.1, 0.5, 0.1]), \
             patch('random.randint', return_value=1):
            
            # 1回目: enabled変化なし, days変化あり
            params = {"enabled": True, "days_before_end": 0}
            new_params = filter_tool.mutate_params(params)
            assert new_params["enabled"] is True
            assert new_params["days_before_end"] == 1
            
            # 2回目: 境界値テスト (上限3)
            params = {"enabled": True, "days_before_end": 3}
            new_params = filter_tool.mutate_params(params)
            assert new_params["days_before_end"] == 3  # 上限で止まるはずだが +1 されて max(..., min(3, 4)) -> 3


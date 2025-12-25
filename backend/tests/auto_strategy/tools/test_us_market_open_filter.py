import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from app.services.auto_strategy.tools.us_market_open_filter import USMarketOpenFilter
from app.services.auto_strategy.tools.base import ToolContext

class TestUSMarketOpenFilter:
    @pytest.fixture
    def filter_tool(self):
        return USMarketOpenFilter()

    @pytest.fixture
    def context(self):
        ctx = MagicMock(spec=ToolContext)
        return ctx

    def test_should_skip_entry_winter_open(self, filter_tool, context):
        # 冬時間 (UTC-5)
        # Open: 09:30 EST -> 14:30 UTC
        # Window: 30分 (デフォルト) -> 14:00 - 15:00
        
        # 範囲内
        context.timestamp = pd.Timestamp("2023-01-04 14:30:00", tz='UTC')
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True
        
        context.timestamp = pd.Timestamp("2023-01-04 14:00:00", tz='UTC')
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True

        # 範囲外
        context.timestamp = pd.Timestamp("2023-01-04 13:59:00", tz='UTC')
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False

    def test_should_skip_entry_summer_open(self, filter_tool, context):
        # 夏時間 (UTC-4)
        # Open: 09:30 EDT -> 13:30 UTC
        # Window: 30分 -> 13:00 - 14:00
        
        # 範囲内
        context.timestamp = pd.Timestamp("2023-07-04 13:30:00", tz='UTC')
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True
        
        context.timestamp = pd.Timestamp("2023-07-04 14:00:00", tz='UTC')
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True

        # 範囲外
        context.timestamp = pd.Timestamp("2023-07-04 14:01:00", tz='UTC')
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False

    def test_should_skip_entry_naive_timestamp(self, filter_tool, context):
        # NaiveなTimestamp (1月なので冬時間扱い -> 14:30 Open)
        context.timestamp = pd.Timestamp("2023-01-04 14:30:00")
        # 内部でUTCとして扱われ、US/Easternに変換されると期待
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True

    def test_mutate_params(self, filter_tool):
        with patch('random.random', side_effect=[0.5, 0.1, 0.5, 0.1]), \
             patch('random.randint', return_value=10):
            
            # 1回目: enabled変化なし, window変化あり
            params = {"enabled": True, "window_minutes": 30}
            new_params = filter_tool.mutate_params(params)
            assert new_params["enabled"] is True
            assert new_params["window_minutes"] == 40
            
            # 2回目: 境界値テスト (上限60)
            params = {"enabled": True, "window_minutes": 55}
            new_params = filter_tool.mutate_params(params)
            assert new_params["window_minutes"] == 60

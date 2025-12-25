import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from app.services.auto_strategy.tools.london_fix_filter import LondonFixFilter
from app.services.auto_strategy.tools.base import ToolContext

class TestLondonFixFilter:
    @pytest.fixture
    def filter_tool(self):
        return LondonFixFilter()

    @pytest.fixture
    def context(self):
        ctx = MagicMock(spec=ToolContext)
        return ctx

    def test_should_skip_entry_winter_fix(self, filter_tool, context):
        # 冬時間 Fix (16:00 UTC)
        # デフォルトwindowは15分 (15:45 - 16:15)
        
        # 範囲内
        context.timestamp = pd.Timestamp("2023-01-04 16:00:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True
        
        context.timestamp = pd.Timestamp("2023-01-04 15:45:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True
        
        context.timestamp = pd.Timestamp("2023-01-04 16:15:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True

        # 範囲外
        context.timestamp = pd.Timestamp("2023-01-04 15:44:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False
        
        context.timestamp = pd.Timestamp("2023-01-04 16:16:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False

    def test_should_skip_entry_summer_fix(self, filter_tool, context):
        # 夏時間 Fix (15:00 UTC)
        # デフォルトwindowは15分 (14:45 - 15:15)
        
        # 範囲内
        context.timestamp = pd.Timestamp("2023-07-04 15:00:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True
        
        context.timestamp = pd.Timestamp("2023-07-04 14:45:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True
        
        context.timestamp = pd.Timestamp("2023-07-04 15:15:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True

        # 範囲外
        context.timestamp = pd.Timestamp("2023-07-04 14:44:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False

    def test_should_skip_entry_custom_window(self, filter_tool, context):
        # ウィンドウを30分に拡大
        params = {"enabled": True, "window_minutes": 30}
        
        # 冬時間 16:00 -> 15:30 - 16:30
        context.timestamp = pd.Timestamp("2023-01-04 15:30:00")
        assert filter_tool.should_skip_entry(context, params) is True
        
        # 15:29 は夏時間Fix (15:00) の範囲 (14:30-15:30) に入るため True になるのが正しい
        context.timestamp = pd.Timestamp("2023-01-04 15:29:00")
        assert filter_tool.should_skip_entry(context, params) is True

        # 安全な範囲外の時間 (12:00 UTC)
        context.timestamp = pd.Timestamp("2023-01-04 12:00:00")
        assert filter_tool.should_skip_entry(context, params) is False

    def test_should_skip_entry_disabled(self, filter_tool, context):
        context.timestamp = pd.Timestamp("2023-01-04 16:00:00")
        assert filter_tool.should_skip_entry(context, {"enabled": False}) is False

    def test_mutate_params(self, filter_tool):
        # グローバルの random.random と random.randint をパッチ
        # london_fix_filter.py は import random しているので、これで効くはず
        with patch('random.random', side_effect=[0.5, 0.1, 0.5, 0.1]), \
             patch('random.randint', return_value=5):
            
            # 1回目: enabled変化なし, window変化あり
            params = {"enabled": True, "window_minutes": 15}
            new_params = filter_tool.mutate_params(params)
            assert new_params["enabled"] is True
            assert new_params["window_minutes"] == 20
            
            # 2回目: 境界値テスト (上限30分)
            # random.randomのside_effectの続きが使われる
            with patch('random.randint', return_value=10):
                 params = {"enabled": True, "window_minutes": 25}
                 new_params = filter_tool.mutate_params(params)
                 assert new_params["window_minutes"] == 30  # max 30

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from app.services.auto_strategy.tools.monday_reversal_filter import MondayReversalFilter
from app.services.auto_strategy.tools.base import ToolContext

class TestMondayReversalFilter:
    @pytest.fixture
    def filter_tool(self):
        return MondayReversalFilter()

    @pytest.fixture
    def context(self):
        ctx = MagicMock(spec=ToolContext)
        return ctx

    def test_should_skip_entry_monday_morning(self, filter_tool, context):
        # 月曜日 (2023-01-02) の午前中 (00:00 - 11:59)
        # デフォルト skip_hours = 12
        
        # 00:00 -> スキップ
        context.timestamp = pd.Timestamp("2023-01-02 00:00:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True
        
        # 11:00 -> スキップ
        context.timestamp = pd.Timestamp("2023-01-02 11:00:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True

    def test_should_skip_entry_monday_afternoon(self, filter_tool, context):
        # 月曜日 (2023-01-02) の午後 (12:00以降)
        
        # 12:00 -> スキップしない
        context.timestamp = pd.Timestamp("2023-01-02 12:00:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False
        
        # 23:00 -> スキップしない
        context.timestamp = pd.Timestamp("2023-01-02 23:00:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False

    def test_should_skip_entry_other_days(self, filter_tool, context):
        # 火曜日 (2023-01-03)
        context.timestamp = pd.Timestamp("2023-01-03 05:00:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False
        
        # 日曜日 (2023-01-01)
        context.timestamp = pd.Timestamp("2023-01-01 05:00:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False

    def test_should_skip_entry_custom_hours(self, filter_tool, context):
        # スキップ時間を6時間に変更
        params = {"enabled": True, "skip_hours": 6}
        
        # 05:00 -> スキップ
        context.timestamp = pd.Timestamp("2023-01-02 05:00:00")
        assert filter_tool.should_skip_entry(context, params) is True
        
        # 06:00 -> スキップしない
        context.timestamp = pd.Timestamp("2023-01-02 06:00:00")
        assert filter_tool.should_skip_entry(context, params) is False

    def test_mutate_params(self, filter_tool):
        # グローバルの random.random と random.randint をパッチ
        with patch('random.random', side_effect=[0.5, 0.1, 0.5, 0.1]), \
             patch('random.randint', return_value=4):
             
             # 1回目: enabled変化なし, skip_hours変化あり
             params = {"enabled": True, "skip_hours": 12}
             new_params = filter_tool.mutate_params(params)
             assert new_params["enabled"] is True
             assert new_params["skip_hours"] == 16
             
             # 2回目: 境界値テスト (上限20)
             with patch('random.randint', return_value=5):
                 params = {"enabled": True, "skip_hours": 18}
                 new_params = filter_tool.mutate_params(params)
                 assert new_params["skip_hours"] == 20

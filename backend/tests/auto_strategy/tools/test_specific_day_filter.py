import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from app.services.auto_strategy.tools.specific_day_filter import SpecificDayFilter
from app.services.auto_strategy.tools.base import ToolContext

class TestSpecificDayFilter:
    @pytest.fixture
    def filter_tool(self):
        return SpecificDayFilter()

    @pytest.fixture
    def context(self):
        ctx = MagicMock(spec=ToolContext)
        return ctx

    def test_should_skip_entry_specific_days(self, filter_tool, context):
        # 木曜日(3)と金曜日(4)をスキップ設定
        params = {"enabled": True, "skip_days": [3, 4]}
        
        # 木曜日 (2023-01-05) -> スキップ
        context.timestamp = pd.Timestamp("2023-01-05")
        assert context.timestamp.dayofweek == 3
        assert filter_tool.should_skip_entry(context, params) is True
        
        # 金曜日 (2023-01-06) -> スキップ
        context.timestamp = pd.Timestamp("2023-01-06")
        assert context.timestamp.dayofweek == 4
        assert filter_tool.should_skip_entry(context, params) is True

        # 水曜日 (2023-01-04) -> スキップしない
        context.timestamp = pd.Timestamp("2023-01-04")
        assert context.timestamp.dayofweek == 2
        assert filter_tool.should_skip_entry(context, params) is False

    def test_should_skip_entry_empty_list(self, filter_tool, context):
        # リストが空ならスキップしない
        params = {"enabled": True, "skip_days": []}
        context.timestamp = pd.Timestamp("2023-01-05")
        assert filter_tool.should_skip_entry(context, params) is False

    def test_mutate_params(self, filter_tool):
        # グローバルの random をパッチ
        with patch('random.random', side_effect=[0.5, 0.1, 0.5, 0.1]), \
             patch('random.randint', return_value=3):
            
            # 1回目: enabled変化なし, day追加 (リスト空 -> 3を追加)
            params = {"enabled": True, "skip_days": []}
            new_params = filter_tool.mutate_params(params)
            assert new_params["enabled"] is True
            assert 3 in new_params["skip_days"]
            
            # 2回目: day削除 (リスト[3] -> 3を選んで削除)
            params = {"enabled": True, "skip_days": [3]}
            new_params = filter_tool.mutate_params(params)
            assert new_params["skip_days"] == []

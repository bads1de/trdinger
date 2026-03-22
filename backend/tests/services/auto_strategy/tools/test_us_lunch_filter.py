import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from app.services.auto_strategy.tools.us_lunch_filter import USLunchFilter
from app.services.auto_strategy.tools.base import ToolContext


class TestUSLunchFilter:
    @pytest.fixture
    def filter_tool(self):
        return USLunchFilter()

    @pytest.fixture
    def context(self):
        ctx = MagicMock(spec=ToolContext)
        return ctx

    def test_should_skip_entry_winter_lunch(self, filter_tool, context):
        # 冬時間 (UTC-5)
        # ランチタイム: 12:00-13:00 EST -> 17:00-18:00 UTC

        # 1月は冬時間
        # 17:00 UTC (12:00 EST) -> スキップ
        context.timestamp = pd.Timestamp("2023-01-04 17:00:00", tz="UTC")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True

        # 17:30 UTC (12:30 EST) -> スキップ
        context.timestamp = pd.Timestamp("2023-01-04 17:30:00", tz="UTC")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True

        # 16:59 UTC (11:59 EST) -> スキップしない
        context.timestamp = pd.Timestamp("2023-01-04 16:59:00", tz="UTC")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False

        # 18:00 UTC (13:00 EST) -> スキップしない
        context.timestamp = pd.Timestamp("2023-01-04 18:00:00", tz="UTC")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False

    def test_should_skip_entry_summer_lunch(self, filter_tool, context):
        # 夏時間 (UTC-4)
        # ランチタイム: 12:00-13:00 EDT -> 16:00-17:00 UTC

        # 7月は夏時間
        # 16:00 UTC (12:00 EDT) -> スキップ
        context.timestamp = pd.Timestamp("2023-07-04 16:00:00", tz="UTC")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True

        # 15:59 UTC (11:59 EDT) -> スキップしない
        context.timestamp = pd.Timestamp("2023-07-04 15:59:00", tz="UTC")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is False

    def test_should_skip_entry_naive_timestamp(self, filter_tool, context):
        # NaiveなTimestampが渡された場合、UTCとして扱われる
        # 冬時間 (1月) 17:00 -> スキップ
        context.timestamp = pd.Timestamp("2023-01-04 17:00:00")
        assert filter_tool.should_skip_entry(context, {"enabled": True}) is True

    def test_mutate_params(self, filter_tool):
        with patch("app.services.auto_strategy.tools.base.random") as mock_random:
            # enabled反転 (random < 0.2)
            mock_random.random.return_value = 0.1
            params = {"enabled": True}
            new_params = filter_tool.mutate_params(params)
            assert new_params["enabled"] is False

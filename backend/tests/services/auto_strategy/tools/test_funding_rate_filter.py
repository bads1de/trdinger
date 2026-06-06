"""
FundingRateFilter のユニットテスト

ファンディングレート(資金調達料金)を用いた市場歪み判定フィルターをテストします。
"""

from unittest.mock import patch

import pytest

from app.services.auto_strategy.tools.base import ToolContext
from app.services.auto_strategy.tools.funding_rate_filter import (
    FundingRateFilter,
)


class TestFundingRateFilter:
    @pytest.fixture
    def filter_tool(self) -> FundingRateFilter:
        return FundingRateFilter()

    @pytest.fixture
    def base_context(self) -> ToolContext:
        return ToolContext(
            timestamp=None,
            current_price=100.0,
            current_high=101.0,
            current_low=99.0,
            current_volume=0.0,
        )

    def test_default_params(self, filter_tool: FundingRateFilter) -> None:
        """デフォルトパラメータが定義通り"""
        params = filter_tool.get_default_params()
        assert params["enabled"] is True
        assert params["max_funding_rate"] == 0.001

    def test_definition_metadata(self, filter_tool: FundingRateFilter) -> None:
        """ツール定義のメタデータが期待通り"""
        assert filter_tool.name == "funding_rate_filter"
        assert filter_tool.definition.priority == "disabled"

    def test_should_skip_when_positive_rate_above_threshold(
        self, filter_tool: FundingRateFilter, base_context: ToolContext
    ) -> None:
        """正のファンディングレートが閾値を超えるとスキップ"""
        base_context.extra_data = {"funding_rate": 0.005}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"max_funding_rate": 0.001, "enabled": True}
            )
            is True
        )

    def test_should_skip_when_negative_rate_above_threshold(
        self, filter_tool: FundingRateFilter, base_context: ToolContext
    ) -> None:
        """負のファンディングレート(絶対値)が閾値を超えるとスキップ"""
        base_context.extra_data = {"funding_rate": -0.005}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"max_funding_rate": 0.001, "enabled": True}
            )
            is True
        )

    def test_should_allow_when_rate_below_threshold(
        self, filter_tool: FundingRateFilter, base_context: ToolContext
    ) -> None:
        """閾値未満の正のレートなら許可"""
        base_context.extra_data = {"funding_rate": 0.0005}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"max_funding_rate": 0.001, "enabled": True}
            )
            is False
        )

    def test_should_allow_when_negative_rate_below_threshold(
        self, filter_tool: FundingRateFilter, base_context: ToolContext
    ) -> None:
        """閾値未満の負のレート(|rate| < threshold)なら許可"""
        base_context.extra_data = {"funding_rate": -0.0005}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"max_funding_rate": 0.001, "enabled": True}
            )
            is False
        )

    def test_should_allow_when_rate_equals_threshold(
        self, filter_tool: FundingRateFilter, base_context: ToolContext
    ) -> None:
        """閾値と等しい場合は許可(> ではなく abs > なので境界値は許可)"""
        base_context.extra_data = {"funding_rate": 0.001}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"max_funding_rate": 0.001, "enabled": True}
            )
            is False
        )

    def test_should_allow_when_rate_is_zero(
        self, filter_tool: FundingRateFilter, base_context: ToolContext
    ) -> None:
        """レートが 0 の場合は許可"""
        base_context.extra_data = {"funding_rate": 0.0}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"max_funding_rate": 0.001, "enabled": True}
            )
            is False
        )

    def test_should_allow_when_funding_rate_missing(
        self, filter_tool: FundingRateFilter, base_context: ToolContext
    ) -> None:
        """funding_rate が extra_data にない場合は許可"""
        base_context.extra_data = {}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"max_funding_rate": 0.001, "enabled": True}
            )
            is False
        )

    def test_disabled_filter_never_skips(
        self, filter_tool: FundingRateFilter, base_context: ToolContext
    ) -> None:
        """enabled=False の場合はスキップしない"""
        base_context.extra_data = {"funding_rate": 0.5}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"max_funding_rate": 0.001, "enabled": False}
            )
            is False
        )

    def test_uses_default_threshold_when_not_provided(
        self, filter_tool: FundingRateFilter, base_context: ToolContext
    ) -> None:
        """max_funding_rate 未指定ならデフォルト 0.001"""
        base_context.extra_data = {"funding_rate": 0.0005}  # 0.0005 < 0.001
        assert filter_tool.should_skip_entry(base_context, {"enabled": True}) is False

    def test_mutate_params_does_not_mutate_input(
        self, filter_tool: FundingRateFilter
    ) -> None:
        """mutate_params は入力 dict を変更しない"""
        with patch("app.services.auto_strategy.tools.base.random") as mock_random:
            mock_random.random.return_value = 0.5
            params = {"enabled": True, "max_funding_rate": 0.001}
            original = dict(params)
            filter_tool.mutate_params(params)
            assert params == original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

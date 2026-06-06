"""
VolatilityFilter のユニットテスト

ATR またはボラティリティ(標準偏差)を用いたボラティリティ判定フィルターをテストします。
"""

from unittest.mock import patch

import pytest

from app.services.auto_strategy.tools.base import ToolContext
from app.services.auto_strategy.tools.volatility_filter import VolatilityFilter


class TestVolatilityFilter:
    @pytest.fixture
    def filter_tool(self) -> VolatilityFilter:
        return VolatilityFilter()

    @pytest.fixture
    def base_context(self) -> ToolContext:
        return ToolContext(
            timestamp=None,
            current_price=100.0,
            current_high=101.0,
            current_low=99.0,
            current_volume=0.0,
        )

    def test_default_params(self, filter_tool: VolatilityFilter) -> None:
        """デフォルトパラメータが定義通り"""
        params = filter_tool.get_default_params()
        assert params["enabled"] is True
        assert params["min_atr_pct"] == 0.001
        assert params["atr_period"] == 14

    def test_definition_metadata(self, filter_tool: VolatilityFilter) -> None:
        """ツール定義のメタデータが期待通り"""
        assert filter_tool.name == "volatility_filter"
        assert filter_tool.definition.priority == "disabled"

    def test_should_skip_when_atr_pct_below_threshold(
        self, filter_tool: VolatilityFilter, base_context: ToolContext
    ) -> None:
        """ATR 比率が閾値未満ならスキップ"""
        base_context.extra_data = {"atr": 0.05}  # 0.05 / 100 = 0.0005 < 0.001
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_atr_pct": 0.001, "enabled": True}
            )
            is True
        )

    def test_should_allow_when_atr_pct_above_threshold(
        self, filter_tool: VolatilityFilter, base_context: ToolContext
    ) -> None:
        """ATR 比率が閾値以上なら許可"""
        base_context.extra_data = {"atr": 0.5}  # 0.5 / 100 = 0.005 >= 0.001
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_atr_pct": 0.001, "enabled": True}
            )
            is False
        )

    def test_should_allow_when_atr_is_zero(
        self, filter_tool: VolatilityFilter, base_context: ToolContext
    ) -> None:
        """ATR が 0 の場合は許可(atr_value > 0 が False なので ATR 分岐に入らない)"""
        base_context.extra_data = {"atr": 0.0}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_atr_pct": 0.001, "enabled": True}
            )
            is False
        )

    def test_falls_back_to_volatility(
        self, filter_tool: VolatilityFilter, base_context: ToolContext
    ) -> None:
        """ATR が無いが volatility がある場合はそちらで判定"""
        # volatility = 0.05 / 100 = 0.0005 < 0.001 → スキップ
        base_context.extra_data = {"volatility": 0.05}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_atr_pct": 0.001, "enabled": True}
            )
            is True
        )

    def test_falls_back_to_volatility_allows(
        self, filter_tool: VolatilityFilter, base_context: ToolContext
    ) -> None:
        """volatility 比率が閾値以上なら許可"""
        # volatility = 0.5 / 100 = 0.005 >= 0.001
        base_context.extra_data = {"volatility": 0.5}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_atr_pct": 0.001, "enabled": True}
            )
            is False
        )

    def test_should_allow_when_data_missing(
        self, filter_tool: VolatilityFilter, base_context: ToolContext
    ) -> None:
        """ATR も volatility もない場合は許可(安全側)"""
        base_context.extra_data = {}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_atr_pct": 0.001, "enabled": True}
            )
            is False
        )

    def test_handles_zero_current_price(
        self, filter_tool: VolatilityFilter, base_context: ToolContext
    ) -> None:
        """current_price が 0 でも max(_, 1e-12) ガードでゼロ除算回避"""
        base_context.current_price = 0.0
        base_context.extra_data = {"atr": 0.5}  # 0.5 / 1e-12 → 十分大きい → 許可
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_atr_pct": 0.001, "enabled": True}
            )
            is False
        )

    def test_disabled_filter_never_skips(
        self, filter_tool: VolatilityFilter, base_context: ToolContext
    ) -> None:
        """enabled=False の場合はスキップしない"""
        base_context.extra_data = {"atr": 0.0001}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_atr_pct": 0.001, "enabled": False}
            )
            is False
        )

    def test_uses_default_threshold_when_not_provided(
        self, filter_tool: VolatilityFilter, base_context: ToolContext
    ) -> None:
        """min_atr_pct 未指定ならデフォルト 0.001"""
        base_context.extra_data = {"atr": 0.2}  # 0.2/100 = 0.002 >= 0.001
        assert filter_tool.should_skip_entry(base_context, {"enabled": True}) is False

    def test_mutate_params_does_not_mutate_input(
        self, filter_tool: VolatilityFilter
    ) -> None:
        """mutate_params は入力 dict を変更しない"""
        with patch("app.services.auto_strategy.tools.base.random") as mock_random:
            mock_random.random.return_value = 0.5
            params = {"enabled": True, "min_atr_pct": 0.001, "atr_period": 14}
            original = dict(params)
            filter_tool.mutate_params(params)
            assert params == original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

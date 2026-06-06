"""
TrendFilter のユニットテスト

ADX(平均方向性指数)を用いたトレンド強度判定フィルターをテストします。
"""

from unittest.mock import patch

import pytest

from app.services.auto_strategy.tools.base import ToolContext
from app.services.auto_strategy.tools.trend_filter import TrendFilter


class TestTrendFilter:
    @pytest.fixture
    def filter_tool(self) -> TrendFilter:
        return TrendFilter()

    @pytest.fixture
    def base_context(self) -> ToolContext:
        return ToolContext(
            timestamp=None,
            current_price=100.0,
            current_high=101.0,
            current_low=99.0,
            current_volume=0.0,
        )

    def test_default_params(self, filter_tool: TrendFilter) -> None:
        """デフォルトパラメータが定義通り"""
        params = filter_tool.get_default_params()
        assert params["enabled"] is True
        assert params["min_adx"] == 25.0
        assert params["adx_period"] == 14

    def test_definition_metadata(self, filter_tool: TrendFilter) -> None:
        """ツール定義のメタデータが期待通り"""
        assert filter_tool.name == "trend_filter"
        assert filter_tool.definition.priority == "optional"
        assert "ADX" in filter_tool.description

    def test_should_skip_entry_when_adx_below_threshold(
        self, filter_tool: TrendFilter, base_context: ToolContext
    ) -> None:
        """ADXが閾値未満ならスキップ"""
        base_context.extra_data = {"adx": 15.0}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_adx": 25.0, "enabled": True}
            )
            is True
        )

    def test_should_allow_entry_when_adx_above_threshold(
        self, filter_tool: TrendFilter, base_context: ToolContext
    ) -> None:
        """ADXが閾値以上なら許可"""
        base_context.extra_data = {"adx": 35.0}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_adx": 25.0, "enabled": True}
            )
            is False
        )

    def test_should_allow_entry_when_adx_equals_threshold(
        self, filter_tool: TrendFilter, base_context: ToolContext
    ) -> None:
        """ADXが閾値と等しい場合は許可(>=ではなく<)"""
        base_context.extra_data = {"adx": 25.0}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_adx": 25.0, "enabled": True}
            )
            is False
        )

    def test_should_allow_entry_when_adx_not_provided(
        self, filter_tool: TrendFilter, base_context: ToolContext
    ) -> None:
        """ADXがextra_dataにない場合は許可(安全側)"""
        base_context.extra_data = {}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_adx": 25.0, "enabled": True}
            )
            is False
        )

    def test_disabled_filter_never_skips(
        self, filter_tool: TrendFilter, base_context: ToolContext
    ) -> None:
        """enabled=False の場合は ADX が低くてもスキップしない"""
        base_context.extra_data = {"adx": 5.0}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_adx": 25.0, "enabled": False}
            )
            is False
        )

    def test_uses_default_threshold_when_not_provided(
        self, filter_tool: TrendFilter, base_context: ToolContext
    ) -> None:
        """min_adx が params にない場合はデフォルト 25.0 を使用"""
        base_context.extra_data = {"adx": 30.0}
        # min_adx 省略 → デフォルト 25.0 → 30 > 25 で許可
        assert filter_tool.should_skip_entry(base_context, {"enabled": True}) is False

    def test_mutate_params_disabled_toggle(self, filter_tool: TrendFilter) -> None:
        """random.random < 0.2 のとき enabled が反転する"""
        with patch("app.services.auto_strategy.tools.base.random") as mock_random:
            mock_random.random.return_value = 0.1
            params = {"enabled": True, "min_adx": 25.0, "adx_period": 14}
            new_params = filter_tool.mutate_params(params)
            assert new_params["enabled"] is False
            # min_adx, adx_period は random.uniform を呼ばない限り維持
            assert "min_adx" in new_params
            assert "adx_period" in new_params

    def test_mutate_params_preserves_when_random_high(
        self, filter_tool: TrendFilter
    ) -> None:
        """random.random >= 0.2 のとき enabled はそのまま(BaseTool 部分の振る舞い)"""
        with patch("app.services.auto_strategy.tools.base.random") as mock_random:
            # BaseTool の enabled 反転条件: random < 0.2 → mock を 0.5 にすれば反転しない
            mock_random.random.return_value = 0.5
            params = {"enabled": True, "min_adx": 25.0, "adx_period": 14}
            new_params = filter_tool.mutate_params(params)
            # サブクラス側の random.uniform は実 random を使うので、min_adx の値は変わる可能性があるが
            # キーは存在し、enabled は True のまま
            assert new_params["enabled"] is True
            assert "min_adx" in new_params
            assert "adx_period" in new_params

    def test_mutate_params_does_not_mutate_input(
        self, filter_tool: TrendFilter
    ) -> None:
        """mutate_params は入力 dict を変更しない"""
        with patch("app.services.auto_strategy.tools.base.random") as mock_random:
            mock_random.random.return_value = 0.5
            params = {"enabled": True, "min_adx": 25.0, "adx_period": 14}
            filter_tool.mutate_params(params)
            assert params == {"enabled": True, "min_adx": 25.0, "adx_period": 14}

    def test_validate_params_default_returns_true(
        self, filter_tool: TrendFilter
    ) -> None:
        """デフォルト実装の validate_params は True"""
        assert filter_tool.validate_params({"enabled": True, "min_adx": 25.0}) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
VolumeFilter のユニットテスト

出来高(Volume)を用いた流動性判定フィルターをテストします。
"""

from unittest.mock import patch

import pytest

from app.services.auto_strategy.tools.base import ToolContext
from app.services.auto_strategy.tools.volume_filter import VolumeFilter


class TestVolumeFilter:
    @pytest.fixture
    def filter_tool(self) -> VolumeFilter:
        return VolumeFilter()

    @pytest.fixture
    def base_context(self) -> ToolContext:
        return ToolContext(
            timestamp=None,
            current_price=100.0,
            current_high=101.0,
            current_low=99.0,
            current_volume=0.0,
        )

    def test_default_params(self, filter_tool: VolumeFilter) -> None:
        """デフォルトパラメータが定義通り"""
        params = filter_tool.get_default_params()
        assert params["enabled"] is True
        assert params["min_volume_ratio"] == 0.5
        assert params["volume_period"] == 20

    def test_definition_metadata(self, filter_tool: VolumeFilter) -> None:
        """ツール定義のメタデータが期待通り"""
        assert filter_tool.name == "volume_filter"
        assert filter_tool.definition.priority == "optional"
        assert (
            "出来高" in filter_tool.description or "Volume" in filter_tool.description
        )

    def test_should_skip_when_volume_ratio_below_threshold(
        self, filter_tool: VolumeFilter, base_context: ToolContext
    ) -> None:
        """出来高比率が閾値未満ならスキップ"""
        base_context.extra_data = {"current_volume": 50.0, "avg_volume": 200.0}
        # 50/200 = 0.25 < 0.5
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_volume_ratio": 0.5, "enabled": True}
            )
            is True
        )

    def test_should_allow_when_volume_ratio_above_threshold(
        self, filter_tool: VolumeFilter, base_context: ToolContext
    ) -> None:
        """出来高比率が閾値以上なら許可"""
        base_context.extra_data = {"current_volume": 150.0, "avg_volume": 200.0}
        # 150/200 = 0.75 >= 0.5
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_volume_ratio": 0.5, "enabled": True}
            )
            is False
        )

    def test_should_allow_when_volume_ratio_equals_threshold(
        self, filter_tool: VolumeFilter, base_context: ToolContext
    ) -> None:
        """出来高比率が閾値と等しい場合は許可"""
        base_context.extra_data = {"current_volume": 100.0, "avg_volume": 200.0}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_volume_ratio": 0.5, "enabled": True}
            )
            is False
        )

    def test_should_allow_when_data_missing(
        self, filter_tool: VolumeFilter, base_context: ToolContext
    ) -> None:
        """current_volume/avg_volume どちらもない場合は許可"""
        base_context.extra_data = {}
        base_context.current_volume = 0.0
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_volume_ratio": 0.5, "enabled": True}
            )
            is False
        )

    def test_should_allow_when_avg_volume_is_zero(
        self, filter_tool: VolumeFilter, base_context: ToolContext
    ) -> None:
        """avg_volume が 0 の場合は許可(ゼロ除算回避)"""
        base_context.extra_data = {"current_volume": 100.0, "avg_volume": 0.0}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_volume_ratio": 0.5, "enabled": True}
            )
            is False
        )

    def test_falls_back_to_context_current_volume(
        self, filter_tool: VolumeFilter, base_context: ToolContext
    ) -> None:
        """extra_data に current_volume が無いが context.current_volume がある場合はそちらを使う"""
        base_context.extra_data = {"avg_volume": 100.0}
        base_context.current_volume = 200.0
        # context.current_volume / avg_volume = 2.0 >= 0.5 → 許可
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_volume_ratio": 0.5, "enabled": True}
            )
            is False
        )

    def test_falls_back_to_context_current_volume_skip(
        self, filter_tool: VolumeFilter, base_context: ToolContext
    ) -> None:
        """context.current_volume < avg_volume * ratio の場合はスキップ"""
        base_context.extra_data = {"avg_volume": 1000.0}
        base_context.current_volume = 10.0
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_volume_ratio": 0.5, "enabled": True}
            )
            is True
        )

    def test_disabled_filter_never_skips(
        self, filter_tool: VolumeFilter, base_context: ToolContext
    ) -> None:
        """enabled=False の場合は出来高が低くてもスキップしない"""
        base_context.extra_data = {"current_volume": 1.0, "avg_volume": 1000.0}
        assert (
            filter_tool.should_skip_entry(
                base_context, {"min_volume_ratio": 0.5, "enabled": False}
            )
            is False
        )

    def test_uses_default_threshold_when_not_provided(
        self, filter_tool: VolumeFilter, base_context: ToolContext
    ) -> None:
        """min_volume_ratio 未指定ならデフォルト 0.5"""
        base_context.extra_data = {"current_volume": 60.0, "avg_volume": 100.0}
        # 60/100 = 0.6 >= 0.5 → 許可
        assert filter_tool.should_skip_entry(base_context, {"enabled": True}) is False

    def test_mutate_params_does_not_mutate_input(
        self, filter_tool: VolumeFilter
    ) -> None:
        """mutate_params は入力 dict を変更しない"""
        with patch("app.services.auto_strategy.tools.base.random") as mock_random:
            mock_random.random.return_value = 0.5
            params = {
                "enabled": True,
                "min_volume_ratio": 0.5,
                "volume_period": 20,
            }
            original = dict(params)
            filter_tool.mutate_params(params)
            assert params == original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

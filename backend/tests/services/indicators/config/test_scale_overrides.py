"""
インジケータースケール分類のオーバーライドテスト

_SPECIAL_CONFIG_OVERRIDES による明示的なスケール修正が
レジストリに反映されていることを検証します。

スケール推定ヒューリスティックの誤分類（RSI等が price_ratio 扱いになり
`RSI > close` のような恒偽条件が生成される問題）の回帰防止が目的です。
"""

import pytest

from app.services.indicators.config import indicator_registry
from app.services.indicators.config.discovery import IndicatorScaleType


class TestScaleTypeOverrides:
    """代表的インジケーターのスケール分類テスト"""

    @pytest.mark.parametrize(
        ("indicator_name", "expected_scale"),
        [
            ("RSI", IndicatorScaleType.OSCILLATOR_0_100),
            ("STOCH", IndicatorScaleType.OSCILLATOR_0_100),
            ("MFI", IndicatorScaleType.OSCILLATOR_0_100),
            ("CHOP", IndicatorScaleType.OSCILLATOR_0_100),
            ("ADX", IndicatorScaleType.OSCILLATOR_0_100),
            ("MACD", IndicatorScaleType.MOMENTUM_ZERO_CENTERED),
            ("TRIX", IndicatorScaleType.MOMENTUM_ZERO_CENTERED),
            ("CMF", IndicatorScaleType.MOMENTUM_ZERO_CENTERED),
            ("PVO", IndicatorScaleType.MOMENTUM_ZERO_CENTERED),
            ("VFI", IndicatorScaleType.MOMENTUM_ZERO_CENTERED),
            ("EMA", IndicatorScaleType.PRICE_ABSOLUTE),
            ("VWAP", IndicatorScaleType.PRICE_ABSOLUTE),
            ("SUPERTREND", IndicatorScaleType.PRICE_ABSOLUTE),
        ],
    )
    def test_scale_type_is_explicitly_overridden(
        self, indicator_name: str, expected_scale: IndicatorScaleType
    ) -> None:
        """明示オーバーライド済みの指標は正しいスケール分類を持つ"""
        indicator_registry.ensure_initialized()
        config = indicator_registry.get_indicator_config(indicator_name)

        assert config is not None, f"{indicator_name} が実装されていません"
        assert config.scale_type == expected_scale

    def test_rsi_has_numeric_thresholds(self) -> None:
        """RSIはclose比較ではなく数値閾値を持つ"""
        indicator_registry.ensure_initialized()
        config = indicator_registry.get_indicator_config("RSI")

        assert config is not None
        assert config.thresholds
        normal = config.thresholds["normal"]
        assert isinstance(normal["long_gt"], (int, float))
        assert isinstance(normal["short_lt"], (int, float))

    def test_adx_threshold_is_classic_trend_level(self) -> None:
        """ADXはトレンド判定の古典的閾値（25）を持つ"""
        indicator_registry.ensure_initialized()
        config = indicator_registry.get_indicator_config("ADX")

        assert config is not None
        assert config.thresholds["normal"] == {"long_gt": 25, "short_lt": 25}

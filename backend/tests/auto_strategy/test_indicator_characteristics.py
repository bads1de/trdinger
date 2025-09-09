"""
テスト: 指標特性の動的初期化
"""
import pytest
from backend.app.services.auto_strategy.utils.indicator_characteristics import INDICATOR_CHARACTERISTICS
from backend.app.services.auto_strategy.utils.indicator_characteristics import _get_merged_characteristics


class TestIndicatorCharacteristics:
    """指標特性の動的生成テスト"""

    def test_get_merged_characteristics_returns_dict(self):
        """_get_merged_characteristics が辞書を返すことを確認"""
        original = {"test": {"type": "test"}}
        result = _get_merged_characteristics(original)
        assert isinstance(result, dict)

    def test_indicator_characteristics_has_expected_keys(self):
        """INDICATOR_CHARACTERISTICS に期待されるキーが含まれていることを確認"""
        expected_keys = ["RSI", "MACD", "SMA", "EMA", "BBANDS"]  # 例として主要なもの
        for key in expected_keys:
            assert key in INDICATOR_CHARACTERISTICS

    def test_indicator_characteristics_structure(self):
        """INDICATOR_CHARACTERISTICS の各値が正しい構造を持っていることを確認"""
        for indicator, characteristics in INDICATOR_CHARACTERISTICS.items():
            assert "type" in characteristics
            assert isinstance(characteristics["type"], str)

    def test_indicator_characteristics_has_rsi_details(self):
        """RSI指標の特性が正しく設定されていることを確認"""
        rsi = INDICATOR_CHARACTERISTICS.get("RSI")
        if rsi:
            assert rsi["type"] == "momentum"
            assert "range" in rsi
            assert "long_zones" in rsi
            assert "short_zones" in rsi
            assert "neutral_zone" in rsi

    def test_indicator_characteristics_has_moving_averages(self):
        """移動平均系の指標が含まれていることを確認"""
        moving_averages = ["SMA", "EMA"]
        for ma in moving_averages:
            if ma in INDICATOR_CHARACTERISTICS:
                characteristics = INDICATOR_CHARACTERISTICS[ma]
                assert characteristics["type"] == "trend"
                assert characteristics.get("price_comparison", False)
                assert characteristics.get("trend_following", False)
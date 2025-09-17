"""
STOCHのmin_length設定に関するテストケース
"""

import pytest
from app.services.indicators.config.indicator_definitions import PANDAS_TA_CONFIG


class TestStochMinLength:
    """STOCHのmin_length設定をテスト"""

    def test_stoch_min_length_default(self):
        """デフォルトパラメータでのmin_length計算をテスト"""
        stoch_config = PANDAS_TA_CONFIG["STOCH"]
        min_length = stoch_config["min_length"]({})
        expected = 14 + 3 + 3  # k_length=14, smooth_k=3, d_length=3
        assert min_length == expected, f"Expected {expected}, got {min_length}"

    def test_stoch_min_length_custom_params(self):
        """カスタムパラメータでのmin_length計算をテスト"""
        stoch_config = PANDAS_TA_CONFIG["STOCH"]
        params = {"k_length": 10, "smooth_k": 2, "d_length": 5}
        min_length = stoch_config["min_length"](params)
        expected = 10 + 2 + 5  # k_length=10, smooth_k=2, d_length=5
        assert min_length == expected, f"Expected {expected}, got {min_length}"

    def test_stoch_min_length_partial_params(self):
        """一部のパラメータのみ指定でのmin_length計算をテスト"""
        stoch_config = PANDAS_TA_CONFIG["STOCH"]
        params = {"k_length": 12, "d_length": 4}
        min_length = stoch_config["min_length"](params)
        expected = 12 + 3 + 4  # k_length=12, smooth_k=3 (default), d_length=4
        assert min_length == expected, f"Expected {expected}, got {min_length}"

    def test_stoch_min_length_empty_params(self):
        """空のパラメータでのmin_length計算をテスト"""
        stoch_config = PANDAS_TA_CONFIG["STOCH"]
        min_length = stoch_config["min_length"]({})
        expected = 14 + 3 + 3  # デフォルト値の合計
        assert min_length == expected, f"Expected {expected}, got {min_length}"
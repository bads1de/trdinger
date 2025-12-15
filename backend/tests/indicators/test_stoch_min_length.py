"""
STOCHのmin_length設定に関するテストケース
"""

from app.services.indicators.config.indicator_config import indicator_registry


class TestStochMinLength:
    """STOCHのmin_length設定をテスト"""

    def test_stoch_min_length_default(self):
        """デフォルトパラメータでのmin_length計算をテスト"""
        stoch_config = indicator_registry.get_indicator_config("STOCH")
        assert stoch_config is not None, "STOCH config not found"
        min_length = stoch_config.min_length_func({})
        expected = 14 + 3 + 3  # k_length=14, smooth_k=3, d_length=3
        assert min_length == expected, f"Expected {expected}, got {min_length}"

    def test_stoch_min_length_custom_params(self):
        """カスタムパラメータでのmin_length計算をテスト"""
        stoch_config = indicator_registry.get_indicator_config("STOCH")
        assert stoch_config is not None, "STOCH config not found"
        params = {"k_length": 10, "smooth_k": 2, "d_length": 5}
        min_length = stoch_config.min_length_func(params)
        expected = 10 + 2 + 5  # k_length=10, smooth_k=2, d_length=5
        assert min_length == expected, f"Expected {expected}, got {min_length}"

    def test_stoch_min_length_partial_params(self):
        """一部のパラメータのみ指定でのmin_length計算をテスト"""
        stoch_config = indicator_registry.get_indicator_config("STOCH")
        assert stoch_config is not None, "STOCH config not found"
        params = {"k_length": 12, "d_length": 4}
        min_length = stoch_config.min_length_func(params)
        expected = 12 + 3 + 4  # k_length=12, smooth_k=3 (default), d_length=4
        assert min_length == expected, f"Expected {expected}, got {min_length}"

    def test_stoch_min_length_empty_params(self):
        """空のパラメータでのmin_length計算をテスト"""
        stoch_config = indicator_registry.get_indicator_config("STOCH")
        assert stoch_config is not None, "STOCH config not found"
        min_length = stoch_config.min_length_func({})
        expected = 14 + 3 + 3  # デフォルト値の合計
        assert min_length == expected, f"Expected {expected}, got {min_length}"



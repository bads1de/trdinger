"""
STOCHのmin_length設定に関するテストケース

注意: このテストは pandas-ta のイントロスペクションベースの min_length 計算を使用します。
pandas-ta の stoch 実装では _length = max(k, d, smooth_k) が使われています。
"""

from app.services.indicators.config.indicator_config import indicator_registry
from app.services.indicators.config.pandas_ta_introspection import calculate_min_length


class TestStochMinLength:
    """STOCHのmin_length設定をテスト"""

    def test_stoch_min_length_default(self):
        """デフォルトパラメータでのmin_length計算をテスト"""
        stoch_config = indicator_registry.get_indicator_config("STOCH")
        assert stoch_config is not None, "STOCH config not found"

        # pandas-ta の実装では max(k, d, smooth_k) = max(14, 3, 3) = 14
        min_length = stoch_config.min_length_func({})
        expected = 14  # max(k=14, d=3, smooth_k=3)
        assert min_length == expected, f"Expected {expected}, got {min_length}"

    def test_stoch_min_length_custom_params(self):
        """カスタムパラメータでのmin_length計算をテスト"""
        stoch_config = indicator_registry.get_indicator_config("STOCH")
        assert stoch_config is not None, "STOCH config not found"

        # max(10, 5, 2) = 10
        params = {"k": 10, "smooth_k": 2, "d": 5}
        min_length = stoch_config.min_length_func(params)
        expected = 10  # max(k=10, d=5, smooth_k=2)
        assert min_length == expected, f"Expected {expected}, got {min_length}"

    def test_stoch_min_length_partial_params(self):
        """一部のパラメータのみ指定でのmin_length計算をテスト"""
        stoch_config = indicator_registry.get_indicator_config("STOCH")
        assert stoch_config is not None, "STOCH config not found"

        # max(12, 4, 3) = 12 (smooth_k defaults to 3)
        params = {"k": 12, "d": 4}
        min_length = stoch_config.min_length_func(params)
        expected = 12  # max(k=12, d=4, smooth_k=3)
        assert min_length == expected, f"Expected {expected}, got {min_length}"

    def test_stoch_min_length_empty_params(self):
        """空のパラメータでのmin_length計算をテスト（イントロスペクション直接確認）"""
        # イントロスペクションを直接テスト
        min_length = calculate_min_length("stoch", {})
        # デフォルト: max(k=14, d=3, smooth_k=3) = 14
        expected = 14
        assert min_length == expected, f"Expected {expected}, got {min_length}"

    def test_stoch_min_length_introspection_consistency(self):
        """indicator_registry と introspection の一貫性をテスト"""
        stoch_config = indicator_registry.get_indicator_config("STOCH")
        assert stoch_config is not None

        params = {"k": 20, "d": 5, "smooth_k": 3}

        # 両方の方法で同じ結果が得られることを確認
        registry_result = stoch_config.min_length_func(params)
        introspection_result = calculate_min_length("stoch", params)

        assert (
            registry_result == introspection_result
        ), f"Registry: {registry_result}, Introspection: {introspection_result}"

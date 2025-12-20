"""
discovery.py のリファクタリングテスト

新しい pandas_ta_introspection モジュールを使用した
動的な指標検出のテスト
"""


class TestDynamicDiscoveryWithIntrospection:
    """イントロスペクションを使用した動的検出のテスト"""

    def test_discover_rsi_min_length_is_dynamic(self):
        """RSIのmin_lengthが動的に取得される"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery
        from app.services.indicators.config.pandas_ta_introspection import (
            calculate_min_length,
        )

        configs = DynamicIndicatorDiscovery.discover_all()
        rsi_config = next((c for c in configs if c.indicator_name == "RSI"), None)

        assert rsi_config is not None

        # 新しいイントロスペクションでの計算結果を確認
        introspection_length = calculate_min_length("rsi", {"length": 14})
        assert introspection_length == 14

    def test_discover_macd_min_length_is_dynamic(self):
        """MACDのmin_lengthが動的に取得される"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery
        from app.services.indicators.config.pandas_ta_introspection import (
            calculate_min_length,
        )

        configs = DynamicIndicatorDiscovery.discover_all()
        macd_config = next((c for c in configs if c.indicator_name == "MACD"), None)

        assert macd_config is not None

        # 新しいイントロスペクションでの計算結果を確認
        # max(12, 26, 9) = 26
        introspection_length = calculate_min_length(
            "macd", {"fast": 12, "slow": 26, "signal": 9}
        )
        assert introspection_length == 26

    def test_discover_stoch_min_length_is_dynamic(self):
        """STOCHのmin_lengthが動的に取得される"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery
        from app.services.indicators.config.pandas_ta_introspection import (
            calculate_min_length,
        )

        configs = DynamicIndicatorDiscovery.discover_all()
        stoch_config = next((c for c in configs if c.indicator_name == "STOCH"), None)

        assert stoch_config is not None

        # 新しいイントロスペクションでの計算結果を確認
        # max(14, 3, 3) = 14
        introspection_length = calculate_min_length(
            "stoch", {"k": 14, "d": 3, "smooth_k": 3}
        )
        assert introspection_length == 14

    def test_no_manual_overrides_for_min_length(self):
        """手動オーバーライドなしでmin_lengthが取得される"""
        from app.services.indicators.config.pandas_ta_introspection import (
            calculate_min_length,
        )

        # 複数の指標でテスト
        test_cases = [
            ("rsi", {"length": 14}, 14),
            ("macd", {"fast": 12, "slow": 26, "signal": 9}, 26),
            ("atr", {"length": 14}, 14),
            ("cci", {"length": 20}, 20),
        ]

        for indicator, params, expected in test_cases:
            result = calculate_min_length(indicator, params)
            assert result == expected, f"{indicator}: expected {expected}, got {result}"


class TestReducedConstants:
    """定数が削減されていることのテスト"""

    def test_indicator_overrides_is_removed(self):
        """INDICATOR_OVERRIDES が完全に削除されていること"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery

        # 定数は削除され、ロジックに移行
        assert not hasattr(
            DynamicIndicatorDiscovery, "INDICATOR_OVERRIDES"
        ), "INDICATOR_OVERRIDES should be removed"

    def test_min_length_formulas_constant_removed(self):
        """_MIN_LENGTH_FORMULAS 定数は完全に削除された"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery

        # この定数は完全に削除されている
        assert not hasattr(DynamicIndicatorDiscovery, "_MIN_LENGTH_FORMULAS")

    def test_introspection_replaces_manual_formulas(self):
        """イントロスペクションが手動定数を置き換える"""
        from app.services.indicators.config.pandas_ta_introspection import (
            calculate_min_length,
        )

        # イントロスペクションで計算できることを確認
        assert calculate_min_length("rsi", {"length": 14}) == 14
        assert calculate_min_length("macd", {"fast": 12, "slow": 26, "signal": 9}) == 26

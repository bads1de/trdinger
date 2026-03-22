"""
定数削減テスト

EXCLUDED_FUNCTIONS, DATA_ARGUMENTS の動的化テスト
"""


class TestExcludedFunctionsDynamic:
    """EXCLUDED_FUNCTIONS の動的化テスト"""

    def test_excluded_functions_is_removed(self):
        """EXCLUDED_FUNCTIONS が削除されていること"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery

        # EXCLUDED_FUNCTIONS は動的検出に移行
        assert not hasattr(
            DynamicIndicatorDiscovery, "EXCLUDED_FUNCTIONS"
        ), "EXCLUDED_FUNCTIONS should be removed (use dynamic detection)"

    def test_utility_functions_are_excluded(self):
        """ユーティリティ関数が検出から除外されること"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery

        configs = DynamicIndicatorDiscovery.discover_all()
        names = {c.indicator_name.lower() for c in configs}

        # ユーティリティ関数が含まれていないこと
        utility_funcs = ["above", "below", "cross", "verify_series", "utils"]
        for func in utility_funcs:
            assert func not in names, f"{func} should be excluded"

    def test_real_indicators_are_included(self):
        """実際の指標が検出されること"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery

        configs = DynamicIndicatorDiscovery.discover_all()
        names = {c.indicator_name for c in configs}

        # 一般的な指標が含まれていること
        real_indicators = {"RSI", "MACD", "STOCH", "ATR", "SMA", "EMA"}
        for ind in real_indicators:
            assert ind in names, f"{ind} should be included"


class TestDataArgumentsDynamic:
    """DATA_ARGUMENTS の動的化テスト"""

    def test_data_arguments_is_minimal(self):
        """DATA_ARGUMENTS が最小限であること"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery

        # DATA_ARGUMENTS はプロジェクト固有の追加分のみ
        if hasattr(DynamicIndicatorDiscovery, "DATA_ARGUMENTS"):
            data_args = DynamicIndicatorDiscovery.DATA_ARGUMENTS
            # 基本5つは動的取得されるべき
            basic = {"open", "high", "low", "close", "volume"}
            overlap = basic & data_args
            assert (
                len(overlap) == 0
            ), f"Basic data args {overlap} should be dynamically detected"


class TestParameterRulesRemoved:
    """PARAMETER_RULES の完全削除テスト"""

    def test_parameter_rules_is_removed(self):
        """PARAMETER_RULES が完全に削除されていること"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery

        # PARAMETER_RULES は動的計算に移行
        assert not hasattr(
            DynamicIndicatorDiscovery, "PARAMETER_RULES"
        ), "PARAMETER_RULES should be removed (use dynamic calculation)"

    def test_parameters_have_sensible_ranges(self):
        """パラメータが合理的な範囲を持つこと"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery

        configs = DynamicIndicatorDiscovery.discover_all()

        # RSI のパラメータをチェック
        rsi_config = next((c for c in configs if c.indicator_name == "RSI"), None)
        if rsi_config and rsi_config.parameters:
            length_param = rsi_config.parameters.get("length")
            if length_param:
                # デフォルト値から計算された範囲
                assert length_param.min_value > 0
                assert length_param.max_value > length_param.min_value
                assert (
                    length_param.min_value
                    <= length_param.default_value
                    <= length_param.max_value
                )

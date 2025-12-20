"""
discovery.py 統合テスト

イントロスペクションモジュールを使用した動的min_length取得のテスト
"""


class TestDiscoveryWithIntrospection:
    """イントロスペクション統合後のdiscoveryテスト"""

    def test_discover_uses_introspection_for_min_length(self):
        """discover_allがイントロスペクションを使用していること"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery

        configs = DynamicIndicatorDiscovery.discover_all()

        # RSI の min_length_func をテスト
        rsi_config = next((c for c in configs if c.indicator_name == "RSI"), None)
        assert rsi_config is not None

        if rsi_config.min_length_func:
            # イントロスペクションベースなら length=14 で 14 を返す
            result = rsi_config.min_length_func({"length": 14})
            assert result == 14, f"RSI min_length expected 14, got {result}"

    def test_macd_min_length_uses_max_formula(self):
        """MACDのmin_lengthがmax(fast, slow, signal)を使用"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery

        configs = DynamicIndicatorDiscovery.discover_all()
        macd_config = next((c for c in configs if c.indicator_name == "MACD"), None)

        assert macd_config is not None

        if macd_config.min_length_func:
            result = macd_config.min_length_func({"fast": 12, "slow": 26, "signal": 9})
            # max(12, 26, 9) = 26
            assert result == 26, f"MACD min_length expected 26, got {result}"

    def test_stoch_min_length_uses_max_formula(self):
        """STOCHのmin_lengthがmax(k, d, smooth_k)を使用"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery

        configs = DynamicIndicatorDiscovery.discover_all()
        stoch_config = next((c for c in configs if c.indicator_name == "STOCH"), None)

        assert stoch_config is not None

        if stoch_config.min_length_func:
            result = stoch_config.min_length_func({"k": 14, "d": 3, "smooth_k": 3})
            # max(14, 3, 3) = 14
            assert result == 14, f"STOCH min_length expected 14, got {result}"

    def test_min_length_formulas_constant_is_deleted(self):
        """_MIN_LENGTH_FORMULAS 定数が完全に削除されていること"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery

        # _MIN_LENGTH_FORMULAS は完全に削除されるべき
        assert not hasattr(
            DynamicIndicatorDiscovery, "_MIN_LENGTH_FORMULAS"
        ), "_MIN_LENGTH_FORMULAS should be completely removed"

    def test_indicator_overrides_is_removed(self):
        """INDICATOR_OVERRIDES が完全に削除されていること"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery

        # 定数は削除され、ロジックに移行
        assert not hasattr(
            DynamicIndicatorDiscovery, "INDICATOR_OVERRIDES"
        ), "INDICATOR_OVERRIDES should be removed"


class TestDiscoveryBackwardCompatibility:
    """後方互換性のテスト"""

    def test_existing_configs_still_work(self):
        """既存の設定が引き続き動作すること"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery

        configs = DynamicIndicatorDiscovery.discover_all()

        # 基本的な指標が検出されていること
        indicator_names = {c.indicator_name for c in configs}

        expected_indicators = {"RSI", "MACD", "STOCH", "BBANDS", "ATR", "SMA", "EMA"}
        for expected in expected_indicators:
            assert expected in indicator_names, f"{expected} should be discovered"

    def test_return_cols_still_work(self):
        """return_colsが引き続き動作すること"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery

        configs = DynamicIndicatorDiscovery.discover_all()

        # MACDのreturn_colsをチェック
        macd_config = next((c for c in configs if c.indicator_name == "MACD"), None)
        assert macd_config is not None

        # return_colsが設定されていること（OVERRIDESから）
        if macd_config.return_cols:
            assert len(macd_config.return_cols) >= 3

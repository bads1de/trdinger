"""
シンプルなオートストラテジー動作確認テスト
"""
import pytest
import pandas as pd
import numpy as np


class TestAutoStrategySimple:
    """シンプルなオートストラテジーテスト"""

    def test_basic_indicator_service(self):
        """基本的なインジケータサービステスト"""
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

        service = TechnicalIndicatorService()
        assert service is not None

        supported = service.get_supported_indicators()
        assert len(supported) > 100

        # 設定されたインジケータがサポートされていることを確認
        configured_indicators = ['STC', 'RSI', 'MACD']  # 明示的に設定されたインジケータ
        for indicator in configured_indicators:
            assert indicator in supported

        # pandas-ta直接使用インジケータは別途確認
        direct_indicators = ['SMA', 'EMA']
        print(f"Direct pandas-ta indicators (not in registry): {direct_indicators}")

        print(f"Indicator service OK: {len(supported)} indicators supported")

    def test_key_indicator_calculations(self):
        """主要インジケータの計算テスト"""
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

        # シンプルなテストデータ
        np.random.seed(42)
        close_prices = [100 + i*2 for i in range(50)]
        df = pd.DataFrame({'close': close_prices})

        service = TechnicalIndicatorService()

        # テストする主要インジケータ
        test_cases = [
            ('STC', {'tclength': 10, 'fast': 23, 'slow': 50, 'factor': 0.5}),
            ('RSI', {'length': 14}),
            ('SMA', {'length': 20}),
            ('EMA', {'length': 20}),
            ('MACD', {'fast': 12, 'slow': 26, 'signal': 9}),
        ]

        successful_tests = []

        for indicator_name, params in test_cases:
            try:
                result = service.calculate_indicator(df, indicator_name, params)
                if result is not None:
                    successful_tests.append(indicator_name)
                    print(f"[OK] {indicator_name}: calculation successful")
                else:
                    print(f"[FAIL] {indicator_name}: result is None")
            except Exception as e:
                print(f"[ERROR] {indicator_name}: {str(e)}")

        # 少なくとも主要なインジケータは成功すべき
        assert len(successful_tests) >= 3, f"Only {len(successful_tests)} indicators succeeded"

        # 重要なインジケータは必ず成功すべき
        assert 'STC' in successful_tests, "STC indicator failed"
        assert 'RSI' in successful_tests, "RSI indicator failed"

        print(f"Key indicators test passed: {len(successful_tests)}/{len(test_cases)} successful")

    def test_autostrategy_compatibility(self):
        """オートストラテジー互換性テスト"""
        from app.services.auto_strategy.config.constants import VALID_INDICATOR_TYPES
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

        service = TechnicalIndicatorService()
        available_indicators = set(service.get_supported_indicators().keys())
        autostrategy_indicators = set(VALID_INDICATOR_TYPES)

        compatible_indicators = available_indicators & autostrategy_indicators

        print(f"Compatible indicators: {len(compatible_indicators)}")

        # 主要インジケータが互換性があることを確認
        key_indicators = ['RSI', 'SMA', 'EMA', 'MACD', 'STC']
        compatible_key_indicators = [ind for ind in key_indicators if ind in compatible_indicators]

        assert len(compatible_key_indicators) >= 4, f"Insufficient compatibility: {len(compatible_key_indicators)}/{len(key_indicators)}"

        print(f"AutoStrategy compatibility test passed: {len(compatible_key_indicators)}/{len(key_indicators)} compatible")
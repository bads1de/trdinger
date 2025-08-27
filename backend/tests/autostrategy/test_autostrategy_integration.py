"""
オートストラテジー統合テスト
"""
import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Any


class TestAutoStrategyIntegration:
    """オートストラテジー統合テスト"""

    def test_indicator_service_for_autostrategy(self):
        """オートストラテジー用のインジケータサービステスト"""
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

        service = TechnicalIndicatorService()
        assert service is not None

        # オートストラテジーで使用される主要インジケータ
        key_indicators = ['RSI', 'SMA', 'EMA', 'MACD', 'BBANDS', 'STC', 'ADX', 'CCI', 'ATR', 'MFI']

        supported = service.get_supported_indicators()
        assert len(supported) > 100, f"サポートインジケータが不足: {len(supported)}個"

        # 主要インジケータがサポートされていることを確認
        for indicator in key_indicators:
            assert indicator in supported, f"主要インジケータ{indicator}がサポートされていません"

        print(f"オートストラテジー用インジケータサービス正常: {len(supported)}個のインジケータ")

    def test_indicator_calculation_for_autostrategy(self):
        """オートストラテジー用のインジケータ計算テスト"""
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

        # テストデータ作成
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')

        base_price = 50000
        price_changes = np.random.normal(0, 0.02, 100)
        close_prices = [base_price]
        for change in price_changes[1:]:
            new_price = close_prices[-1] * (1 + change)
            close_prices.append(max(1, new_price))

        df = pd.DataFrame({
            'timestamp': dates,
            'open': [price * (1 + np.random.normal(0, 0.005)) for price in close_prices],
            'high': [price * (1 + abs(np.random.normal(0, 0.01))) for price in close_prices],
            'low': [price * (1 - abs(np.random.normal(0, 0.01))) for price in close_prices],
            'close': close_prices,
            'volume': np.random.uniform(1000000, 10000000, 100)
        })

        service = TechnicalIndicatorService()

        # オートストラテジーで使用される主要インジケータのテスト
        test_cases = [
            ('RSI', {'length': 14}, 'single'),
            ('SMA', {'length': 20}, 'single'),
            ('EMA', {'length': 20}, 'single'),
            ('MACD', {'fast': 12, 'slow': 26, 'signal': 9}, 'complex'),
            ('BBANDS', {'length': 20, 'std': 2.0}, 'complex'),
            ('STC', {'tclength': 10, 'fast': 23, 'slow': 50, 'factor': 0.5}, 'single'),
            ('ADX', {'length': 14}, 'single'),
            ('CCI', {'period': 14}, 'single'),
            ('ATR', {'period': 14}, 'single'),
            ('MFI', {'length': 14}, 'single'),
        ]

        successful_calculations = []
        failed_calculations = []

        for indicator_name, params, expected_type in test_cases:
            try:
                result = service.calculate_indicator(df, indicator_name, params)

                if result is not None:
                    # 結果の検証
                    if expected_type == 'single':
                        assert isinstance(result, np.ndarray), f"{indicator_name}: 結果がndarrayではありません"
                        assert len(result) == len(df), f"{indicator_name}: 結果の長さが不正"
                        valid_values = np.sum(~np.isnan(result))
                        assert valid_values > 0, f"{indicator_name}: 有効な値がありません"
                    elif expected_type == 'complex':
                        assert isinstance(result, tuple), f"{indicator_name}: 結果がtupleではありません"
                        assert len(result) > 0, f"{indicator_name}: tupleが空です"

                    successful_calculations.append(indicator_name)
                    print(f"[OK] {indicator_name}: 計算成功")
                else:
                    failed_calculations.append(indicator_name)
                    print(f"[FAIL] {indicator_name}: 結果がNone")

            except Exception as e:
                failed_calculations.append(indicator_name)
                print(f"❌ {indicator_name}: 例外発生 - {str(e)}")

        # 結果のサマリー
        print("\n=== オートストラテジー用インジケータ計算結果 ===")
        print(f"成功: {len(successful_calculations)}個")
        print(f"失敗: {len(failed_calculations)}個")
        print(f"総計: {len(test_cases)}個")

        if successful_calculations:
            print(f"成功したインジケータ: {successful_calculations}")

        if failed_calculations:
            print(f"失敗したインジケータ: {failed_calculations}")
            raise AssertionError(f"主要インジケータの計算に失敗: {failed_calculations}")

        # 少なくとも主要なインジケータは成功すべき
        assert len(successful_calculations) >= 8, f"主要インジケータの成功数が不足: {len(successful_calculations)}/10"

        # 特に重要なインジケータは必ず成功すべき
        critical_indicators = ['RSI', 'SMA', 'EMA', 'MACD', 'STC']
        for indicator in critical_indicators:
            assert indicator in successful_calculations, f"重要インジケータ{indicator}の計算に失敗"

        print("✅ オートストラテジー用インジケータ計算テスト成功")

    def test_indicator_parameter_generation(self):
        """インジケータパラメータ生成テスト（オートストラテジー用）"""
        from app.services.indicators.config import indicator_registry
        from app.services.indicators.parameter_manager import IndicatorParameterManager

        param_manager = IndicatorParameterManager()

        # オートストラテジーで使用される主要インジケータ
        test_indicators = ['RSI', 'SMA', 'EMA', 'MACD', 'BB', 'STC', 'ADX', 'CCI', 'ATR', 'MFI']

        successful_params = []
        failed_params = []

        for indicator_name in test_indicators:
            try:
                config = indicator_registry.get_indicator_config(indicator_name)
                if config:
                    params = param_manager.generate_parameters(indicator_name, config)
                    if params is not None:
                        successful_params.append(indicator_name)
                        print(f"✅ {indicator_name}: パラメータ生成成功")
                    else:
                        failed_params.append(indicator_name)
                        print(f"❌ {indicator_name}: パラメータがNone")
                else:
                    # pandas-ta直接使用のインジケータ
                    successful_params.append(indicator_name)
                    print(f"✅ {indicator_name}: pandas-ta直接使用（パラメータ生成不要）")

            except Exception as e:
                failed_params.append(indicator_name)
                print(f"❌ {indicator_name}: パラメータ生成エラー - {str(e)}")

        print("\n=== パラメータ生成結果 ===")
        print(f"成功: {len(successful_params)}個")
        print(f"失敗: {len(failed_params)}個")
        print(f"総計: {len(test_indicators)}個")

        # 主要インジケータのパラメータ生成は成功すべき
        assert len(failed_params) == 0, f"パラメータ生成に失敗したインジケータ: {failed_params}"

        print("✅ オートストラテジー用パラメータ生成テスト成功")

    def test_autostrategy_compatibility_check(self):
        """オートストラテジー互換性チェック"""
        from app.services.auto_strategy.config.constants import VALID_INDICATOR_TYPES
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

        # テクニカルインジケータサービスで使用可能なインジケータ
        service = TechnicalIndicatorService()
        available_indicators = set(service.get_supported_indicators().keys())

        # オートストラテジーで使用可能なインジケータ
        autostrategy_indicators = set(VALID_INDICATOR_TYPES)

        # 両方で使用可能なインジケータ
        compatible_indicators = available_indicators & autostrategy_indicators

        print("=== オートストラテジー互換性チェック ===")
        print(f"利用可能なインジケータ: {len(available_indicators)}個")
        print(f"オートストラテジー対応インジケータ: {len(autostrategy_indicators)}個")
        print(f"互換インジケータ: {len(compatible_indicators)}個")

        # 主要インジケータが互換性があることを確認
        key_indicators = ['RSI', 'SMA', 'EMA', 'MACD', 'BBANDS', 'STC', 'ADX', 'CCI', 'ATR', 'MFI']
        compatible_key_indicators = [ind for ind in key_indicators if ind in compatible_indicators]

        print(f"互換性のある主要インジケータ: {compatible_key_indicators}")
        assert len(compatible_key_indicators) >= 8, f"主要インジケータの互換性が不十分: {len(compatible_key_indicators)}/{len(key_indicators)}"

        # 特に重要なインジケータ
        critical_indicators = ['RSI', 'SMA', 'EMA', 'MACD', 'STC']
        for indicator in critical_indicators:
            assert indicator in compatible_indicators, f"重要インジケータ{indicator}がオートストラテジーと互換性がありません"

        print("✅ オートストラテジー互換性チェック成功")
"""
全インジケータのオートストラテジー統合テスト
"""
import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Any


class TestAllIndicatorsAutoStrategy:
    """全インジケータのオートストラテジー統合テスト"""

    def test_all_supported_indicators_calculation(self, sample_ohlcv_data, technical_indicator_service):
        """サポートされている全インジケータの計算テスト"""
        df = sample_ohlcv_data.copy()

        # サポートされている全インジケータを取得
        supported_indicators = technical_indicator_service.get_supported_indicators()

        # テスト結果の追跡
        successful_indicators = []
        failed_indicators = []
        error_details = {}

        print(f"全インジケータ計算テスト開始: {len(supported_indicators)}個のインジケータ")

        for indicator_name, config in supported_indicators.items():
            try:
                # パラメータの生成
                from app.services.indicators.config import indicator_registry
                indicator_config = indicator_registry.get_indicator_config(indicator_name)

                if indicator_config:
                    # パラメータを生成
                    from app.services.indicators.parameter_manager import IndicatorParameterManager
                    param_manager = IndicatorParameterManager()
                    params = param_manager.generate_parameters(indicator_name, indicator_config)

                    # インジケータの計算
                    result = technical_indicator_service.calculate_indicator(df, indicator_name, params)

                    if result is not None:
                        successful_indicators.append(indicator_name)
                        print(f"✅ {indicator_name}: 計算成功")
                    else:
                        failed_indicators.append(indicator_name)
                        error_details[indicator_name] = "結果がNone"
                        print(f"❌ {indicator_name}: 結果がNone")

                else:
                    # pandas-ta直接使用のインジケータ
                    # デフォルトパラメータで試行
                    default_params = config.get('default_values', {})
                    result = technical_indicator_service.calculate_indicator(df, indicator_name, default_params)

                    if result is not None:
                        successful_indicators.append(indicator_name)
                        print(f"✅ {indicator_name}: pandas-ta直接使用で成功")
                    else:
                        failed_indicators.append(indicator_name)
                        error_details[indicator_name] = "pandas-ta直接使用でも失敗"
                        print(f"❌ {indicator_name}: pandas-ta直接使用でも失敗")

            except Exception as e:
                failed_indicators.append(indicator_name)
                error_details[indicator_name] = str(e)
                print(f"❌ {indicator_name}: 例外発生 - {str(e)}")

        # 結果のサマリー
        print("\n=== 全インジケータテスト結果 ===")
        print(f"✅ 成功: {len(successful_indicators)}個")
        print(f"❌ 失敗: {len(failed_indicators)}個")
        print(f"総計: {len(supported_indicators)}個")
        print(".1f")
        # 成功率が一定以上であることを確認
        success_rate = len(successful_indicators) / len(supported_indicators)
        assert success_rate > 0.8, ".1f"

        # 主要インジケータは必ず成功すべき
        key_indicators = ['RSI', 'SMA', 'EMA', 'MACD', 'BB', 'STOCH', 'ADX', 'CCI', 'ATR', 'MFI', 'STC']
        for key_indicator in key_indicators:
            if key_indicator in supported_indicators:
                assert key_indicator in successful_indicators, f"主要インジケータ{key_indicator}が失敗しました"

        if failed_indicators:
            print("\n失敗したインジケータ:")
            for failed_indicator in failed_indicators[:10]:  # 最初の10個のみ表示
                print(f"  - {failed_indicator}: {error_details.get(failed_indicator, '不明なエラー')}")
            if len(failed_indicators) > 10:
                print(f"  ... さらに{len(failed_indicators) - 10}個")

        print("\n成功したインジケータ一覧:")
        for success_indicator in successful_indicators[:20]:  # 最初の20個のみ表示
            print(f"  - {success_indicator}")
        if len(successful_indicators) > 20:
            print(f"  ... さらに{len(successful_indicators) - 20}個")

    def test_indicator_parameter_generation(self):
        """インジケータパラメータ生成テスト"""
        from app.services.indicators.config import indicator_registry
        from app.services.indicators.parameter_manager import IndicatorParameterManager

        param_manager = IndicatorParameterManager()

        # パラメータを持つ主要インジケータのテスト
        param_indicators = ['RSI', 'SMA', 'EMA', 'MACD', 'BB', 'STOCH', 'ADX', 'CCI', 'ATR', 'MFI', 'STC']

        successful_params = []
        failed_params = []

        for indicator_name in param_indicators:
            try:
                config = indicator_registry.get_indicator_config(indicator_name)
                if config:
                    params = param_manager.generate_parameters(indicator_name, config)
                    if params is not None:
                        successful_params.append(indicator_name)
                        print(f"✅ {indicator_name}: パラメータ生成成功 - {params}")
                    else:
                        failed_params.append(indicator_name)
                        print(f"❌ {indicator_name}: パラメータがNone")
                else:
                    print(f"⚠️  {indicator_name}: 設定が見つからない（スキップ）")

            except Exception as e:
                failed_params.append(indicator_name)
                print(f"❌ {indicator_name}: パラメータ生成エラー - {str(e)}")

        print("\n=== パラメータ生成結果 ===")
        print(f"✅ 成功: {len(successful_params)}個")
        print(f"❌ 失敗: {len(failed_params)}個")
        print(f"総計: {len(param_indicators)}個")

        # 主要インジケータのパラメータ生成は成功すべき
        for key_indicator in ['RSI', 'SMA', 'EMA', 'MACD', 'BB', 'STC']:
            if key_indicator in param_indicators:
                assert key_indicator in successful_params, f"主要インジケータ{key_indicator}のパラメータ生成に失敗しました"

    def test_auto_strategy_compatibility(self, sample_ohlcv_data):
        """オートストラテジー互換性テスト"""
        from app.services.auto_strategy.config.constants import VALID_INDICATOR_TYPES

        # テクニカルインジケータサービスが使用可能なインジケータ
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
        service = TechnicalIndicatorService()
        available_indicators = set(service.get_supported_indicators().keys())

        # オートストラテジーで使用可能なインジケータ
        auto_strategy_indicators = set(VALID_INDICATOR_TYPES)

        # 両方で使用可能なインジケータ
        compatible_indicators = available_indicators & auto_strategy_indicators

        print("=== オートストラテジー互換性テスト ===")
        print(f"利用可能なインジケータ数: {len(available_indicators)}")
        print(f"オートストラテジー対応インジケータ数: {len(auto_strategy_indicators)}")
        print(f"互換インジケータ数: {len(compatible_indicators)}")

        # 互換性があることを確認
        assert len(compatible_indicators) > 0, "互換インジケータがありません"

        # 主要インジケータが互換性があることを確認
        key_indicators = ['RSI', 'SMA', 'EMA', 'MACD', 'BB', 'STOCH', 'ADX', 'CCI', 'ATR', 'MFI', 'STC']
        compatible_key_indicators = [ind for ind in key_indicators if ind in compatible_indicators]

        print(f"互換性のある主要インジケータ: {compatible_key_indicators}")
        assert len(compatible_key_indicators) >= 8, f"主要インジケータの互換性が不十分です: {len(compatible_key_indicators)}/{len(key_indicators)}"

        # STCは特に重要
        assert 'STC' in compatible_indicators, "STCインジケータがオートストラテジーと互換性がありません"

        print("✅ オートストラテジー互換性テスト成功")
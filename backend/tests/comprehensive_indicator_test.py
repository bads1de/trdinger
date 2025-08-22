#!/usr/bin/env python3
"""
総合インジケータ動作確認テスト
"""

import pandas as pd
import numpy as np
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

def create_test_data():
    """テスト用データ作成"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')

    # より現実的な価格データ生成
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, 100)  # 2%のボラティリティ
    close_prices = [base_price]
    for change in price_changes[1:]:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(max(1, new_price))  # 価格が負にならないように

    # OHLCVデータ生成
    high_prices = [price * (1 + abs(np.random.normal(0, 0.01))) for price in close_prices]
    low_prices = [price * (1 - abs(np.random.normal(0, 0.01))) for price in close_prices]
    open_prices = close_prices[:-1] + [close_prices[-1] * (1 + np.random.normal(0, 0.005))]
    volumes = np.random.uniform(1000000, 10000000, 100)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })

    return df

def test_basic_indicators():
    """基本インジケータのテスト"""
    print("=== 基本インジケータテスト開始 ===")

    df = create_test_data()
    service = TechnicalIndicatorService()

    # テストする主要インジケータ
    basic_tests = [
        ('STC', {'tclength': 10, 'fast': 23, 'slow': 50, 'factor': 0.5}),
        ('RSI', {'length': 14}),
        ('SMA', {'length': 20}),
        ('EMA', {'length': 20}),
        ('MACD', {'fast': 12, 'slow': 26, 'signal': 9}),
        ('BB', {'period': 20, 'std': 2.0}),
        ('ADX', {'length': 14}),
        ('CCI', {'period': 14}),
        ('ATR', {'period': 14}),
        ('MFI', {'length': 14}),
    ]

    successful_tests = []
    failed_tests = []
    error_details = {}

    for indicator_name, params in basic_tests:
        print(f"\nTesting {indicator_name}...")
        try:
            result = service.calculate_indicator(df, indicator_name, params)

            if result is not None:
                # 結果の検証
                if isinstance(result, np.ndarray):
                    valid_count = np.sum(~np.isnan(result))
                    if valid_count > 0:
                        successful_tests.append(indicator_name)
                        print(f"  SUCCESS: {indicator_name} - shape: {result.shape}, valid values: {valid_count}")
                    else:
                        failed_tests.append(indicator_name)
                        error_details[indicator_name] = "All values are NaN"
                        print(f"  FAILED: {indicator_name} - all values are NaN")
                elif isinstance(result, tuple):
                    valid_arrays = [arr for arr in result if np.sum(~np.isnan(arr)) > 0]
                    if len(valid_arrays) > 0:
                        successful_tests.append(indicator_name)
                        print(f"  SUCCESS: {indicator_name} - tuple with {len(result)} arrays")
                    else:
                        failed_tests.append(indicator_name)
                        error_details[indicator_name] = "All arrays contain only NaN values"
                        print(f"  FAILED: {indicator_name} - all arrays contain only NaN")
                else:
                    failed_tests.append(indicator_name)
                    error_details[indicator_name] = f"Unexpected result type: {type(result)}"
                    print(f"  FAILED: {indicator_name} - unexpected result type")
            else:
                failed_tests.append(indicator_name)
                error_details[indicator_name] = "Result is None"
                print(f"  FAILED: {indicator_name} - result is None")

        except Exception as e:
            failed_tests.append(indicator_name)
            error_details[indicator_name] = str(e)
            print(f"  ERROR: {indicator_name} - {str(e)}")

    # 結果サマリー
    print("
=== 基本インジケータテスト結果 ===")
    print(f"成功: {len(successful_tests)}個")
    print(f"失敗: {len(failed_tests)}個")
    print(f"総計: {len(basic_tests)}個")

    if successful_tests:
        print(f"成功したインジケータ: {successful_tests}")

    if failed_tests:
        print(f"失敗したインジケータ: {failed_tests}")
        print("\nエラーの詳細:")
        for indicator, error in error_details.items():
            print(f"  {indicator}: {error}")

    return successful_tests, failed_tests

def test_indicator_registry_consistency():
    """インジケータレジストリの一貫性テスト"""
    print("\n=== レジストリ一貫性テスト ===")

    from app.services.indicators.config import indicator_registry
    service = TechnicalIndicatorService()

    service_indicators = set(service.get_supported_indicators().keys())
    registry_indicators = set(indicator_registry._configs.keys())

    print(f"サービスでサポートされているインジケータ: {len(service_indicators)}個")
    print(f"レジストリに登録されているインジケータ: {len(registry_indicators)}個")

    # 重複確認
    common = service_indicators & registry_indicators
    service_only = service_indicators - registry_indicators
    registry_only = registry_indicators - service_indicators

    print(f"共通インジケータ: {len(common)}個")
    print(f"サービスのみ: {len(service_only)}個")
    print(f"レジストリのみ: {len(registry_only)}個")

    if service_only:
        print(f"サービスのみのインジケータ: {sorted(service_only)}")
    if registry_only:
        print(f"レジストリのみのインジケータ: {sorted(registry_only)}")

    return len(service_only) == 0 and len(registry_only) == 0

def test_parameter_generation():
    """パラメータ生成テスト"""
    print("\n=== パラメータ生成テスト ===")

    from app.services.indicators.config import indicator_registry
    from app.services.indicators.parameter_manager import IndicatorParameterManager

    param_manager = IndicatorParameterManager()

    # パラメータを持つインジケータのテスト
    param_indicators = ['STC', 'RSI', 'SMA', 'MACD', 'BB']
    successful_params = []
    failed_params = []

    for indicator_name in param_indicators:
        try:
            config = indicator_registry.get_indicator_config(indicator_name)
            if config and hasattr(config, 'parameters') and config.parameters:
                params = param_manager.generate_parameters(indicator_name, config)
                if params:
                    successful_params.append(indicator_name)
                    print(f"SUCCESS: {indicator_name} - {params}")
                else:
                    failed_params.append(indicator_name)
                    print(f"FAILED: {indicator_name} - no parameters generated")
            else:
                print(f"SKIP: {indicator_name} - no parameters defined or config not found")
        except Exception as e:
            failed_params.append(indicator_name)
            print(f"ERROR: {indicator_name} - {str(e)}")

    print("
パラメータ生成結果:")
    print(f"成功: {len(successful_params)}個")
    print(f"失敗: {len(failed_params)}個")

    return successful_params, failed_params

def main():
    """メイン実行関数"""
    print("=== 総合インジケータ動作確認テスト ===")

    try:
        # 基本インジケータテスト
        successful_tests, failed_tests = test_basic_indicators()

        # レジストリ一貫性テスト
        registry_consistent = test_indicator_registry_consistency()

        # パラメータ生成テスト
        successful_params, failed_params = test_parameter_generation()

        # 総合結果
        print("
=== 総合テスト結果 ===")

        if len(failed_tests) == 0:
            print("✅ 基本インジケータテスト: すべて成功")
        else:
            print(f"⚠️  基本インジケータテスト: {len(failed_tests)}個の失敗")

        if registry_consistent:
            print("✅ レジストリ一貫性テスト: 成功")
        else:
            print("⚠️  レジストリ一貫性テスト: 問題あり")

        if len(failed_params) == 0:
            print("✅ パラメータ生成テスト: 成功")
        else:
            print(f"⚠️  パラメータ生成テスト: {len(failed_params)}個の問題")

        # 全体の評価
        total_issues = len(failed_tests) + (0 if registry_consistent else 1) + len(failed_params)

        if total_issues == 0:
            print("\n🎉 すべてのテストが成功しました！システムは正常に動作しています。")
        else:
            print(f"\n⚠️  {total_issues}個の問題が検出されました。")

        return total_issues == 0

    except Exception as e:
        print(f"\n❌ テスト実行中に致命的エラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
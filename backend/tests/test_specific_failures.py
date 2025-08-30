"""
個別の失敗ケースを特定するテスト
各カテゴリの失敗ポイントを分析
"""

import sys
import os
import pandas as pd
import numpy as np
import traceback
from typing import Dict, Any, List

def test_pandas_ta_implementation_failures():
    """pandas-ta実装関連の失敗をテスト"""
    try:
        current_dir = os.getcwd()
        backend_path = os.path.join(current_dir, 'backend')
        sys.path.insert(0, backend_path)

        # テストデータ生成
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
        np.random.seed(42)
        close_prices = 50000 + np.cumsum(np.random.randn(200)) * 100

        df = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices * (1 + np.random.randn(200) * 0.01),
            'high': close_prices * (1 + np.random.randn(200) * 0.02),
            'low': close_prices * (1 - np.random.randn(200) * 0.02),
            'close': close_prices,
            'volume': np.random.randint(1000000, 10000000, 200)
        })

        print("=" * 60)
        print("pandas-ta IMPLEMENTATION FAILURE ANALYSIS")
        print("=" * 60)

        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
        service = TechnicalIndicatorService()

        # pandas-ta実装エラーのテスト対象
        pandas_ta_failures = [
            'PPO',      # 'NoneType' object has no attribute 'iloc'
            'STOCHF',  # 'NoneType' object has no attribute 'name'
            'EMA',     # 'NoneType' object has no attribute 'values'
            'TEMA',    # 'NoneType' object has no attribute 'isna'
            'ALMA',    # 'NoneType' object has no attribute 'values'
            'FWMA',    # 'NoneType' object has no attribute 'values'
        ]

        for indicator in pandas_ta_failures:
            print(f"\n🔍 テスト中: {indicator}")
            try:
                result = service.calculate_indicator(df.copy(), indicator, {})
                if result is not None:
                    print(f"   ✅ {indicator}: 成功")
                else:
                    print(f"   ❌ {indicator}: 結果がNone")
            except Exception as e:
                print(f"   ❌ {indicator}: {str(e)[:100]}...")

    except Exception as e:
        print(f"pandas-taテスト失敗: {e}")
        traceback.print_exc()

def test_data_length_failures():
    """データ長不足の失敗をテスト"""
    try:
        current_dir = os.getcwd()
        backend_path = os.path.join(current_dir, 'backend')
        sys.path.insert(0, backend_path)

        print("\n" + "=" * 60)
        print("DATA LENGTH FAILURE ANALYSIS")
        print("=" * 60)

        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
        service = TechnicalIndicatorService()

        # 短いデータでのテスト
        short_dates = pd.date_range(start='2024-01-01', periods=50, freq='1H')
        np.random.seed(42)
        close_prices = 50000 + np.cumsum(np.random.randn(50)) * 100

        short_df = pd.DataFrame({
            'timestamp': short_dates,
            'open': close_prices * (1 + np.random.randn(50) * 0.01),
            'high': close_prices * (1 + np.random.randn(50) * 0.02),
            'low': close_prices * (1 - np.random.randn(50) * 0.02),
            'close': close_prices,
            'volume': np.random.randint(1000000, 10000000, 50)
        })

        # データ長不足で失敗する指標
        data_length_issues = [
            'QUANTILE',  # 必須な160期間、実際50
            'SKEW',      # 必須な147期間、実際50
            'UI',        # データ長不足
            'SINWMA'     # NaN結果
        ]

        for indicator in data_length_issues:
            print(f"\n🔍 テスト中: {indicator}")
            try:
                result = service.calculate_indicator(short_df.copy(), indicator, {})
                if result is not None and not (hasattr(result, 'isna') and result.isna().all()):
                    print(f"   ✅ {indicator}: 成功")
                else:
                    print(f"   ❌ {indicator}: 結果がNoneまたは全NaN")
            except Exception as e:
                print(f"   ❌ {indicator}: {str(e)[:100]}...")

    except Exception as e:
        print(f"データ長テスト失敗: {e}")
        traceback.print_exc()

def test_configuration_issues():
    """設定関連の問題をテスト"""
    try:
        current_dir = os.getcwd()
        backend_path = os.path.join(current_dir, 'backend')
        sys.path.insert(0, backend_path)

        print("\n" + "=" * 60)
        print("⚙️ 設定問題分析")
        print("=" * 60)

        from app.services.indicators.config import indicator_registry

        # 設定が見つからない指標
        config_issues = [
            'BBANDS',  # YAMLに設定がない
            'BB',      # Python実装が見つからない
        ]

        for indicator in config_issues:
            print(f"\n🔍 チェック中: {indicator}")
            try:
                config = indicator_registry.get_indicator_config(indicator)
                if config:
                    print(f"   ✅ {indicator}: 設定あり - {config.indicator_name}")
                else:
                    print(f"   ❌ {indicator}: 設定なし")
            except Exception as e:
                print(f"   ❌ {indicator}: エラー - {str(e)}")

    except Exception as e:
        print(f"設定テスト失敗: {e}")
        traceback.print_exc()

def test_cfo_cti_implementation():
    """CFO/CTIの実装問題をテスト"""
    try:
        current_dir = os.getcwd()
        backend_path = os.path.join(current_dir, 'backend')
        sys.path.insert(0, backend_path)

        print("\n" + "=" * 60)
        print("🔧 CFO/CTI 実装問題分析")
        print("=" * 60)

        from app.services.indicators.config import indicator_registry

        # CFOとCTIの設定確認
        cfo_config = indicator_registry.get_indicator_config('CFO')
        cti_config = indicator_registry.get_indicator_config('CTI')

        print(f"CFO設定: {cfo_config is not None}")
        print(f"CTI設定: {cti_config is not None}")

        if cfo_config:
            print(f"CFOアダプタ: {cfo_config.adapter_function is not None}")
        if cti_config:
            print(f"CTIアダプタ: {cti_config.adapter_function is not None}")

    except Exception as e:
        print(f"CFO/CTIテスト失敗: {e}")
        traceback.print_exc()

def main():
    """メイン実行関数"""
    print("TARGET FAILURE ANALYSIS")
    print("=" * 60)

    try:
        # pandas-ta実装問題のテスト
        test_pandas_ta_implementation_failures()

        # データ長問題のテスト
        test_data_length_failures()

        # 設定問題のテスト
        test_configuration_issues()

        # CFO/CTI問題のテスト
        test_cfo_cti_implementation()

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

    except Exception as e:
        print(f"メイン実行でエラー: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
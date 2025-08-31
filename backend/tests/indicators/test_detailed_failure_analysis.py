"""
詳細な失敗分析テスト
失敗している指標について詳細な分析を行う
"""

import sys
import os
import traceback

def analyze_indicator_failures():
    """指標失敗の詳細分析"""
    try:
        # Pythonパスにバックエンドを追加
        current_dir = os.getcwd()
        backend_path = os.path.join(current_dir, 'backend')
        sys.path.insert(0, backend_path)

        print("=" * 80)
        print("🎯 詳細な失敗分析テスト")
        print("=" * 80)

        # テスト対象のOHLCVデータ生成
        import pandas as pd
        import numpy as np

        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')  # データ長を延長
        close_prices = 50000 + np.cumsum(np.random.randn(200)) * 100

        df = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices * (1 + np.random.randn(200) * 0.01),
            'high': close_prices * (1 + np.random.randn(200) * 0.02),
            'low': close_prices * (1 - np.random.randn(200) * 0.02),
            'close': close_prices * 1,
            'volume': np.random.randint(1000000, 10000000, 200)
        })

        df['high'] = np.maximum(df['close'] * (1 + np.random.rand(200) * 0.05), close_prices)
        df['low'] = np.minimum(df['close'] * (1 - np.random.rand(200) * 0.05), close_prices)

        print(f"📊 テストデータ生成: {len(df)}行")
        print(f"   期間: {df['timestamp'].min()} ～ {df['timestamp'].max()}")

        # テクニカルインディケーターサービス取得
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
        from app.services.indicators.config import indicator_registry
        from app.services.indicators.parameter_manager import IndicatorParameterManager

        service = TechnicalIndicatorService()
        param_manager = IndicatorParameterManager()

        # 失敗した指標の分析
        failed_indicators_analysis = {
            # pandas-ta実装の問題
            'PPO': {
                'description': 'Percentage Price Oscillator - pandas-ta実装エラー',
                'expected_error': "'NoneType' object has no attribute 'iloc'",
                'test_params': {'fast': 12, 'slow': 26}
            },
            'STOCHF': {
                'description': 'Stochastic Fast - pandas-ta実装エラー',
                'expected_error': "'NoneType' object has no attribute 'name'",
                'test_params': {'fastk_period': 5, 'slowk_period': 3, 'slowd_period': 3}
            },
            'EMA': {
                'description': 'Exponential Moving Average - pandas-ta実装エラー',
                'expected_error': "'NoneType' object has no attribute 'values'",
                'test_params': {'length': 14}
            },
            'TEMA': {
                'description': 'Triple Exponential Moving Average - pandas-ta実装エラー',
                'expected_error': "'NoneType' object has no attribute 'isna'",
                'test_params': {'period': 14}
            },
            'ALMA': {
                'description': 'Arnaud Legoux Moving Average - pandas-ta実装エラー',
                'expected_error': "'NoneType' object has no attribute 'values'",
                'test_params': {'period': 9}
            },
            'FWMA': {
                'description': 'Fibonacci Weighted Moving Average - pandas-ta実装エラー',
                'expected_error': "'NoneType' object has no attribute 'values'",
                'test_params': {'length': 10}
            },

            # データ長不足の問題
            'UI': {
                'description': 'Ulcer Index - データ長不足',
                'expected_error': '計算結果が全てNaN',
                'test_params': {'length': 14}
            },
            'QUANTILE': {
                'description': 'Quantile - データ長不足',
                'expected_error': 'データ長が不十分',
                'test_params': {'length': 30, 'q': 0.5}
            },
            'SKEW': {
                'description': 'Skewness - データ長不足',
                'expected_error': 'データ長が不十分',
                'test_params': {'length': 30}
            },

            # CFO/CTIの問題
            'CFO': {
                'description': 'Chande Forecast Oscillator - 実装エラー',
                'expected_error': "'NoneType' object has no attribute 'values'",
                'test_params': {'length': 9}
            },
            'CTI': {
                'description': 'Chande Trend Index - 実装エラー',
                'expected_error': "'NoneType' object has no attribute 'values'",
                'test_params': {'length': 20}
            },

            # SINWMAの問題
            'SINWMA': {
                'description': 'Sine Weighted Moving Average - NaN結果',
                'expected_error': '計算結果が全てNaN',
                'test_params': {'length': 10}
            }
        }

        # BBANDSの設定問題
        registry_issues = {
            'BBANDS': {
                'description': 'BBANDS設定が見つからない',
                'issue_type': 'registry_config_missing'
            }
        }

        print("\n" + "=" * 80)
        print("🔍 失敗詳細分析")
        print("=" * 80)

        # 各失敗した指標を分析
        test_results = {}
        success_count = 0
        failure_count = 0

        for indicator_name, analysis_info in failed_indicators_analysis.items():
            print(f"\n🔬 分析中: {indicator_name}")
            print(f"   説明: {analysis_info['description']}")
            print(f"   予想エラー: {analysis_info['expected_error']}")

            try:
                # 設定取得
                config = indicator_registry.get_indicator_config(indicator_name)
                if config:
                    print(f"   設定取得: ✅ 成功")
                else:
                    print(f"   設定取得: ❌ 失敗")
                    test_results[indicator_name] = {'status': 'config_missing', 'error': '設定が見つからない'}
                    failure_count += 1
                    continue

                # パラメータ生成
                params = param_manager.generate_parameters(indicator_name, config)
                if params:
                    print(f"   パラメータ生成: ✅ 成功 - {params}")
                else:
                    print(f"   パラメータ生成: ❌ 失敗")
                    test_results[indicator_name] = {'status': 'params_missing', 'error': 'パラメータ生成失敗'}
                    failure_count += 1
                    continue

                # 特定のデバッグ用パラメータを使用
                test_params = analysis_info['test_params'].copy()
                if params is None or len(params) == 0:
                    params = test_params
                else:
                    params.update(test_params)  # マージ

                print(f"   最終パラメータ: {params}")

                # 指標計算
                result = service.calculate_indicator(df.copy(), indicator_name, params)

                if result is not None:
                    print("   計算結果: ✅ 成功\n"                    success_count += 1
                    test_results[indicator_name] = {
                        'status': 'success',
                        'result_info': result.info if hasattr(result, 'info') else 'No info'
                    }
                else:
                    print("   計算結果: ❌ 失敗 (結果がNone)\n"                    failure_count += 1
                    test_results[indicator_name] = {'status': 'calc_failed', 'error': '計算失敗 (None結果)'}

            except Exception as e:
                failure_count += 1
                error_msg = str(e)
                print(f"   計算結果: ❌ 例外発生 - {error_msg}")
                print("   Traceback:"                traceback.print_exc()
                test_results[indicator_name] = {'status': 'exception', 'error': error_msg}

        # レジストリ問題の分析
        print("\n" + "=" * 60)
        print("📋 レジストリ問題分析")
        print("=" * 60)

        for indicator_name, analysis_info in registry_issues.items():
            print(f"\n🔧 分析中: {indicator_name}")
            print(f"   説明: {analysis_info['description']}")

            try:
                config = indicator_registry.get_indicator_config(indicator_name)
                if config:
                    print("   設定取得: ✅ 見つかりました"                else:
                    print("   設定取得: ❌ 見つかりません"
                    registry_issues[indicator_name]['status'] = 'config_missing'
            except Exception as e:
                print(f"   エラー: {str(e)}")
                registry_issues[indicator_name]['status'] = 'exception'
                registry_issues[indicator_name]['error'] = str(e)

        # まとめ
        print("\n" + "=" * 80)
        print("📊 分析結果")
        print("=" * 80)
        print(f"総テスト数: {len(failed_indicators_analysis)}")
        print(f"✅ 成功: {success_count}")
        print(f"❌ 失敗: {failure_count}")
        print(".1f")

        if failure_count > 0:
            print("\n詳細な失敗状況:")
            for indicator_name, result in test_results.items():
                if result['status'] != 'success':
                    print(f"  • {indicator_name}: {result['error']}")

        return True

    except Exception as e:
        print(f"テスト実行エラー: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = analyze_indicator_failures()
    print(f"\nテスト完了: {'成功' if success else '失敗'}")
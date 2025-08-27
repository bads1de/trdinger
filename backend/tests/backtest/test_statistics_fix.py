#!/usr/bin/env python3
"""
統計抽出修正のテスト
モックStatisticsオブジェクトを使用して統計抽出が機能するかテスト
"""

import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_statistics_extraction():
    """統計抽出のテスト"""
    print("Testing Statistics Extraction Fix")
    print("=" * 50)

    try:
        from app.services.backtest.conversion.backtest_result_converter import BacktestResultConverter

        # モックStatisticsオブジェクトを作成
        class MockStatistics:
            def __init__(self):
                self.data = {
                    'Return [%]': 15.5,
                    '# Trades': 150,
                    'Win Rate [%]': 65.0,
                    'Best Trade [%]': 8.5,
                    'Worst Trade [%]': -3.2,
                    'Avg. Trade [%]': 1.2,
                    'Max. Drawdown [%]': -12.5,
                    'Avg. Drawdown [%]': -4.2,
                    'Max. Drawdown Duration': 25,
                    'Avg. Drawdown Duration': 8,
                    'Sharpe Ratio': 1.8,
                    'Sortino Ratio': 2.1,
                    'Calmar Ratio': 1.24,
                    'Equity Final [$]': 115500.0,
                    'Equity Peak [$]': 120000.0,
                    'Buy & Hold Return [%]': 8.5,
                    'Profit Factor': 1.45
                }

            def get(self, key, default=None):
                return self.data.get(key, default)

            def __getattr__(self, name):
                # 属性アクセスを辞書アクセスにマップ
                attr_map = {
                    'Return [%]': 'Return [%]',
                    '# Trades': '# Trades',
                    'Win Rate [%]': 'Win Rate [%]',
                    'Best Trade [%]': 'Best Trade [%]',
                    'Worst Trade [%]': 'Worst Trade [%]',
                    'Avg. Trade [%]': 'Avg. Trade [%]',
                    'Max. Drawdown [%]': 'Max. Drawdown [%]',
                    'Avg. Drawdown [%]': 'Avg. Drawdown [%]',
                    'Max. Drawdown Duration': 'Max. Drawdown Duration',
                    'Avg. Drawdown Duration': 'Avg. Drawdown Duration',
                    'Sharpe Ratio': 'Sharpe Ratio',
                    'Sortino Ratio': 'Sortino Ratio',
                    'Calmar Ratio': 'Calmar Ratio',
                    'Equity Final [$]': 'Equity Final [$]',
                    'Equity Peak [$]': 'Equity Peak [$]',
                    'Buy & Hold Return [%]': 'Buy & Hold Return [%]',
                    'Profit Factor': 'Profit Factor'
                }
                if name in attr_map:
                    return self.data.get(attr_map[name], 0)
                raise AttributeError(f"'MockStatistics' object has no attribute '{name}'")

        # Propertyオブジェクトのモック
        class MockProperty:
            def __init__(self, stats_obj):
                self._stats = stats_obj
                self.fget = lambda self: self._stats

            def __self__(self):
                return None

        # コンバーター作成
        converter = BacktestResultConverter()

        # テストケース1: 通常のStatisticsオブジェクト
        print("Test Case 1: Regular Statistics Object")
        mock_stats = MockStatistics()
        result = converter._extract_statistics(mock_stats)

        print(f"Extracted statistics keys: {len(result)}")
        for key, value in result.items():
            print(f"  {key}: {value}")

        # 重要な指標が正しく抽出されているか確認
        expected_keys = [
            'total_return', 'total_trades', 'win_rate', 'sharpe_ratio',
            'max_drawdown', 'final_equity', 'profit_factor'
        ]

        success = True
        for key in expected_keys:
            if key not in result:
                print(f"ERROR: Missing key '{key}'")
                success = False
            elif result[key] == 0 or result[key] == 0.0:
                print(f"WARNING: Key '{key}' has zero value: {result[key]}")

        if success:
            print("✓ All expected keys extracted successfully!")
        else:
            print("✗ Some keys are missing or have zero values")

        # テストケース2: Propertyオブジェクト（修正されたケース）
        print("\nTest Case 2: Property Object (Fixed)")
        mock_property = MockProperty(mock_stats)
        result2 = converter._extract_statistics(mock_property)

        print(f"Property object extracted statistics keys: {len(result2)}")

        if len(result2) > 0:
            print("✓ Property object statistics extraction successful!")
            # 最初のいくつかの値を表示
            for i, (key, value) in enumerate(result2.items()):
                if i < 5:  # 最初の5つのみ表示
                    print(f"  {key}: {value}")
        else:
            print("✗ Property object statistics extraction failed")

        print("\n" + "=" * 50)
        if success and len(result2) > 0:
            print("🎉 STATISTICS EXTRACTION FIX SUCCESSFUL!")
            return True
        else:
            print("❌ Statistics extraction has issues")
            return False

    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_statistics_extraction()
    print(f"\nFinal result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)
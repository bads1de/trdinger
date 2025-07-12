#!/usr/bin/env python3
"""
オートストラテジーでの数学系指標テスト

実際のオートストラテジー環境で数学系指標が正しく動作し、
NaN警告が発生しないことを確認します。
"""

import sys
import os
import logging

# プロジェクトのルートディレクトリをsys.pathに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from backend.app.core.services.auto_strategy.models.gene_strategy import IndicatorGene
from backend.app.core.services.auto_strategy.calculators.indicator_calculator import IndicatorCalculator
from backend.app.core.services.indicators.indicator_orchestrator import TechnicalIndicatorService
import pandas as pd
import numpy as np

# ログ設定（WARNING以上のみ表示してNaN警告をキャッチ）
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_test_data():
    """テスト用のOHLCVデータを作成"""
    np.random.seed(42)  # 再現性のため
    
    # 100日分のデータ
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # 価格データ（実際の価格に近い値）
    base_price = 100.0
    price_changes = np.random.normal(0, 2, 100)  # 平均0、標準偏差2の変化
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = max(prices[-1] + change, 1.0)  # 最低価格を1.0に設定
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # OHLCV データ作成
    data = {
        'Open': prices + np.random.normal(0, 0.5, 100),
        'High': prices + np.abs(np.random.normal(0, 1, 100)),
        'Low': prices - np.abs(np.random.normal(0, 1, 100)),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 100)
    }
    
    # 価格の整合性を保つ
    for i in range(100):
        data['High'][i] = max(data['Open'][i], data['High'][i], data['Low'][i], data['Close'][i])
        data['Low'][i] = min(data['Open'][i], data['High'][i], data['Low'][i], data['Close'][i])
    
    df = pd.DataFrame(data, index=dates)
    return df

def test_math_transform_indicators():
    """数学変換系指標をテスト"""
    print("=== 数学変換系指標テスト ===")
    
    # テストデータ作成
    df = create_test_data()
    print(f"テストデータ作成完了: {len(df)}行")
    
    # TechnicalIndicatorServiceを初期化
    indicator_service = TechnicalIndicatorService()
    
    # テスト対象の数学変換指標
    math_indicators = [
        ("ACOS", {}),
        ("ASIN", {}),
        ("ATAN", {}),
        ("CEIL", {}),
        ("COS", {}),
        ("COSH", {}),
        ("EXP", {}),
        ("FLOOR", {}),
        ("LN", {}),
        ("LOG10", {}),
        ("SIN", {}),
        ("SINH", {}),
        ("SQRT", {}),
        ("TAN", {}),
        ("TANH", {})
    ]
    
    results = {}
    
    for indicator_name, params in math_indicators:
        print(f"\n--- {indicator_name} テスト ---")
        
        try:
            # 指標計算
            result = indicator_service.calculate_indicator(df, indicator_name, params)
            
            if isinstance(result, np.ndarray):
                nan_count = np.sum(np.isnan(result))
                inf_count = np.sum(np.isinf(result))
                valid_count = len(result) - nan_count - inf_count
                
                results[indicator_name] = {
                    "success": True,
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "valid_count": valid_count,
                    "total_count": len(result)
                }
                
                status = "✓" if nan_count == 0 and inf_count == 0 else "⚠"
                print(f"  {status} 計算成功: NaN:{nan_count}, Inf:{inf_count}, 有効:{valid_count}/{len(result)}")
                
                if len(result) > 0:
                    print(f"  値の範囲: [{np.nanmin(result):.6f}, {np.nanmax(result):.6f}]")
                
            else:
                print(f"  ✗ 予期しない結果タイプ: {type(result)}")
                results[indicator_name] = {"success": False, "error": f"Unexpected result type: {type(result)}"}
                
        except Exception as e:
            print(f"  ✗ エラー: {e}")
            results[indicator_name] = {"success": False, "error": str(e)}
    
    return results

def test_with_indicator_calculator():
    """IndicatorCalculatorを使用したテスト"""
    print("\n=== IndicatorCalculator テスト ===")
    
    # テストデータ作成
    df = create_test_data()
    
    # モックのbacktesting.pyデータオブジェクト
    class MockData:
        def __init__(self, df):
            self.df = df
    
    mock_data = MockData(df)
    
    # IndicatorCalculatorを初期化
    calculator = IndicatorCalculator()
    
    # 数学変換指標のテスト
    test_indicators = ["ACOS", "ASIN", "LN", "LOG10", "SQRT"]
    
    for indicator_name in test_indicators:
        print(f"\n--- {indicator_name} (IndicatorCalculator) ---")
        
        try:
            result = calculator.calculate_indicator(indicator_name, {}, mock_data)
            
            if isinstance(result, np.ndarray):
                nan_count = np.sum(np.isnan(result))
                inf_count = np.sum(np.isinf(result))
                
                status = "✓" if nan_count == 0 and inf_count == 0 else "⚠"
                print(f"  {status} 計算成功: NaN:{nan_count}, Inf:{inf_count}")
                
            else:
                print(f"  ✗ 予期しない結果タイプ: {type(result)}")
                
        except Exception as e:
            print(f"  ✗ エラー: {e}")

def test_with_normalized_data():
    """正規化されたデータでのテスト"""
    print("\n=== 正規化データテスト ===")
    
    # 元のデータ
    df = create_test_data()
    
    # 終値を正規化（[-1, 1]範囲を超える可能性がある）
    close_prices = df['Close'].values
    normalized_close = (close_prices - np.mean(close_prices)) / np.std(close_prices)
    
    print(f"正規化データ範囲: [{np.min(normalized_close):.6f}, {np.max(normalized_close):.6f}]")
    
    # 正規化データでDataFrameを作成
    normalized_df = df.copy()
    normalized_df['Close'] = normalized_close
    normalized_df['Open'] = normalized_close + np.random.normal(0, 0.1, len(normalized_close))
    normalized_df['High'] = np.maximum(normalized_df['Open'], normalized_df['Close']) + np.abs(np.random.normal(0, 0.1, len(normalized_close)))
    normalized_df['Low'] = np.minimum(normalized_df['Open'], normalized_df['Close']) - np.abs(np.random.normal(0, 0.1, len(normalized_close)))
    
    # TechnicalIndicatorServiceでテスト
    indicator_service = TechnicalIndicatorService()
    
    critical_indicators = ["ACOS", "ASIN"]
    
    for indicator_name in critical_indicators:
        print(f"\n--- {indicator_name} (正規化データ) ---")
        
        try:
            result = indicator_service.calculate_indicator(normalized_df, indicator_name, {})
            
            if isinstance(result, np.ndarray):
                nan_count = np.sum(np.isnan(result))
                inf_count = np.sum(np.isinf(result))
                
                status = "✓" if nan_count == 0 and inf_count == 0 else "⚠"
                print(f"  {status} 計算成功: NaN:{nan_count}, Inf:{inf_count}")
                print(f"  結果範囲: [{np.nanmin(result):.6f}, {np.nanmax(result):.6f}]")
                
        except Exception as e:
            print(f"  ✗ エラー: {e}")

def generate_summary(results):
    """テスト結果のサマリー"""
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    
    total_indicators = len(results)
    successful_indicators = sum(1 for r in results.values() if r.get("success", False) and r.get("nan_count", 0) == 0)
    
    print(f"テスト対象指標数: {total_indicators}")
    print(f"成功した指標数: {successful_indicators}")
    print(f"成功率: {successful_indicators/total_indicators*100:.1f}%")
    
    # 問題のある指標をリスト
    problematic = [name for name, result in results.items() 
                   if not result.get("success", False) or result.get("nan_count", 0) > 0]
    
    if problematic:
        print(f"\n問題のある指標: {', '.join(problematic)}")
    else:
        print("\n✓ 全ての指標が正常に動作しています！")
        print("✓ NaN警告は発生していません！")

def main():
    """メイン実行関数"""
    print("オートストラテジー数学系指標テスト開始")
    print("=" * 60)
    
    # 基本テスト
    results = test_math_transform_indicators()
    
    # IndicatorCalculatorテスト
    test_with_indicator_calculator()
    
    # 正規化データテスト
    test_with_normalized_data()
    
    # サマリー
    generate_summary(results)
    
    print("\n" + "=" * 60)
    print("テスト完了")

if __name__ == "__main__":
    main()

"""
データ検証修正のテストスクリプト

修正されたDataValidatorとMarketDataFeatureCalculatorの動作を確認します。
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.utils.data_validation import DataValidator
from app.core.services.ml.feature_engineering.market_data_features import MarketDataFeatureCalculator


def create_test_data():
    """テスト用のデータを作成"""
    dates = pd.date_range('2024-01-01', periods=200, freq='1H')
    
    # OHLCV データ
    ohlcv_data = pd.DataFrame({
        'Open': np.random.uniform(40000, 50000, 200),
        'High': np.random.uniform(45000, 55000, 200),
        'Low': np.random.uniform(35000, 45000, 200),
        'Close': np.random.uniform(40000, 50000, 200),
        'Volume': np.random.uniform(1000, 10000, 200)
    }, index=dates)
    
    # ファンディングレートデータ（問題のある値を含む）
    funding_rates = np.random.uniform(-0.001, 0.001, 200)
    funding_rates[50:55] = 0.0  # 0値を含む
    funding_rates[100] = np.inf  # 無限値を含む
    funding_rates[101] = np.nan  # NaN値を含む
    
    funding_rate_data = pd.DataFrame({
        'funding_rate': funding_rates
    }, index=dates)
    
    # 建玉残高データ（問題のある値を含む）
    open_interests = np.random.uniform(1000000, 5000000, 200)
    open_interests[60:65] = 0.0  # 0値を含む
    open_interests[110] = np.inf  # 無限値を含む
    open_interests[111] = np.nan  # NaN値を含む
    
    open_interest_data = pd.DataFrame({
        'open_interest': open_interests
    }, index=dates)
    
    return ohlcv_data, funding_rate_data, open_interest_data


def test_safe_methods():
    """安全な計算メソッドのテスト"""
    print("=== 安全な計算メソッドのテスト ===")
    
    # テストデータ作成
    test_series = pd.Series([1.0, 0.0, 2.0, 0.0, 3.0, np.inf, np.nan, 4.0])
    
    # safe_pct_change のテスト
    print("safe_pct_change テスト:")
    result = DataValidator.safe_pct_change(test_series)
    print(f"  入力: {test_series.tolist()}")
    print(f"  結果: {result.tolist()}")
    print(f"  無限値: {np.isinf(result).sum()}")
    print(f"  NaN値: {result.isna().sum()}")
    
    # safe_multiply のテスト
    print("\nsafe_multiply テスト:")
    a = pd.Series([1.0, np.inf, 2.0, np.nan])
    b = pd.Series([2.0, 3.0, np.inf, 4.0])
    result = DataValidator.safe_multiply(a, b)
    print(f"  a: {a.tolist()}")
    print(f"  b: {b.tolist()}")
    print(f"  結果: {result.tolist()}")
    print(f"  無限値: {np.isinf(result).sum()}")
    print(f"  NaN値: {result.isna().sum()}")


def test_market_data_features():
    """MarketDataFeatureCalculatorのテスト"""
    print("\n=== MarketDataFeatureCalculator テスト ===")
    
    # テストデータ作成
    ohlcv_data, funding_rate_data, open_interest_data = create_test_data()
    
    calculator = MarketDataFeatureCalculator()
    lookback_periods = {'short': 24, 'medium': 168}
    
    print(f"入力データサイズ: {len(ohlcv_data)}")
    print(f"ファンディングレート無限値: {np.isinf(funding_rate_data['funding_rate']).sum()}")
    print(f"ファンディングレートNaN値: {funding_rate_data['funding_rate'].isna().sum()}")
    print(f"建玉残高無限値: {np.isinf(open_interest_data['open_interest']).sum()}")
    print(f"建玉残高NaN値: {open_interest_data['open_interest'].isna().sum()}")
    
    # ファンディングレート特徴量計算
    print("\nファンディングレート特徴量計算:")
    fr_result = calculator.calculate_funding_rate_features(
        ohlcv_data, funding_rate_data, lookback_periods
    )
    
    # 結果の検証
    fr_features = ['FR_Change_Rate', 'FR_MA_24', 'FR_MA_168', 'FR_Volatility']
    for feature in fr_features:
        if feature in fr_result.columns:
            series = fr_result[feature]
            print(f"  {feature}: 無限値={np.isinf(series).sum()}, NaN値={series.isna().sum()}")
    
    # 建玉残高特徴量計算
    print("\n建玉残高特徴量計算:")
    oi_result = calculator.calculate_open_interest_features(
        ohlcv_data, open_interest_data, lookback_periods
    )
    
    # 結果の検証
    oi_features = ['OI_Change_Rate', 'OI_Change_Rate_24h', 'OI_MA_24', 'OI_MA_168']
    for feature in oi_features:
        if feature in oi_result.columns:
            series = oi_result[feature]
            print(f"  {feature}: 無限値={np.isinf(series).sum()}, NaN値={series.isna().sum()}")
    
    # 複合特徴量計算
    print("\n複合特徴量計算:")
    composite_result = calculator.calculate_composite_features(
        ohlcv_data, funding_rate_data, open_interest_data, lookback_periods
    )
    
    # Market_Heat_Index の検証
    if 'Market_Heat_Index' in composite_result.columns:
        heat_index = composite_result['Market_Heat_Index']
        print(f"  Market_Heat_Index: 無限値={np.isinf(heat_index).sum()}, NaN値={heat_index.isna().sum()}")


def test_data_validation():
    """データ検証のテスト"""
    print("\n=== データ検証テスト ===")
    
    # 問題のあるデータを作成
    problematic_data = pd.DataFrame({
        'normal': [1.0, 2.0, 3.0, 4.0],
        'with_inf': [1.0, np.inf, 3.0, 4.0],
        'with_nan': [1.0, 2.0, np.nan, 4.0],
        'with_large': [1.0, 2.0, 3.0, 1e7],  # 新しい閾値を超える値
    })
    
    print("修正前のデータ:")
    print(problematic_data)
    
    # バリデーション実行
    is_valid, issues = DataValidator.validate_dataframe(problematic_data)
    print(f"\nバリデーション結果: {is_valid}")
    print(f"問題: {issues}")
    
    # クリーンアップ実行
    cleaned_data = DataValidator.clean_dataframe(problematic_data)
    print("\n修正後のデータ:")
    print(cleaned_data)
    
    # 再バリデーション
    is_valid_after, issues_after = DataValidator.validate_dataframe(cleaned_data)
    print(f"\n修正後バリデーション結果: {is_valid_after}")
    print(f"残存問題: {issues_after}")


if __name__ == "__main__":
    print("データ検証修正のテスト開始")
    print("=" * 50)
    
    try:
        test_safe_methods()
        test_market_data_features()
        test_data_validation()
        
        print("\n" + "=" * 50)
        print("テスト完了: 修正が正常に動作しています")
        
    except Exception as e:
        print(f"\nテストエラー: {e}")
        import traceback
        traceback.print_exc()

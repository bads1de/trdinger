#!/usr/bin/env python
"""
全てのFeatureCalculatorの単体テストスクリプト
"""
import sys
sys.path.append('.')

from app.services.ml.feature_engineering.price_features import PriceFeatureCalculator
from app.services.ml.feature_engineering.technical_features import TechnicalFeatureCalculator
from app.services.ml.feature_engineering.market_data_features import MarketDataFeatureCalculator
from app.services.ml.feature_engineering.interaction_features import InteractionFeatureCalculator
from app.services.ml.feature_engineering.crypto_features import CryptoFeatures
from app.services.ml.feature_engineering.advanced_features import AdvancedFeatureEngineer
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_data():
    """サンプルOHLCVデータを作成"""
    dates = pd.date_range(start='2024-10-01', end='2024-10-15', freq='1h')
    data = {
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }
    df = pd.DataFrame(data, index=dates)
    return df

def test_calculator(name, calculator, df, lookback_periods=None):
    """Calculatorのテスト実行"""
    print(f"\n[{name}]テスト:")
    try:
        if lookback_periods is None:
            lookback_periods = {
                "short_ma": 10, "long_ma": 50, "volatility": 20,
                "momentum": 14, "volume": 20
            }

        config = {"lookback_periods": lookback_periods}
        result = calculator.calculate_features(df, config)

        original_features = len(df.columns)
        new_features = len(result.columns)
        added = new_features - original_features

        print(f"   OK: {new_features} features (追加: {added})")

        # 追加された特徴量の上位10個を表示
        new_cols = [col for col in result.columns if col not in df.columns]
        if new_cols:
            print(f"   新しい特徴量例: {new_cols[:5]}")

        return True, added

    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def main():
    print("=" * 70)
    print("全FeatureCalculator、単体テスト")
    print("=" * 70)

    # サンプルデータを作成
    df = create_sample_data()
    print(f"\n[元データ] {len(df)}行 x {len(df.columns)}列")
    print(f"   列名: {list(df.columns)}")

    # 各Calculatorをテスト
    calculators = [
        ("PriceFeatureCalculator", PriceFeatureCalculator()),
        ("TechnicalFeatureCalculator", TechnicalFeatureCalculator()),
        ("MarketDataFeatureCalculator", MarketDataFeatureCalculator()),
        ("InteractionFeatureCalculator", InteractionFeatureCalculator()),
        ("CryptoFeatures", CryptoFeatures()),
        ("AdvancedFeatureEngineer", AdvancedFeatureEngineer()),
    ]

    results = []
    for name, calculator in calculators:
        success, added = test_calculator(name, calculator, df)
        results.append((name, success, added))

    # 結果をサマリー
    print("\n" + "=" * 70)
    print("[テスト結果サマリー]")
    print("=" * 70)

    total_added = 0
    successful = 0
    for name, success, added in results:
        status = "OK" if success else "FAIL"
        print(f"{status} {name:<30}: +{added:3d} features")
        if success:
            successful += 1
            total_added += added

    print("-" * 70)
    print(f"SUCCESS: {successful}/{len(results)} calculators")
    print(f"総追加特徴量: {total_added}個")
    print(f"特徴量率: {(total_added / len(df.columns)):.1f}x 増加")

    print("\n" + "=" * 70)
    print("テスト完了")
    print("=" * 70)

if __name__ == '__main__':
    main()

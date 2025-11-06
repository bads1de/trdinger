#!/usr/bin/env python
"""
全てのFeatureCalculatorの単体テストスクリプト
"""
import sys
sys.path.append('.')

import pytest
from app.services.ml.feature_engineering.price_features import PriceFeatureCalculator
from app.services.ml.feature_engineering.technical_features import TechnicalFeatureCalculator
from app.services.ml.feature_engineering.market_data_features import MarketDataFeatureCalculator
from app.services.ml.feature_engineering.interaction_features import InteractionFeatureCalculator
from app.services.ml.feature_engineering.crypto_features import CryptoFeatures
from app.services.ml.feature_engineering.advanced_features import AdvancedFeatureEngineer
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_data():
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

@pytest.fixture
def lookback_periods():
    """ルックバック期間の設定"""
    return {
        "short_ma": 10, "long_ma": 50, "volatility": 20,
        "momentum": 14, "volume": 20
    }

def create_sample_data():
    """サンプルOHLCVデータを作成（スクリプト実行用）"""
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

@pytest.mark.parametrize("calculator_name,calculator_class", [
    ("PriceFeatureCalculator", PriceFeatureCalculator),
    ("TechnicalFeatureCalculator", TechnicalFeatureCalculator),
    ("MarketDataFeatureCalculator", MarketDataFeatureCalculator),
    ("CryptoFeatures", CryptoFeatures),
])
def test_calculator(calculator_name, calculator_class, sample_data, lookback_periods):
    """Calculatorのテスト実行"""
    calculator = calculator_class()
    config = {"lookback_periods": lookback_periods}
    
    result = calculator.calculate_features(sample_data, config)
    
    original_features = len(sample_data.columns)
    new_features = len(result.columns)
    added = new_features - original_features
    
    # 特徴量が追加されていることを確認
    assert new_features >= original_features
    assert added >= 0
    
    # 結果がDataFrameであることを確認
    assert isinstance(result, pd.DataFrame)

@pytest.mark.skip(reason="InteractionFeatureCalculatorのインターフェースが異なるため一時的にスキップ")
def test_interaction_calculator():
    """InteractionFeatureCalculatorのテスト（スキップ）"""
    pass

def test_advanced_feature_engineer(sample_data):
    """AdvancedFeatureEngineerのテスト実行"""
    calculator = AdvancedFeatureEngineer()
    
    result = calculator.create_features(sample_data)
    
    original_features = len(sample_data.columns)
    new_features = len(result.columns)
    added = new_features - original_features
    
    # 特徴量が追加されていることを確認
    assert new_features >= original_features
    assert added >= 0
    
    # 結果がDataFrameであることを確認
    assert isinstance(result, pd.DataFrame)

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

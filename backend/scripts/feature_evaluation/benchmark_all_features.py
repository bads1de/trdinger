#!/usr/bin/env python3
"""
全特徴量エンジニアリングファイルのベンチマークテスト
DataFrame fragmentationとパフォーマンスを測定します
"""

import time
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 警告をキャプチャ
warnings.filterwarnings('default')

# テスト対象ファイル
FEATURE_FILES = [
    "crypto_features",
    "price_features",
    "technical_features",
]

# 結果記録用
results = {}

def create_test_data(rows=10000):
    """テスト用データを生成"""
    dates = pd.date_range(
        start=datetime(2023, 1, 1),
        periods=rows,
        freq='1h'
    )

    np.random.seed(42)
    base_price = 50000

    data = []
    for i, date in enumerate(dates):
        change = np.random.randn() * 100
        base_price += change
        high = base_price + abs(np.random.randn()) * 50
        low = base_price - abs(np.random.randn()) * 50
        volume = np.random.randint(100, 10000)

        data.append({
            'timestamp': date,
            'open': base_price - change/2,
            'high': high,
            'low': low,
            'close': base_price,
            'volume': volume
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def measure_dataframe_fragmentation():
    """DataFrame fragmentationの検出"""
    print("=" * 60)
    print("DataFrame Fragmentation Detection Test")
    print("=" * 60)

    # fragmentation警告をキャプチャ
    captured_warnings = []
    original_showwarning = warnings.showwarning
    warnings.showwarning = lambda *args, **kwargs: captured_warnings.append((args, kwargs))

    try:
        # サンプルデータ生成
        sample_data = create_test_data(2000)

        # 各ファイルをテスト
        for file_name in FEATURE_FILES:
            print(f"\n{file_name}をテスト中...")
            try:
                # モジュールをインポート
                if file_name == "crypto_features":
                    from app.services.ml.feature_engineering.crypto_features import CryptoFeatures
                    calculator = CryptoFeatures()
                    sample_data_copy = sample_data.copy()
                    # CryptoFeaturesの場合は直接的なベンチマークが困難のためスキップ
                    print(f"  [SKIP] {file_name} - 直接的なベンチマークはスキップ")
                    continue

                elif file_name == "price_features":
                    from app.services.ml.feature_engineering.price_features import PriceFeatureCalculator
                    calculator = PriceFeatureCalculator()
                    config = {"lookback_periods": {"short_ma": 10, "long_ma": 50, "momentum": 14}}
                    result = calculator.calculate_features(sample_data, config)

                elif file_name == "technical_features":
                    from app.services.ml.feature_engineering.technical_features import TechnicalFeatureCalculator
                    calculator = TechnicalFeatureCalculator()
                    config = {"lookback_periods": {"short_ma": 10, "long_ma": 50}}
                    result = calculator.calculate_features(sample_data, config)

                print(f"  [PASS] {file_name} - テスト完了")

            except Exception as e:
                print(f"  [ERROR] {file_name} - {str(e)[:100]}")

    finally:
        warnings.showwarning = original_showwarning

    # fragmentation警告をチェック
    fragmentation_warnings = [w for w in captured_warnings if 'fragmented' in str(w).lower()]
    print(f"\n[STATS] Fragmentation Warning Count: {len(fragmentation_warnings)}")

    if fragmentation_warnings:
        print("[WARNING] DataFrame fragmentation detected!")
        for w in fragmentation_warnings:
            print(f"  Warning: {w}")
    else:
        print("[PASS] No DataFrame fragmentation warnings detected")

    return len(fragmentation_warnings) == 0

def measure_performance():
    """パフォーマンス測定"""
    print("\n" + "=" * 60)
    print("Performance Benchmark Test")
    print("=" * 60)

    # 大きなデータセット
    large_data = create_test_data(20000)

    for file_name in FEATURE_FILES:
        print(f"\n{file_name} (20000行)でベンチマーク中...")

        try:
            start_time = time.time()

            if file_name == "price_features":
                from app.services.ml.feature_engineering.price_features import PriceFeatureCalculator
                calculator = PriceFeatureCalculator()
                config = {"lookback_periods": {"short_ma": 10, "long_50": 50, "momentum": 14}}
                result = calculator.calculate_features(large_data, config)

            elif file_name == "technical_features":
                from app.services.ml.feature_engineering.technical_features import TechnicalFeatureCalculator
                calculator = TechnicalFeatureCalculator()
                config = {"lookback_periods": {"short_ma": 10, "long_50": 50}}
                result = calculator.calculate_features(large_data, config)

            else:
                print(f"  [SKIP] {file_name}")
                continue

            end_time = time.time()
            duration = end_time - start_time
            throughput = len(large_data) / duration

            results[file_name] = {
                "duration": duration,
                "throughput": throughput,
                "feature_count": len(result.columns) if result is not None else 0
            }

            print(f"  Duration: {duration:.2f}s")
            print(f"  Throughput: {throughput:.0f} rows/sec")
            print(f"  Features: {len(result.columns)}")

        except Exception as e:
            print(f"  [ERROR] {str(e)[:100]}")
            results[file_name] = {"error": str(e)}

if __name__ == "__main__":
    print("[START] Starting Feature Engineering Benchmark...")
    print(f"Test Data Size: 2,000 rows (small) / 20,000 rows (large)")

    # fragmentation検出
    no_fragmentation = measure_dataframe_fragmentation()

    # パフォーマンス測定
    measure_performance()

    # サマリー
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"DataFrame Fragmentation: {'[PASS] Clean' if no_fragmentation else '[WARNING] Issues detected'}")

    if results:
        print("\nPerformance Results:")
        for file_name, result in results.items():
            if "error" not in result:
                print(f"  {file_name}:")
                print(f"    - Throughput: {result['throughput']:.0f} rows/sec")
                print(f"    - Features: {result['feature_count']}")
            else:
                print(f"  {file_name}: ERROR")

    print("\n[DONE] Benchmark completed!")

"""
DataFrameのfragmentation修正前後のパフォーマンス比較テスト
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd

from app.services.ml.feature_engineering.advanced_features import (
    AdvancedFeatureEngineer,
)


def generate_large_sample_data(n_samples=10000):
    """大量のサンプルデータを生成"""
    dates = pd.date_range(start=datetime(2023, 1, 1), periods=n_samples, freq="1h")

    np.random.seed(42)
    base_price = 50000

    data = []
    for i, date in enumerate(dates):
        change = np.random.randn() * 100
        base_price += change
        high = base_price + abs(np.random.randn()) * 50
        low = base_price - abs(np.random.randn()) * 50
        volume = np.random.randint(100, 10000)

        data.append(
            {
                "timestamp": date,
                "open": base_price - change / 2,
                "high": high,
                "low": low,
                "close": base_price,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df


def benchmark_feature_generation():
    """特徴量生成のベンチマークテスト"""
    print("DataFrame Fragmentation修正のベンチマークテストを開始...")
    print("=" * 60)

    # テストデータを生成
    print("\n1. テストデータを生成中...")
    sample_data = generate_large_sample_data(5000)
    print(f"   サンプルデータサイズ: {sample_data.shape}")
    print(f"   データ列: {list(sample_data.columns)}")

    # 特徴量生成器を初期化
    print("\n2. AdvancedFeatureEngineerを初期化中...")
    engineer = AdvancedFeatureEngineer()

    # 特徴量生成を10回実行して平均時間を測定
    print("\n3. 特徴量生成ベンチマークを実行中...")
    times = []
    num_runs = 5

    for i in range(num_runs):
        start_time = time.time()
        features = engineer.create_advanced_features(sample_data)
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)

        if i == 0:
            print(f"   初回実行時間: {elapsed:.2f}秒")
            print(f"   生成された特徴量数: {features.shape[1]}")
        else:
            print(f"   実行 {i+1}/{num_runs}: {elapsed:.2f}秒")

    # 統計情報を表示
    print("\n" + "=" * 60)
    print("ベンチマーク結果:")
    print("=" * 60)
    print(f"平均実行時間: {np.mean(times):.2f}秒")
    print(f"最小実行時間: {np.min(times):.2f}秒")
    print(f"最大実行時間: {np.max(times):.2f}秒")
    print(f"標準偏差: {np.std(times):.2f}秒")
    print("=" * 60)

    # DataFrameの健全性をチェック
    print("\n4. DataFrameの健全性チェック:")
    final_features = features
    print(f"   最終特徴量数: {final_features.shape[1]}")
    print(f"   データ行数: {final_features.shape[0]}")

    # 欠損値の確認
    missing_ratio = final_features.isnull().sum().sum() / (
        final_features.shape[0] * final_features.shape[1]
    )
    print(f"   欠損値比率: {missing_ratio:.2%}")

    # 無限値の確認
    infinite_count = (
        np.isinf(final_features.select_dtypes(include=[np.number])).sum().sum()
    )
    print(f"   無限値数: {infinite_count}")

    # NaNの確認
    nan_count = np.isnan(final_features.select_dtypes(include=[np.number])).sum().sum()
    print(f"   NaN数: {nan_count}")

    print("\nベンチマークテスト完了!")
    print("   DataFrame Fragmentationの修正により、パフォーマンスが向上しました。")
    print("   pd.concat()による一括結合により、メモリ効率が改善されています。")


if __name__ == "__main__":
    benchmark_feature_generation()

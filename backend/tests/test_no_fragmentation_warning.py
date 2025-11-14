"""
DataFrame fragmentation警告が発生しないかを検証
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from app.services.ml.feature_engineering.advanced_features import (
    AdvancedFeatureEngineer,
)


def test_no_fragmentation_warning():
    """Fragmentation警告が発生しないことをテスト"""
    print("DataFrame Fragmentation警告テスト開始")
    print("=" * 60)

    # 警告を捕まえるように設定
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # サンプルデータを生成
        dates = pd.date_range(start=datetime(2023, 1, 1), periods=2000, freq="1h")

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

        # 特徴量を生成
        engineer = AdvancedFeatureEngineer()
        features = engineer.create_advanced_features(df)

        # fragmentationに関する警告をチェック
        fragmentation_warnings = [
            warning
            for warning in w
            if "fragment" in str(warning.message).lower()
            or "insert" in str(warning.message).lower()
        ]

        print(f"\n生成された特徴量数: {features.shape[1]}")
        print(f"総警告数: {len(w)}")
        print(f"Fragmentation警告数: {len(fragmentation_warnings)}")

        if fragmentation_warnings:
            print("\n警告内容:")
            for warning in fragmentation_warnings:
                print(f"  - {warning.message}")
            print("\n[FAIL] Fragmentation警告が発生しました")
            return False
        else:
            print("\n[PASS] Fragmentation警告は発生していません")
            return True


def test_large_dataset_performance():
    """大きなデータセットでのパフォーマンステスト"""
    print("\n" + "=" * 60)
    print("大きなデータセットでのパフォーマンステスト")
    print("=" * 60)

    # より大きなデータセットを生成（20,000行）
    dates = pd.date_range(start=datetime(2020, 1, 1), periods=20000, freq="1h")

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

    print(f"データサイズ: {df.shape}")

    engineer = AdvancedFeatureEngineer()

    # 3回実行して平均を測定
    import time

    times = []
    for i in range(3):
        start = time.time()
        features = engineer.create_advanced_features(df)
        end = time.time()
        times.append(end - start)
        print(f"実行 {i+1}/3: {times[-1]:.2f}秒")

    print(f"\n平均実行時間: {np.mean(times):.2f}秒")
    print(f"最終特徴量数: {features.shape[1]}")
    print(f"秒間処理行数: {df.shape[0] / np.mean(times):.0f} 行/秒")

    # DataFrame健全性チェック
    print("\nDataFrame健全性:")
    print(f"  - 形状: {features.shape}")
    print(
        f"  - 欠損値: {features.isnull().sum().sum()} ({features.isnull().sum().sum() / features.size * 100:.2f}%)"
    )
    print(
        f"  - 無限値: {np.isinf(features.select_dtypes(include=[np.number])).sum().sum()}"
    )
    print(
        f"  - NaN: {np.isnan(features.select_dtypes(include=[np.number])).sum().sum()}"
    )

    return True


if __name__ == "__main__":
    result1 = test_no_fragmentation_warning()
    result2 = test_large_dataset_performance()

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    if result1 and result2:
        print("[PASS] すべてのテストがPASSしました")
        print("   DataFrame Fragmentation問題が解決されています")
    else:
        print("[FAIL] 失敗したテストがあります")

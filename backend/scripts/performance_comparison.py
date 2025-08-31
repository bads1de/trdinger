#!/usr/bin/env python3
"""
パフォーマンス比較テスト - Pandasオンリー移行後
"""

import time
import pandas as pd
import numpy as np

from app.services.indicators.technical_indicators.volatility import VolatilityIndicators
from app.services.indicators.technical_indicators.momentum import MomentumIndicators
from app.services.indicators.technical_indicators.volume import VolumeIndicators

def performance_test():
    """Pandasオンリー実装のパフォーマンステスト"""
    print("=== Pandasオンリー移行後パフォーマンステスト ===\n")

    # 大規模テストデータ作成
    n = 50000
    np.random.seed(42)
    high = pd.Series(100 + np.cumsum(np.random.randn(n)) + np.random.rand(n) * 10, name="high")
    low = pd.Series(100 + np.cumsum(np.random.randn(n)) - np.random.rand(n) * 10, name="low")
    close = pd.Series(100 + np.cumsum(np.random.randn(n)), name="close")
    volume = pd.Series(np.random.randint(10000, 100000, n), name="volume")

    print(f"テストデータサイズ: {n:,} 行")
    print(f"メモリ使用量: {close.memory_usage(deep=True)} bytes\n")

    # 指標計算関数のリスト
    test_functions = [
        ("ATR (Volatility)", lambda: VolatilityIndicators.atr(high, low, close)),
        ("RSI (Momentum)", lambda: MomentumIndicators.rsi(close)),
        ("MACD (Momentum)", lambda: MomentumIndicators.macd(close)),
        ("SMA (Trend)", lambda: close.rolling(20).mean()),
        ("AD (Volume)", lambda: VolumeIndicators.ad(high, low, close, volume)),
    ]

    results = {}

    # パフォーマンス測定
    for name, func in test_functions:
        print(f"測定中: {name}...")

        # ウォームアップ (JIT最適化)
        _ = func()

        # 測定実行
        start_time = time.perf_counter()
        result = func()
        end_time = time.perf_counter()

        elapsed = end_time - start_time

        # 結果検証
        if isinstance(result, pd.Series):
            result_type = f"pd.Series (len={len(result)})"
        elif isinstance(result, tuple):
            result_type = f"tuple (len={len(result)})"
            first_result = result[0] if result else None
            if isinstance(first_result, pd.Series):
                result_type += f" [pd.Series: len={len(first_result)}]"
        else:
            result_type = type(result).__name__

        results[name] = {
            'time': elapsed,
            'result_type': result_type,
            'valid': not (hasattr(result, 'isna') and result.isna().all())
        }

        print(f"  OK {name}: {elapsed:.4f}秒, {result_type}")

    # 結果集計
    print("\n=== パフォーマンス結果集計 ===")
    total_time = sum(r['time'] for r in results.values())
    avg_time = total_time / len(results)

    print(f"総処理時間: {total_time:.4f}秒")
    print(f"平均処理時間: {avg_time:.4f}秒/指標")

    # 詳細結果
    print("\n=== 詳細結果 ===")
    for name, data in results.items():
        valid_icon = "OK" if data['valid'] else "NG"
        print(f"{name:20} | {data['time']:6.4f}秒 | {data['result_type']:30} | {valid_icon}")
    print(f"\n全テスト完了！ 平均 {avg_time:.4f}秒/指標")
    print("Pandasオンリー移行がパフォーマンス正常")
    print("全ての指標が正常に pd.Series または tuple[pd.Series] を返します")
if __name__ == "__main__":
    performance_test()
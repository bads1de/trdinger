#!/usr/bin/env python3
"""
統合オペランドグループ化システム パフォーマンステスト

分類処理・互換性計算・オペランド探索の統合パフォーマンス測定
統計分析機能を強化し、複数反復実行での安定したパフォーマンス評価を行う。
"""

import time
import statistics
import sys
import os

# PYTHONPATH設定
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, backend_path)

from app.services.auto_strategy.core.operand_grouping import operand_grouping_system

# テスト指標リスト（両ファイルから統合）
test_indicators = [
    # PRICE_BASED (15種)
    "SMA", "EMA", "BB", "close", "open", "high", "low", "HMA", "ZLMA", "VWMA",
    "AA", "JMA", "MCGD", "ICHIMOKU", "HILO", "HWMA", "HL2", "HLC3", "OHLC4",
    "WCP", "SSF", "VIDYA", "KELTNER", "DONCHIAN", "SUPERTREND", "VWAP",

    # PERCENTAGE_0_100 (15種)
    "RSI", "STOCH", "ADX", "MFI", "ULTOSC", "QQE", "DX", "PLUS_DI", "MINUS_DI",
    "ADXR", "HWC", "UI", "MASSI", "VORTEX", "CHOP",

    # PERCENTAGE_NEG100_100 (5種)
    "CCI", "CMO", "AROONOSC", "TRIX", "ER",

    # ZERO_CENTERED (20種)
    "MACD", "ROC", "MOM", "ROCP", "ROCR", "TRIX", "WILLR", "APO", "PPO", "BOP",
    "TSI", "RVI", "RMI", "DPO", "T3", "EMV", "NVI", "PVI", "PVT", "EFI",


    # SPECIAL_SCALE (5種)
    "ATR", "volume", "OpenInterest", "FundingRate", "TRANGE",

    # 追加指標
    "EOM", "KVO", "CMF", "OBV", "STOCHRSI", "SMI", "PVO",
    "CFO", "CTI", "KST", "STC", "COPPOCK", "BIAS",
    "NON_EXISTENT_INDICATOR_123"
]

def get_statistics(times):
    """測定結果の統計値計算"""
    if not times:
        return 0, 0, 0, 0, 0, 0

    mean = statistics.mean(times)
    median = statistics.median(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)
    total = sum(times)

    return mean, median, stdev, min_time, max_time, total

def test_classification_performance(iterations=100):
    """分類処理パフォーマンステスト（複数反復）"""
    print("=== 分類処理パフォーマンステスト ===")
    print(f"反復回数: {iterations}")

    all_times = []

    for _ in range(iterations):
        iteration_times = []

        for indicator in test_indicators:
            start_time = time.perf_counter()
            operand_grouping_system.get_operand_group(indicator)
            end_time = time.perf_counter()

            elapsed = (end_time - start_time) * 1_000_000  # microseconds
            iteration_times.append(elapsed)

        all_times.extend(iteration_times)

    mean, median, stdev, min_t, max_t, total = get_statistics(all_times)

    total_operations = len(all_times)
    total_time_seconds = total / 1_000_000
    ops_per_sec = total_operations / total_time_seconds if total_time_seconds > 0 else 0

    print(f"\n統計結果:")
    print(f"  総操作数: {total_operations}")
    print(f"  平均時間: {mean:.3f} μs")
    print(f"  中央値: {median:.3f} μs")
    print(f"  標準偏差: {stdev:.3f} μs")
    print(f"  最小時間: {min_t:.3f} μs")
    print(f"  最大時間: {max_t:.3f} μs")
    print(f"  総実行時間: {total_time_seconds:.4f} 秒")
    print(f"  処理速度: {ops_per_sec:.2f} ops/sec")

    return mean, median, stdev, total_time_seconds

def test_compatibility_performance(iterations=50):
    """互換性計算パフォーマンステスト（複数反復）"""
    print("\n=== 互換性計算パフォーマンステスト ===")
    print(f"反復回数: {iterations}")

    test_pairs = [
        ('SMA', 'EMA'), ('RSI', 'MACD'), ('SMA', 'HMA'),
        ('RSI', 'CCI'), ('volume', 'OBV'), ('RSI', 'STOCH'),
        ('MACD', 'ROCP'), ('CCI', 'AROONOSC'), ('volume', 'EFI')
    ]

    sample_pairs = test_pairs[:20]  # パフォーマンステスト用
    all_times = []

    print(f"テストペア数: {len(sample_pairs)}")

    for iter_num in range(iterations):
        iteration_times = []

        for pair in sample_pairs:
            start_time = time.perf_counter()
            operand_grouping_system.get_compatibility_score(pair[0], pair[1])
            end_time = time.perf_counter()

            elapsed = (end_time - start_time) * 1_000_000  # microseconds
            iteration_times.append(elapsed)

        all_times.extend(iteration_times)

        # 進捗表示
        if (iter_num + 1) % 10 == 0:
            print(f"  完了: {iter_num+1}/{iterations}")

    mean, median, stdev, min_t, max_t, total = get_statistics(all_times)

    total_operations = len(all_times)
    total_time_seconds = total / 1_000_000
    ops_per_sec = total_operations / total_time_seconds if total_time_seconds > 0 else 0

    print(f"\n統計結果:")
    print(f"  総操作数: {total_operations}")
    print(f"  平均時間: {mean:.3f} μs")
    print(f"  中央値: {median:.3f} μs")
    print(f"  標準偏差: {stdev:.3f} μs")
    print(f"  最小時間: {min_t:.3f} μs")
    print(f"  最大時間: {max_t:.3f} μs")
    print(f"  総実行時間: {total_time_seconds:.4f} 秒")
    print(f"  処理速度: {ops_per_sec:.2f} ops/sec")

    return mean, median, stdev, total_time_seconds

def test_operand_discovery_performance(iterations=50):
    """オペランド探索パフォーマンステスト（複数反復）"""
    print("\n=== オペランド探索パフォーマンステスト ===")
    print(f"反復回数: {iterations}")

    target = 'SMA'
    available = [
        'EMA', 'RSI', 'MACD', 'CCI', 'volume', 'ATR', 'NVI', 'PVI', 'HMA', 'VWMA',
        'KELTNER', 'DONCHIAN', 'TSI', 'VORTEX', 'STOCH', 'ADX'
    ]

    print(f"対象オペランド: {target}")
    print(f"利用可能オペランド数: {len(available)}")

    all_times = []

    for iter_num in range(iterations):
        start_time = time.perf_counter()
        result = operand_grouping_system.get_compatible_operands(target, available, 0.8)
        end_time = time.perf_counter()

        elapsed = (end_time - start_time) * 1000  # milliseconds
        all_times.append(elapsed)

        # 進捗表示
        if (iter_num + 1) % 10 == 0:
            print(f"  完了: {iter_num+1}/{iterations}")

    mean, median, stdev, min_t, max_t, total = get_statistics(all_times)

    total_time_seconds = total / 1000

    print(f"\n統計結果:")
    print(f"  総探索数: {len(all_times)}")
    print(f"  平均時間: {mean:.3f} ms")
    print(f"  中央値: {median:.3f} ms")
    print(f"  標準偏差: {stdev:.3f} ms")
    print(f"  最小時間: {min_t:.3f} ms")
    print(f"  最大時間: {max_t:.3f} ms")
    print(f"  総実行時間: {total_time_seconds:.4f} 秒")

    return mean, median, stdev, total_time_seconds

def evaluate_performance(class_stats, compat_stats, discovery_stats):
    """総合パフォーマンス評価"""
    print("\n=== 総合パフォーマンス評価 ===")

    class_mean, class_median, class_stdev, class_total = class_stats
    compat_mean, compat_median, compat_stdev, compat_total = compat_stats
    disc_mean, disc_median, disc_stdev, disc_total = discovery_stats

    # 統合基準（マイクロ秒単位）
    integrated_performance = class_mean + compat_mean

    print(f"分類処理平均: {class_mean:.3f} μs")
    print(f"互換性計算平均: {compat_mean:.3f} μs")
    print(f"オペランド探索平均: {disc_mean:.2f} ms")

    # パフォーマンスグレード
    if integrated_performance < 10:  # 10μs以下
        grade = "A+ (優秀)"
    elif integrated_performance < 50:  # 50μs以下
        grade = "A (良好)"
    elif integrated_performance < 100:  # 100μs以下
        grade = "B (可)"
    elif integrated_performance < 500:  # 500μs以下
        grade = "C (要改善)"
    else:
        grade = "D (性能不良)"

    print(f"\n統合パフォーマンス基準: {integrated_performance:.1f} μs")
    print(f"パフォーマンス評価: {grade}")

    # 閾値情報
    if class_stdev > class_mean * 0.3:
        print("WARN: 分類処理のばらつきが大きい可能性あり")
    if compat_stdev > compat_mean * 0.3:
        print("WARN: 互換性計算のばらつきが大きい可能性あり")

    return grade

def main():
    """メイン実行関数"""
    print("統合オペランドグループ化システム パフォーマンステスト開始")
    print("=" * 60)

    iterations_class = 100  # 分類処理の反復回数
    iterations_compat = 50   # 互換性計算の反復回数
    iterations_discovery = 50  # オペランド探索の反復回数

    try:
        # 分類処理テスト
        class_stats = test_classification_performance(iterations_class)

        # 互換性計算テスト
        compat_stats = test_compatibility_performance(iterations_compat)

        # オペランド探索テスト
        discovery_stats = test_operand_discovery_performance(iterations_discovery)

        # 総合評価
        grade = evaluate_performance(class_stats, compat_stats, discovery_stats)

        print("\n" + "=" * 60)
        print("パフォーマンステスト完了 (PASS)")
        print("=" * 60)

        # 成功判定
        return grade in ["A+ (優秀)", "A (良好)"]

    except Exception as e:
        print(f"\nFAIL: パフォーマンステスト失敗: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
AutoMLメモリ使用量分析スクリプト

AutoFeat特徴量生成のメモリ使用量を詳細に分析し、
最適化の効果を測定します。
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.services.ml.feature_engineering.automl_features.autofeat_calculator import (
    AutoFeatCalculator,
)
from app.core.services.ml.feature_engineering.automl_features.automl_config import (
    AutoFeatConfig,
)
from app.core.services.ml.feature_engineering.automl_features.performance_optimizer import (
    PerformanceOptimizer,
)


def generate_test_data(
    rows: int = 1000, features: int = 10
) -> tuple[pd.DataFrame, pd.Series]:
    """
    テスト用のデータを生成

    Args:
        rows: 行数
        features: 特徴量数

    Returns:
        (DataFrame, target Series)
    """
    print(f"テストデータ生成: {rows}行 x {features}特徴量")

    # 特徴量データ生成
    data = {}
    for i in range(features):
        if i % 3 == 0:
            # 整数型特徴量
            data[f"feature_{i}"] = np.random.randint(0, 100, rows)
        elif i % 3 == 1:
            # 浮動小数点型特徴量
            data[f"feature_{i}"] = np.random.normal(0, 1, rows)
        else:
            # カテゴリカル特徴量
            categories = ["A", "B", "C", "D", "E"]
            data[f"feature_{i}"] = np.random.choice(categories, rows)

    df = pd.DataFrame(data)

    # ターゲット変数生成（回帰用）
    target = (
        df[f"feature_0"] * 0.5 + df[f"feature_1"] * 0.3 + np.random.normal(0, 0.1, rows)
    )

    return df, pd.Series(target, name="target")


def analyze_memory_usage_basic():
    """基本的なメモリ使用量分析"""
    print("\n=== 基本メモリ使用量分析 ===")

    # 小量データでのテスト
    df_small, target_small = generate_test_data(500, 5)

    # 中量データでのテスト
    df_medium, target_medium = generate_test_data(2000, 10)

    # 大量データでのテスト
    df_large, target_large = generate_test_data(5000, 15)

    test_cases = [
        ("小量データ", df_small, target_small),
        ("中量データ", df_medium, target_medium),
        ("大量データ", df_large, target_large),
    ]

    results = []

    for name, df, target in test_cases:
        print(f"\n--- {name}テスト ---")
        print(f"データサイズ: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}MB")

        # デフォルト設定でのテスト
        config_default = AutoFeatConfig()
        calculator_default = AutoFeatCalculator(config_default)

        start_time = time.time()
        with calculator_default as calc:
            result_df, info = calc.generate_features(df, target, max_features=20)

        execution_time = time.time() - start_time

        result = {
            "test_name": name,
            "data_size_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "execution_time": execution_time,
            "original_features": len(df.columns),
            "generated_features": info.get("generated_features", 0),
            "total_features": len(result_df.columns),
            "memory_before": calc._memory_usage_before,
            "memory_after": calc._memory_usage_after,
            "memory_freed": calc._memory_usage_before - calc._memory_usage_after,
        }

        results.append(result)

        print(f"実行時間: {execution_time:.2f}秒")
        print(f"生成特徴量: {info.get('generated_features', 0)}個")
        print(f"メモリ解放: {result['memory_freed']:.2f}MB")

    return results


def analyze_memory_optimization_effect():
    """メモリ最適化の効果分析"""
    print("\n=== メモリ最適化効果分析 ===")

    # 大量データでの比較テスト
    df, target = generate_test_data(3000, 12)
    data_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

    print(f"テストデータサイズ: {data_size_mb:.2f}MB")

    # 最適化前の設定（デフォルト設定）
    config_before = AutoFeatConfig(
        max_features=100,
        feateng_steps=3,
        max_gb=4.0,
        featsel_runs=2,
        verbose=1,
        n_jobs=2,
    )

    # 最適化後の設定（データサイズに基づく最適化）
    config_after = AutoFeatConfig(
        max_features=30,  # 大幅に削減
        feateng_steps=1,  # ステップ数を最小限に
        max_gb=1.0,  # メモリ制限を厳しく
        featsel_runs=1,  # 特徴量選択回数を削減
        verbose=0,  # ログを最小限に
        n_jobs=1,  # 並列処理を無効化
    )

    print(f"\n最適化前設定:")
    print(f"  max_features: {config_before.max_features}")
    print(f"  feateng_steps: {config_before.feateng_steps}")
    print(f"  max_gb: {config_before.max_gb}")
    print(f"  featsel_runs: {config_before.featsel_runs}")
    print(f"  n_jobs: {config_before.n_jobs}")

    print(f"\n最適化後設定:")
    print(f"  max_features: {config_after.max_features}")
    print(f"  feateng_steps: {config_after.feateng_steps}")
    print(f"  max_gb: {config_after.max_gb}")
    print(f"  featsel_runs: {config_after.featsel_runs}")
    print(f"  n_jobs: {config_after.n_jobs}")

    # パフォーマンス比較
    results = {}

    for config_name, config in [
        ("最適化前", config_before),
        ("最適化後", config_after),
    ]:
        print(f"\n--- {config_name}テスト実行 ---")

        calculator = AutoFeatCalculator(config)

        start_time = time.time()
        with calculator as calc:
            result_df, info = calc.generate_features(df, target, max_features=50)

        execution_time = time.time() - start_time

        results[config_name] = {
            "execution_time": execution_time,
            "generated_features": info.get("generated_features", 0),
            "memory_before": calc._memory_usage_before,
            "memory_after": calc._memory_usage_after,
            "memory_freed": calc._memory_usage_before - calc._memory_usage_after,
            "peak_memory": max(calc._memory_usage_before, calc._memory_usage_after),
        }

        print(f"実行時間: {execution_time:.2f}秒")
        print(f"生成特徴量: {info.get('generated_features', 0)}個")
        print(f"ピークメモリ: {results[config_name]['peak_memory']:.2f}MB")
        print(f"メモリ解放: {results[config_name]['memory_freed']:.2f}MB")

    # 改善効果の計算
    time_improvement = (
        (results["最適化前"]["execution_time"] - results["最適化後"]["execution_time"])
        / results["最適化前"]["execution_time"]
        * 100
    )
    memory_improvement = (
        (results["最適化前"]["peak_memory"] - results["最適化後"]["peak_memory"])
        / results["最適化前"]["peak_memory"]
        * 100
    )

    print(f"\n=== 改善効果 ===")
    print(f"実行時間改善: {time_improvement:+.1f}%")
    print(f"ピークメモリ改善: {memory_improvement:+.1f}%")

    return results


def analyze_pandas_memory_optimization():
    """pandasメモリ最適化の効果分析"""
    print("\n=== pandasメモリ最適化分析 ===")

    # テストデータ生成
    df, _ = generate_test_data(2000, 10)

    # 最適化前のメモリ使用量
    original_memory = df.memory_usage(deep=True).sum()
    print(f"最適化前メモリ使用量: {original_memory / 1024 / 1024:.2f}MB")

    # パフォーマンス最適化ツールを使用
    optimizer = PerformanceOptimizer()

    # 通常の最適化
    optimized_df = optimizer.optimize_pandas_memory_usage(df, aggressive=False)
    optimized_memory = optimized_df.memory_usage(deep=True).sum()

    # 積極的な最適化
    aggressive_df = optimizer.optimize_pandas_memory_usage(df, aggressive=True)
    aggressive_memory = aggressive_df.memory_usage(deep=True).sum()

    print(f"通常最適化後: {optimized_memory / 1024 / 1024:.2f}MB")
    print(f"積極的最適化後: {aggressive_memory / 1024 / 1024:.2f}MB")

    normal_reduction = (original_memory - optimized_memory) / original_memory * 100
    aggressive_reduction = (original_memory - aggressive_memory) / original_memory * 100

    print(f"通常最適化削減率: {normal_reduction:.1f}%")
    print(f"積極的最適化削減率: {aggressive_reduction:.1f}%")

    return {
        "original_memory_mb": original_memory / 1024 / 1024,
        "optimized_memory_mb": optimized_memory / 1024 / 1024,
        "aggressive_memory_mb": aggressive_memory / 1024 / 1024,
        "normal_reduction_percent": normal_reduction,
        "aggressive_reduction_percent": aggressive_reduction,
    }


def main():
    """メイン実行関数"""
    print("AutoMLメモリ使用量分析開始")
    print("=" * 50)

    try:
        # 基本的なメモリ使用量分析
        basic_results = analyze_memory_usage_basic()

        # メモリ最適化効果分析
        optimization_results = analyze_memory_optimization_effect()

        # pandasメモリ最適化分析
        pandas_results = analyze_pandas_memory_optimization()

        print("\n" + "=" * 50)
        print("分析完了")

        # 結果サマリー
        print("\n=== 分析結果サマリー ===")
        print("1. 基本分析:")
        for result in basic_results:
            print(
                f"   {result['test_name']}: {result['execution_time']:.2f}秒, "
                f"メモリ解放: {result['memory_freed']:.2f}MB"
            )

        print("\n2. 最適化効果:")
        if optimization_results:
            before = optimization_results["最適化前"]
            after = optimization_results["最適化後"]
            time_diff = before["execution_time"] - after["execution_time"]
            memory_diff = before["peak_memory"] - after["peak_memory"]
            print(f"   実行時間短縮: {time_diff:.2f}秒")
            print(f"   メモリ使用量削減: {memory_diff:.2f}MB")

        print("\n3. pandasメモリ最適化:")
        print(f"   通常最適化: {pandas_results['normal_reduction_percent']:.1f}%削減")
        print(
            f"   積極的最適化: {pandas_results['aggressive_reduction_percent']:.1f}%削減"
        )

    except Exception as e:
        print(f"分析中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

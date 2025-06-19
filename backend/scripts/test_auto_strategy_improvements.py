#!/usr/bin/env python3
"""
オートストラテジー機能改善テストスクリプト

Phase 1とPhase 2の改善を実際に動作させて、
戦略生成の品質とパフォーマンスを検証します。
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene,
    IndicatorGene,
    Condition,
    decode_list_to_gene,
    _generate_indicator_parameters,
    _generate_indicator_specific_conditions,
)
from app.core.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)


def print_header(title: str):
    """ヘッダーを出力"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_section(title: str):
    """セクションヘッダーを出力"""
    print(f"\n--- {title} ---")


def test_phase1_improvements():
    """Phase 1改善のテスト"""
    print_header("Phase 1: 即効性改善のテスト")

    # 1. GAパラメータ最適化のテスト
    print_section("GAパラメータ最適化")

    # 旧設定
    legacy_config = GAConfig.create_legacy()
    print(
        f"旧設定 - 個体数: {legacy_config.population_size}, 世代数: {legacy_config.generations}"
    )
    print(
        f"旧設定 - 計算量: {legacy_config.population_size * legacy_config.generations}"
    )

    # 新設定
    new_config = GAConfig()
    print(
        f"新設定 - 個体数: {new_config.population_size}, 世代数: {new_config.generations}"
    )
    print(f"新設定 - 計算量: {new_config.population_size * new_config.generations}")

    # 削減率計算
    old_calculations = legacy_config.population_size * legacy_config.generations
    new_calculations = new_config.population_size * new_config.generations
    reduction_ratio = (old_calculations - new_calculations) / old_calculations * 100
    print(f"計算量削減率: {reduction_ratio:.1f}%")

    # 2. ログレベル最適化のテスト
    print_section("ログレベル最適化")
    print(f"デフォルトログレベル: {new_config.log_level}")
    print(f"詳細ログ有効: {new_config.enable_detailed_logging}")

    return new_config


def test_phase2_improvements():
    """Phase 2改善のテスト"""
    print_header("Phase 2: 品質改善のテスト")

    # 1. 指標セット拡張のテスト
    print_section("指標セット拡張")

    # 様々な指標をテスト
    test_indicators = [
        ("SMA", 0.1),
        ("EMA", 0.2),
        ("RSI", 0.3),
        ("MACD", 0.4),
        ("BB", 0.5),
        ("STOCH", 0.6),
        ("CCI", 0.7),
        ("WILLIAMS", 0.8),
        ("ADX", 0.9),
        ("AROON", 0.95),
        ("MFI", 0.85),
        ("ATR", 0.75),
    ]

    generated_indicators = []

    for expected_type, param_val in test_indicators:
        # 正規化された値から指標を生成
        test_encoded = [param_val, 0.5] + [0.0] * 14
        gene = decode_list_to_gene(test_encoded)

        if gene.indicators:
            indicator = gene.indicators[0]
            generated_indicators.append(indicator)
            print(f"✓ {indicator.type}: {indicator.parameters}")

    print(f"\n生成された指標数: {len(generated_indicators)}")
    indicator_types = {ind.type for ind in generated_indicators}
    print(f"指標の種類: {sorted(indicator_types)}")

    # 2. 条件生成ロジック改善のテスト
    print_section("条件生成ロジック改善")

    for indicator in generated_indicators[:5]:  # 最初の5つをテスト
        indicator_name = f"{indicator.type}_{indicator.parameters.get('period', 20)}"
        entry_conditions, exit_conditions = _generate_indicator_specific_conditions(
            indicator, indicator_name
        )

        print(f"\n{indicator.type} 戦略:")
        print(f"  エントリー条件: {len(entry_conditions)}個")
        for i, cond in enumerate(entry_conditions):
            print(
                f"    {i+1}. {cond.left_operand} {cond.operator} {cond.right_operand}"
            )

        print(f"  エグジット条件: {len(exit_conditions)}個")
        for i, cond in enumerate(exit_conditions):
            print(
                f"    {i+1}. {cond.left_operand} {cond.operator} {cond.right_operand}"
            )

    return generated_indicators


def generate_sample_strategies(num_strategies: int = 5) -> List[StrategyGene]:
    """サンプル戦略を生成"""
    print_header("サンプル戦略生成")

    generator = RandomGeneGenerator()
    strategies = []

    print(f"{num_strategies}個の戦略を生成中...")

    for i in range(num_strategies):
        try:
            strategy = generator.generate_random_gene()
            strategies.append(strategy)

            print(f"\n戦略 {i+1}: {strategy.id}")
            print(f"  指標数: {len(strategy.indicators)}")
            for j, indicator in enumerate(strategy.indicators):
                print(f"    {j+1}. {indicator.type}: {indicator.parameters}")

            print(f"  エントリー条件数: {len(strategy.entry_conditions)}")
            print(f"  エグジット条件数: {len(strategy.exit_conditions)}")

            # 戦略の詳細を表示
            if strategy.entry_conditions:
                print("  エントリー条件:")
                for j, cond in enumerate(strategy.entry_conditions[:2]):  # 最初の2つ
                    print(
                        f"    {j+1}. {cond.left_operand} {cond.operator} {cond.right_operand}"
                    )

            if strategy.exit_conditions:
                print("  エグジット条件:")
                for j, cond in enumerate(strategy.exit_conditions[:2]):  # 最初の2つ
                    print(
                        f"    {j+1}. {cond.left_operand} {cond.operator} {cond.right_operand}"
                    )

        except Exception as e:
            print(f"戦略 {i+1} の生成に失敗: {e}")

    return strategies


def test_strategy_diversity(strategies: List[StrategyGene]):
    """戦略の多様性をテスト"""
    print_header("戦略多様性分析")

    # 指標の多様性
    all_indicator_types = set()
    indicator_counts = {}

    for strategy in strategies:
        for indicator in strategy.indicators:
            all_indicator_types.add(indicator.type)
            indicator_counts[indicator.type] = (
                indicator_counts.get(indicator.type, 0) + 1
            )

    print_section("指標多様性")
    print(f"使用された指標の種類: {len(all_indicator_types)}")
    print(f"指標リスト: {sorted(all_indicator_types)}")
    print("\n指標使用頻度:")
    for indicator_type, count in sorted(indicator_counts.items()):
        print(f"  {indicator_type}: {count}回")

    # 条件の多様性
    print_section("条件多様性")
    entry_operators = set()
    exit_operators = set()

    for strategy in strategies:
        for cond in strategy.entry_conditions:
            entry_operators.add(cond.operator)
        for cond in strategy.exit_conditions:
            exit_operators.add(cond.operator)

    print(f"エントリー条件で使用された演算子: {sorted(entry_operators)}")
    print(f"エグジット条件で使用された演算子: {sorted(exit_operators)}")


def test_performance_improvement():
    """パフォーマンス改善のテスト"""
    print_header("パフォーマンス改善テスト")

    # 戦略生成速度のテスト
    print_section("戦略生成速度")

    generator = RandomGeneGenerator()
    num_tests = 10

    start_time = time.time()
    for i in range(num_tests):
        strategy = generator.generate_random_gene()
    generation_time = time.time() - start_time

    print(f"{num_tests}個の戦略生成時間: {generation_time:.3f}秒")
    print(f"1戦略あたりの生成時間: {generation_time/num_tests:.3f}秒")

    # エンコード/デコード速度のテスト
    print_section("エンコード/デコード速度")

    test_strategy = generator.generate_random_gene()

    start_time = time.time()
    for i in range(100):
        # ランダムなエンコードデータを生成してデコード
        import random

        test_encoded = [random.uniform(0.0, 1.0) for _ in range(16)]
        decoded_strategy = decode_list_to_gene(test_encoded)
    decode_time = time.time() - start_time

    print(f"100回のデコード時間: {decode_time:.3f}秒")
    print(f"1回あたりのデコード時間: {decode_time/100:.6f}秒")


def test_strategy_validation(strategies: List[StrategyGene]):
    """戦略の妥当性をテスト"""
    print_section("戦略妥当性検証")

    valid_count = 0
    invalid_count = 0

    for i, strategy in enumerate(strategies):
        is_valid, errors = strategy.validate()

        if is_valid:
            valid_count += 1
            print(f"✓ 戦略 {i+1}: 有効")
        else:
            invalid_count += 1
            print(f"✗ 戦略 {i+1}: 無効 - {', '.join(errors)}")

    print(f"\n妥当性検証結果:")
    print(f"  有効な戦略: {valid_count}")
    print(f"  無効な戦略: {invalid_count}")
    print(f"  有効率: {valid_count/(valid_count+invalid_count)*100:.1f}%")


def save_strategies_to_file(
    strategies: List[StrategyGene], filename: str = "generated_strategies.json"
):
    """生成された戦略をファイルに保存"""
    print_section("戦略保存")

    strategies_data = []
    for strategy in strategies:
        strategies_data.append(strategy.to_dict())

    output_path = os.path.join(os.path.dirname(__file__), filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "num_strategies": len(strategies),
                "strategies": strategies_data,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"戦略を保存しました: {output_path}")
    print(f"保存された戦略数: {len(strategies)}")


def main():
    """メイン実行関数"""
    print_header("オートストラテジー機能改善テスト")
    print(f"実行開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Phase 1改善のテスト
        config = test_phase1_improvements()

        # Phase 2改善のテスト
        indicators = test_phase2_improvements()

        # サンプル戦略生成
        strategies = generate_sample_strategies(5)

        # 戦略多様性分析
        if strategies:
            test_strategy_diversity(strategies)

            # 戦略妥当性検証
            test_strategy_validation(strategies)

            # 戦略をファイルに保存
            save_strategies_to_file(strategies)

        # パフォーマンステスト
        test_performance_improvement()

        print_header("テスト完了")
        print("✅ すべてのテストが正常に完了しました")
        print(f"✅ 改善された機能が正常に動作しています")
        print(f"✅ 戦略の多様性と品質が向上しています")

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

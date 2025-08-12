#!/usr/bin/env python3
"""
シンプルな戦略生成テスト

リファクタリング後の基本的な動作確認
"""

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.models.gene_serialization import GeneSerializer

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_simple_strategy_generation():
    """シンプルな戦略生成テスト"""
    logger.info("=== シンプル戦略生成テスト開始 ===")

    # 再現性のためのシード設定
    random.seed(123)
    np.random.seed(123)

    # シンプルなGA設定
    ga_config = GAConfig(
        population_size=3,
        generations=2,
        max_indicators=2,
        min_indicators=1,
        max_conditions=2,
        min_conditions=1,
        indicator_mode="technical_only",
        allowed_indicators=["SMA", "EMA", "RSI", "MACD"],  # 基本的な指標のみ
        log_level="INFO",
    )

    logger.info(f"GA設定: {ga_config.indicator_mode}")
    logger.info(f"許可指標: {ga_config.allowed_indicators}")

    # 遺伝子生成器を作成
    gene_generator = RandomGeneGenerator(ga_config, enable_smart_generation=True)

    # 戦略を生成
    strategies = []
    for i in range(3):
        logger.info(f"\n--- 戦略 {i+1} 生成 ---")
        try:
            gene = gene_generator.generate_random_gene()

            logger.info(f"指標: {[ind.type for ind in gene.indicators]}")
            logger.info(f"ロング条件数: {len(gene.long_entry_conditions)}")
            logger.info(f"ショート条件数: {len(gene.short_entry_conditions)}")

            # 条件の詳細を表示（ConditionGroupの場合は文字列表現を使用）
            for j, cond in enumerate(gene.long_entry_conditions):
                if hasattr(cond, "left_operand"):
                    logger.info(
                        f"  ロング条件{j+1}: {cond.left_operand} {cond.operator} {cond.right_operand}"
                    )
                else:
                    logger.info(f"  ロング条件{j+1}: {str(cond)}")

            for j, cond in enumerate(gene.short_entry_conditions):
                if hasattr(cond, "left_operand"):
                    logger.info(
                        f"  ショート条件{j+1}: {cond.left_operand} {cond.operator} {cond.right_operand}"
                    )
                else:
                    logger.info(f"  ショート条件{j+1}: {str(cond)}")

            strategies.append(gene)

        except Exception as e:
            logger.error(f"戦略{i+1}生成エラー: {e}")

    logger.info(f"\n✅ {len(strategies)}個の戦略を生成しました")
    return strategies


def test_strategy_serialization(strategies):
    """戦略のシリアライゼーションテスト"""
    if not strategies:
        logger.error("テスト対象の戦略がありません")
        return None

    logger.info("\n=== シリアライゼーションテスト ===")

    test_strategy = strategies[0]
    serializer = GeneSerializer()

    try:
        # 辞書形式にシリアライズ
        strategy_dict = serializer.strategy_gene_to_dict(test_strategy)
        logger.info("✅ 辞書シリアライゼーション成功")

        # 辞書から復元
        restored_strategy = serializer.dict_to_strategy_gene(
            strategy_dict, type(test_strategy)
        )
        logger.info("✅ 辞書デシリアライゼーション成功")

        return strategy_dict

    except Exception as e:
        logger.error(f"❌ シリアライゼーションエラー: {e}")
        return None


def display_strategy_summary(strategy_dict):
    """戦略の要約を表示"""
    if not strategy_dict:
        return

    print("\n" + "=" * 50)
    print("🎯 生成された戦略の要約")
    print("=" * 50)

    # 指標情報
    indicators = strategy_dict.get("indicators", [])
    print(f"\n📊 使用指標 ({len(indicators)}個):")
    for i, ind in enumerate(indicators, 1):
        params = ind.get("parameters", {})
        params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        print(f"  {i}. {ind.get('type', 'N/A')} ({params_str})")

    # エントリー条件
    long_conditions = strategy_dict.get("long_entry_conditions", [])
    short_conditions = strategy_dict.get("short_entry_conditions", [])

    print(f"\n📈 ロングエントリー条件 ({len(long_conditions)}個):")
    for i, cond in enumerate(long_conditions, 1):
        print(
            f"  {i}. {cond.get('left_operand', 'N/A')} {cond.get('operator', 'N/A')} {cond.get('right_operand', 'N/A')}"
        )

    print(f"\n📉 ショートエントリー条件 ({len(short_conditions)}個):")
    for i, cond in enumerate(short_conditions, 1):
        print(
            f"  {i}. {cond.get('left_operand', 'N/A')} {cond.get('operator', 'N/A')} {cond.get('right_operand', 'N/A')}"
        )

    # TP/SL設定
    tpsl_gene = strategy_dict.get("tpsl_gene", {})
    if tpsl_gene and tpsl_gene.get("enabled"):
        print(f"\n🎯 TP/SL設定:")
        print(f"  方式: {tpsl_gene.get('method', 'N/A')}")
        print(f"  ストップロス: {tpsl_gene.get('stop_loss_pct', 0)*100:.2f}%")
        print(f"  テイクプロフィット: {tpsl_gene.get('take_profit_pct', 0)*100:.2f}%")

    print("=" * 50)


def analyze_strategy_patterns(strategies):
    """戦略パターンの分析"""
    if not strategies:
        return

    print("\n" + "=" * 50)
    print("📊 戦略パターン分析")
    print("=" * 50)

    # 指標の使用頻度
    indicator_count = {}
    for strategy in strategies:
        for ind in strategy.indicators:
            indicator_count[ind.type] = indicator_count.get(ind.type, 0) + 1

    print("\n📈 指標使用頻度:")
    for indicator, count in sorted(
        indicator_count.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {indicator}: {count}回")

    # 条件パターンの分析
    operators = {}
    for strategy in strategies:
        for cond in strategy.long_entry_conditions + strategy.short_entry_conditions:
            operators[cond.operator] = operators.get(cond.operator, 0) + 1

    print("\n🔄 演算子使用頻度:")
    for op, count in sorted(operators.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op}: {count}回")

    print("=" * 50)


def main():
    """メイン実行関数"""
    logger.info("🚀 シンプル戦略生成テスト開始")

    try:
        # 1. 戦略生成テスト
        strategies = test_simple_strategy_generation()
        if not strategies:
            logger.error("❌ 戦略生成に失敗しました")
            return

        # 2. シリアライゼーションテスト
        strategy_dict = test_strategy_serialization(strategies)

        # 3. 戦略詳細表示
        if strategy_dict:
            display_strategy_summary(strategy_dict)

        # 4. パターン分析
        analyze_strategy_patterns(strategies)

        logger.info("\n✅ すべてのテストが正常に完了しました！")

        # 戦略の例を返す
        return strategies[0] if strategies else None

    except Exception as e:
        logger.error(f"❌ テスト実行中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n🎉 戦略生成成功！戦略ID: {result.id}")

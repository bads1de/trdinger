#!/usr/bin/env python3
"""
ロング・ショート戦略不均衡デバッグツール

オートストラテジーで生成された戦略のロング・ショート条件の
バランスを分析し、問題点を特定する。
"""

import logging
import sys
from pathlib import Path
import traceback

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.auto_strategy.models.strategy_models import StrategyGene
from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
from app.services.auto_strategy.models.strategy_models import IndicatorGene

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_indicators():
    """テスト用の指標リストを作成"""
    indicators = [
        IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
        IndicatorGene(type="EMA", parameters={"period": 50}, enabled=True),
        IndicatorGene(type="MACD", parameters={}, enabled=True),
        IndicatorGene(type="BB", parameters={"period": 20}, enabled=True),
    ]
    return indicators


def analyze_strategy_balance():
    """戦略のロング・ショートバランスを分析"""
    logger.info("=== オートストラテジー ロング・ショートバランス分析を開始 ===")

    try:
        # スマート条件生成器を作成
        generator = SmartConditionGenerator()

        # テスト指標を作成
        test_indicators = create_test_indicators()
        logger.info(f"テスト指標: {[ind.type for ind in test_indicators]}")

        # 複数の戦略を生成・分析
        strategies_results = []

        for i in range(5):
            logger.info(f"\n--- 戦略 {i+1} 生成・分析 ---")

            try:
                # 条件を生成
                long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(test_indicators)

                # 結果を分析
                result = {
                    "strategy_id": f"test_strategy_{i+1}",
                    "long_conditions_count": len(long_conds),
                    "short_conditions_count": len(short_conds),
                    "exit_conditions_count": len(exit_conds),
                    "long_conditions": long_conds,
                    "short_conditions": short_conds,
                    "exit_conditions": exit_conds,
                }

                strategies_results.append(result)

                # 個別分析結果を出力
                logger.info(f"ロング条件数: {result['long_conditions_count']}")
                logger.info(f"ショート条件数: {result['short_conditions_count']}")
                logger.info(f"イグジット条件数: {result['exit_conditions_count']}")

                # 条件詳細を出力
                logger.info("--- ロング条件詳細 ---")
                for j, cond in enumerate(long_conds):
                    if hasattr(cond, 'left_operand'):
                        logger.info(f"  {j+1}: {cond.left_operand} {cond.operator} {cond.right_operand}")
                    else:
                        logger.info(f"  {j+1}: 条件グループ ({len(cond.conditions)}個)")

                logger.info("--- ショート条件詳細 ---")
                for j, cond in enumerate(short_conds):
                    if hasattr(cond, 'left_operand'):
                        logger.info(f"  {j+1}: {cond.left_operand} {cond.operator} {cond.right_operand}")
                    else:
                        logger.info(f"  {j+1}: 条件グループ ({len(cond.conditions)}個)")

            except Exception as e:
                logger.error(f"戦略{i+1}生成エラー: {e}")
                traceback.print_exc()

        # 全体統計分析
        logger.info("\n=== 全体分析結果 ===")

        if strategies_results:
            total_long = sum(r['long_conditions_count'] for r in strategies_results)
            total_short = sum(r['short_conditions_count'] for r in strategies_results)
            avg_long = total_long / len(strategies_results)
            avg_short = total_short / len(strategies_results)

            logger.info(".2f")
            logger.info(".2f")
            logger.info(".2f")

            # 問題点の特定
            imbalance_ratio = avg_short / avg_long if avg_long > 0 else float('inf')
            logger.info(".2f")

            if imbalance_ratio < 0.5:
                logger.warning("⚠️  問題検出: ショート条件がロング条件の50%未満 - ロングオンリーの原因！")
            elif imbalance_ratio > 2.0:
                logger.warning("⚠️  問題検出: ショート条件が多すぎる（条件成立しにくくなる可能性）")
            else:
                logger.info("✅ バランス良好: ロング・ショート条件の割合が適切")

        return strategies_results

    except Exception as e:
        logger.error(f"分析エラー: {e}")
        traceback.print_exc()
        return []


def debug_condition_generation_internals():
    """内部生成ロジックのデバッグ"""
    logger.info("\n=== 条件生成内部ロジックデバッグ ===")

    try:
        generator = SmartConditionGenerator()
        test_indicators = create_test_indicators()

        logger.info("各戦略タイプによる条件生成をテスト:")

        strategy_types = [
            "DIFFERENT_INDICATORS",
            "TIME_SEPARATION",
            "COMPLEX_CONDITIONS",
            "INDICATOR_CHARACTERISTICS"
        ]

        for strategy_type in strategy_types:
            logger.info(f"\n--- {strategy_type} タイプテスト ---")

            try:
                # 戦略タイプを直接指定して条件生成（テスト用）
                from app.services.auto_strategy.generators.smart_condition_generator import StrategyType

                if hasattr(StrategyType, strategy_type):
                    st = getattr(StrategyType, strategy_type)

                    # メソッド名を生成
                    method_name = f"_generate_{strategy_type.lower()}_strategy"
                    if hasattr(generator, method_name):
                        method = getattr(generator, method_name)
                        long_conds, short_conds, exit_conds = method(test_indicators)

                        logger.info(f"  ロング条件数: {len(long_conds)}")
                        logger.info(f"  ショート条件数: {len(short_conds)}")
                        logger.info(".2f")
                    else:
                        logger.warning(f"  メソッド {method_name} が見つからない")

            except Exception as e:
                logger.error(f"  {strategy_type} テストエラー: {e}")

    except Exception as e:
        logger.error(f"内部デバッグエラー: {e}")
        traceback.print_exc()


def generate_strategy_gene_test():
    """StrategyGene統合テスト"""
    logger.info("\n=== StrategyGene統合テスト ===")

    try:
        generator = SmartConditionGenerator()
        test_indicators = create_test_indicators()

        # 条件生成
        long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(test_indicators)

        # StrategyGene作成
        strategy_gene = StrategyGene(
            id="debug_test_strategy",
            indicators=test_indicators,
            long_entry_conditions=long_conds,
            short_entry_conditions=short_conds,
            exit_conditions=exit_conds,
        )

        logger.info("StrategyGene作成成功")
        logger.info(f"Long/Short分離: {strategy_gene.has_long_short_separation()}")

        # 条件取得テスト
        effective_long = strategy_gene.get_effective_long_conditions()
        effective_short = strategy_gene.get_effective_short_conditions()

        logger.info(f"有効ロング条件数: {len(effective_long)}")
        logger.info(f"有効ショート条件数: {len(effective_short)}")

        # 条件詳細ログ
        logger.info("--- 有効ロング条件 ---")
        for i, cond in enumerate(effective_long):
            if hasattr(cond, 'left_operand'):
                logger.info(f"  {i+1}: {cond.left_operand} {cond.operator} {cond.right_operand}")
            else:
                logger.info(f"  {i+1}: 条件グループ")

        logger.info("--- 有効ショート条件 ---")
        for i, cond in enumerate(effective_short):
            if hasattr(cond, 'left_operand'):
                logger.info(f"  {i+1}: {cond.left_operand} {cond.operator} {cond.right_operand}")
            else:
                logger.info(f"  {i+1}: 条件グループ")

        return strategy_gene

    except Exception as e:
        logger.error(f"StrategyGeneテストエラー: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    logger.info("ロング・ショート戦略不均衡デバッグツール開始")

    try:
        # メイン分析実行
        results = analyze_strategy_balance()

        # 内部ロジックデバッグ
        debug_condition_generation_internals()

        # StrategyGene統合テスト
        strategy_gene = generate_strategy_gene_test()

        logger.info("\n=== デバッグ完了 ===")
        logger.info(f"計{len(results)}個の戦略で分析を実行しました")

        if strategy_gene:
            logger.info("StrategyGeneテスト成功")
        else:
            logger.warning("StrategyGeneテスト失敗")

    except Exception as e:
        logger.error(f"メインエラー: {e}")
        traceback.print_exc()
        sys.exit(1)
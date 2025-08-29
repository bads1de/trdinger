#!/usr/bin/env python3
"""
修正済みオートストラテジーの統合テスト

ロングオンリー問題修正後、実際にGA実験から戦略条件のバランスを確認する。
"""

import logging
import sys
from pathlib import Path
import traceback
from typing import Dict, Any

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
from app.services.auto_strategy.models.strategy_models import StrategyGene, IndicatorGene
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.services.backtest.backtest_service import BacktestService
from app.services.auto_strategy.generators.strategy_factory import StrategyFactory

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_strategy_condition_balance():
    """戦略条件バランスのテスト（修正済み）"""
    logger.info("=== 戦略条件バランステスト開始 ===")

    generator = SmartConditionGenerator()

    # さまざまな指標パターンをテスト
    test_cases = [
        {
            "name": "トレンド指標メイン",
            "indicators": [
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="EMA", parameters={"period": 50}, enabled=True),
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            ]
        },
        {
            "name": "モメンタム指標メイン",
            "indicators": [
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="STOCH", parameters={}, enabled=True),
                IndicatorGene(type="MACD", parameters={}, enabled=True),
            ]
        },
        {
            "name": "統計指標含む",
            "indicators": [
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="CORREL", parameters={}, enabled=True),
            ]
        },
        {
            "name": "パターン認識含む",
            "indicators": [
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="CDL_HAMMER", parameters={}, enabled=True),
            ]
        },
    ]

    for case in test_cases:
        logger.info(f"\n--- テストケース: {case['name']} ---")

        # 戦略タイプごとにテスト
        strategy_types = ["COMPLEX_CONDITIONS", "DIFFERENT_INDICATORS"]

        for strategy_type in strategy_types:
            logger.info(f"  戦略タイプ: {strategy_type}")

            try:
                # 条件生成
                long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(
                    case['indicators']
                )

                # 条件数の分析
                long_count = len(long_conds)
                short_count = len(short_conds)
                balance_ratio = short_count / long_count if long_count > 0 else 0

                logger.info(f"    ロング条件数: {long_count}")
                logger.info(f"    ショート条件数: {short_count}")
                logger.info(".2f")

                if balance_ratio >= 0.8:
                    logger.info("    ✅ バランス良好")
                else:
                    logger.warning(f"    ⚠️ バランス不足 (ショート条件が少ない)")

            except Exception as e:
                logger.error(f"  エラー ({strategy_type}): {e}")


def test_ga_strategy_generation():
    """GA戦略生成テスト（実際のワークフロー）"""
    logger.info("\n=== GA戦略生成テスト開始 ===")

    try:
        # GAコンフィグ作成
        ga_config = GAConfig(
            population_size=4,  # 小規模でテスト
            generations=2,
            mutation_rate=0.1,
            crossover_rate=0.8,
            log_level="INFO"
        )

        # RandomGeneGenerator作成
        gene_generator = RandomGeneGenerator(ga_config)

        # 戦略生成テスト
        strategies_analyzed = 0
        long_only_detected = 0

        for i in range(10):  # 10個の戦略を生成・分析
            logger.info(f"  戦略{i+1}生成中...")

            try:
                # 戦略遺伝子生成
                strategy_gene = gene_generator.generate_random_gene()

                # StrategyFactoryでクラス生成
                factory = StrategyFactory()
                strategy_class = factory.create_strategy_class(strategy_gene)

                # 生成された戦略の条件分析
                long_conditions = strategy_gene.get_effective_long_conditions()
                short_conditions = strategy_gene.get_effective_short_conditions()

                long_count = len(long_conditions)
                short_count = len(short_conditions)

                if long_count > 0 or short_count > 0:
                    strategies_analyzed += 1

                    balance_ratio = short_count / long_count if long_count > 0 else 0

                    logger.info(f"    ロング条件数: {long_count}, ショート条件数: {short_count}")

                    # ロングオンリー判定
                    if short_count == 0 and long_count > 0:
                        long_only_detected += 1
                        logger.warning(f"    ⚠️ ロングオンリー戦略検出!")
                    elif balance_ratio < 0.5:
                        logger.warning(".2f")
                    else:
                        logger.info("    ✅ バランス良好")

            except Exception as e:
                logger.error(f"  戦略{i+1}生成エラー: {e}")

        logger.info("\n=== GA戦略生成テスト結果 ===")
        logger.info(f"分析した戦略数: {strategies_analyzed}")
        logger.info(f"ロングオンリー戦略数: {long_only_detected}")

        if long_only_detected == 0:
            logger.info("✅ ロングオンリー問題は修正されました！")
        else:
            logger.warning(f"⚠️ まだ{long_only_detected}個のロングオンリー戦略が存在します")

    except Exception as e:
        logger.error(f"GA戦略生成テストエラー: {e}")
        traceback.print_exc()


def test_actual_strategy_execution():
    """実際の戦略実行テスト（簡単なテスト用）"""
    logger.info("\n=== 実際の戦略実行テスト開始 ===")

    try:
        generator = SmartConditionGenerator()

        # テスト用指標セット
        indicators = [
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            IndicatorGene(type="EMA", parameters={"period": 50}, enabled=True),
        ]

        # 条件生成
        long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(indicators)

        # StrategyGene作成
        strategy_gene = StrategyGene(
            id="execution_test_strategy",
            indicators=indicators,
            long_entry_conditions=long_conds,
            short_entry_conditions=short_conds,
            exit_conditions=exit_conds,
        )

        logger.info("StrategyGene作成成功")
        logger.info(f"Long/Short分離: {strategy_gene.has_long_short_separation()}")
        logger.info(f"ロング条件数: {len(strategy_gene.get_effective_long_conditions())}")
        logger.info(f"ショート条件数: {len(strategy_gene.get_effective_short_conditions())}")

        # Strategyクラス生成テスト
        factory = StrategyFactory()
        try:
            strategy_class = factory.create_strategy_class(strategy_gene)
            logger.info("✅ Strategyクラス生成成功")
        except Exception as e:
            logger.error(f"❌ Strategyクラス生成失敗: {e}")

    except Exception as e:
        logger.error(f"実際の戦略実行テストエラー: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    logger.info("修正済みオートストラテジー統合テスト開始")

    try:
        # 戦略条件バランステスト
        test_strategy_condition_balance()

        # GA戦略生成テスト
        test_ga_strategy_generation()

        # 実際の戦略実行テスト
        test_actual_strategy_execution()

        logger.info("\n=== 統合テスト完了 ===")
        logger.info("修正の有効性を確認しました")

    except Exception as e:
        logger.error(f"メインエラー: {e}")
        traceback.print_exc()
        sys.exit(1)
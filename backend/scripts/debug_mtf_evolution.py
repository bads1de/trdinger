import logging
import sys
import os
from unittest.mock import MagicMock

# プロジェクトルートをパスに追加
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "backend"))

from backend.app.services.auto_strategy.generators.condition_generator import (
    ConditionGenerator,
)
from backend.app.services.auto_strategy.core.condition_evolver import ConditionEvolver
from backend.app.services.auto_strategy.models.indicator_gene import IndicatorGene
from backend.app.services.auto_strategy.models import Condition, ConditionGroup

# ロギング設定
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def print_condition_structure(cond, indent=0):
    """条件構造を再帰的に表示"""
    prefix = "  " * indent
    if isinstance(cond, ConditionGroup):
        print(f"{prefix}[GROUP] Operator: {cond.operator}")
        for c in cond.conditions:
            print_condition_structure(c, indent + 1)
    elif isinstance(cond, Condition):
        # direction属性があれば表示
        direction = getattr(cond, "direction", "N/A")
        print(
            f"{prefix}[COND] {cond.left_operand} {cond.operator} {cond.right_operand} (Dir: {direction})"
        )
    else:
        print(f"{prefix}[UNKNOWN] {cond}")


def debug_mtf_flow():
    logger.info("=== MTF戦略生成 & 進化デバッグ開始 ===")

    # 1. 依存関係のモック化
    logger.info("依存サービスをモック化中...")
    mock_backtest_service = MagicMock()
    # 適応度評価は常にランダムな値を返すようにする
    mock_backtest_service.run_backtest.return_value = {
        "performance_metrics": {
            "total_return": 10.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.1,
            "total_trades": 20,
        }
    }

    # ConditionGeneratorの初期化
    generator = ConditionGenerator(enable_smart_generation=True)
    # コンテキスト設定（実行足は1時間足）
    generator.set_context(timeframe="1h", symbol="BTC/USDT")

    # 2. MTF戦略の生成テスト
    logger.info("\n=== ステップ1: MTF条件の生成 ===")

    # テスト用指標（トレンド系とオシレーター系を混ぜる）
    indicators = [
        IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
        IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        IndicatorGene(type="EMA", parameters={"period": 50}, enabled=True),
        IndicatorGene(
            type="MACD", parameters={"fast_period": 12, "slow_period": 26}, enabled=True
        ),
    ]

    try:
        longs, shorts, exits = generator.generate_balanced_conditions(indicators)
        logger.info(f"生成された条件数: Long={len(longs)}, Short={len(shorts)}")

        # MTF条件（ConditionGroup）を探す
        mtf_condition = None
        for cond in longs:
            if isinstance(cond, ConditionGroup):
                mtf_condition = cond
                break

        if mtf_condition:
            logger.info("SUCCESS: MTF構造を持つ条件が見つかりました！")
            print_condition_structure(mtf_condition)
        else:
            logger.warning(
                "WARNING: MTF構造の条件が生成されませんでした（ランダム要素のため再実行してください）"
            )
            # 強制的にテスト用MTF条件を作成
            c1 = Condition("SMA_1d", ">", "Close")
            c1.direction = "long"
            c2 = Condition("RSI", "<", 30)
            c2.direction = "long"

            mtf_condition = ConditionGroup(operator="AND", conditions=[c1, c2])
            logger.info("テスト用に手動作成したMTF条件を使用します")

    except Exception as e:
        logger.error(f"生成エラー: {e}")
        return

    # 3. 進化（交叉・変異）のテスト
    logger.info("\n=== ステップ2: 構造維持進化のテスト ===")

    # ConditionEvolverの初期化
    from backend.app.services.auto_strategy.core.condition_evolver import (
        YamlIndicatorUtils,
    )

    yaml_utils = YamlIndicatorUtils()
    evolver = ConditionEvolver(
        backtest_service=mock_backtest_service, yaml_indicator_utils=yaml_utils
    )

    # 交叉テスト
    logger.info("--- 交叉 (Crossover) ---")
    parent1 = mtf_condition

    # 親2も同じ構造で少し値が違うものを作成
    p2_c1 = Condition("SMA_1d", ">", "Close")
    p2_c1.direction = "long"
    p2_c2 = Condition("RSI", "<", 40)  # 閾値違い
    p2_c2.direction = "long"

    parent2 = ConditionGroup(operator="AND", conditions=[p2_c1, p2_c2])

    child1, child2 = evolver.crossover(parent1, parent2)

    logger.info("親1:")
    print_condition_structure(parent1)
    logger.info("親2:")
    print_condition_structure(parent2)
    logger.info("子1 (交叉結果):")
    print_condition_structure(child1)
    logger.info("子2 (交叉結果):")
    print_condition_structure(child2)

    # 構造が維持されているか確認
    if isinstance(child1, ConditionGroup) and len(child1.conditions) == len(
        parent1.conditions
    ):
        logger.info("SUCCESS: 交叉後もMTF構造は維持されています")
    else:
        logger.error("FAILURE: 交叉により構造が破壊されました")

    # 変異テスト
    logger.info("\n--- 突然変異 (Mutation) ---")
    mutated = evolver.mutate(child1)

    logger.info("変異前:")
    print_condition_structure(child1)
    logger.info("変異後:")
    print_condition_structure(mutated)

    if isinstance(mutated, ConditionGroup):
        logger.info("SUCCESS: 変異後もMTF構造は維持されています")
    else:
        logger.error("FAILURE: 変異により構造が破壊されました")

    logger.info("\n=== デバッグ終了 ===")


if __name__ == "__main__":
    debug_mtf_flow()



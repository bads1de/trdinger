"""
戦略遺伝子ユーティリティ
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_default_strategy_gene(strategy_gene_class) -> Any:
    """
    デフォルト戦略遺伝子を作成

    Args:
        strategy_gene_class: StrategyGeneクラス

    Returns:
        デフォルトの戦略遺伝子オブジェクト
    """
    try:

        from ..models.gene_strategy import Condition, IndicatorGene

        # デフォルト指標
        indicators = [
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        ]

        # デフォルト条件（JSON形式の指標名）
        entry_conditions = [
            Condition(left_operand="RSI", operator="<", right_operand=30)
        ]
        exit_conditions = [
            Condition(left_operand="RSI", operator=">", right_operand=70)
        ]

        # リスク管理設定
        risk_management = {
            "stop_loss": 0.03,
            "take_profit": 0.15,
            "position_size": 0.1,
        }

        return strategy_gene_class(
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            risk_management=risk_management,
            metadata={"generated_by": "default_gene_utility"},
        )

    except Exception as e:
        logger.error(f"デフォルト戦略遺伝子作成エラー (ユーティリティ): {e}")
        return strategy_gene_class()

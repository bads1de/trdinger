"""
遺伝子交叉ユーティリティ
"""
from __future__ import annotations

from typing import Tuple

from ..utils.gene_utils import GeneticUtils
from .position_sizing_gene import PositionSizingGene
from .tpsl_gene import TPSLGene


def crossover_position_sizing_genes(
    parent1: PositionSizingGene, parent2: PositionSizingGene
) -> Tuple[PositionSizingGene, PositionSizingGene]:
    """ポジションサイジング遺伝子の交叉（ジェネリック関数使用）"""

    # フィールドのカテゴリ分け
    numeric_fields = [
        "optimal_f_multiplier",
        "atr_multiplier",
        "risk_per_trade",
        "fixed_ratio",
        "fixed_quantity",
        "min_position_size",
        "max_position_size",
        "priority",
        "lookback_period",
        "atr_period",
    ]
    enum_fields = ["method"]
    choice_fields = ["enabled"]

    return GeneticUtils.crossover_generic_genes(
        parent1_gene=parent1,
        parent2_gene=parent2,
        gene_class=PositionSizingGene,
        numeric_fields=numeric_fields,
        enum_fields=enum_fields,
        choice_fields=choice_fields,
    )


def crossover_tpsl_genes(
    parent1: TPSLGene, parent2: TPSLGene
) -> Tuple[TPSLGene, TPSLGene]:
    """TP/SL遺伝子の交叉（ジェネリック関数使用）"""

    # 基本フィールドのカテゴリ分け
    numeric_fields = [
        "stop_loss_pct",
        "take_profit_pct",
        "risk_reward_ratio",
        "base_stop_loss",
        "atr_multiplier_sl",
        "atr_multiplier_tp",
        "confidence_threshold",
        "priority",
        "lookback_period",
        "atr_period",
    ]
    enum_fields = ["method"]
    choice_fields = ["enabled"]

    # ジェネリック交叉を実行
    child1, child2 = GeneticUtils.crossover_generic_genes(
        parent1_gene=parent1,
        parent2_gene=parent2,
        gene_class=TPSLGene,
        numeric_fields=numeric_fields,
        enum_fields=enum_fields,
        choice_fields=choice_fields,
    )

    # 共有参照を防ぐため、method_weightsをコピー
    if hasattr(child1, 'method_weights') and isinstance(child1.method_weights, dict):
        child1.method_weights = child1.method_weights.copy()
    if hasattr(child2, 'method_weights') and isinstance(child2.method_weights, dict):
        child2.method_weights = child2.method_weights.copy()

    # method_weightsの特殊処理
    # 辞書の各キーにたいして比率の平均を取る
    all_keys = set(parent1.method_weights.keys()) | set(parent2.method_weights.keys())
    for key in all_keys:
        if key in parent1.method_weights and key in parent2.method_weights:
            # 両方にある場合、平均を取る
            child1.method_weights[key] = (
                parent1.method_weights[key] + parent2.method_weights[key]
            ) / 2
            child2.method_weights[key] = (
                parent1.method_weights[key] + parent2.method_weights[key]
            ) / 2
        else:
            # 片方しかない場合、そのまま継承
            if key in parent1.method_weights:
                child1.method_weights[key] = parent1.method_weights[key]
                child2.method_weights[key] = parent1.method_weights[key]
            else:
                child1.method_weights[key] = parent2.method_weights[key]
                child2.method_weights[key] = parent2.method_weights[key]

    return child1, child2

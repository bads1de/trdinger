"""
遺伝子突然変異ユーティリティ
"""
from __future__ import annotations

import random
from typing import Dict, List

from ..utils.gene_utils import GeneticUtils
from .tpsl_gene import TPSLGene
from .position_sizing_gene import PositionSizingGene


def mutate_position_sizing_gene(
    gene: PositionSizingGene, mutation_rate: float = 0.1
) -> PositionSizingGene:
    """ポジションサイジング遺伝子の突然変異（ジェネリック関数使用）"""

    # フィールドルール定義
    numeric_fields: List[str] = [
        "lookback_period",
        "optimal_f_multiplier",
        "atr_multiplier",
        "risk_per_trade",
        "fixed_ratio",
        "fixed_quantity",
        "min_position_size",
        "max_position_size",
        "priority",
        "atr_period",
    ]

    enum_fields = ["method"]

    # 各フィールドの許容範囲
    numeric_ranges: Dict[str, tuple[float, float]] = {
        "lookback_period": (50, 200),
        "optimal_f_multiplier": (0.25, 0.75),
        "atr_multiplier": (0.1, 5.0),
        "risk_per_trade": (0.001, 0.1),
        "fixed_ratio": (0.001, 1.0),
        "fixed_quantity": (0.1, 10.0),
        "min_position_size": (0.001, 0.1),
        "max_position_size": (5.0, 50.0),
        "priority": (0.5, 1.5),
        "atr_period": (10, 30),
    }

    return GeneticUtils.mutate_generic_gene(
        gene=gene,
        gene_class=PositionSizingGene,
        mutation_rate=mutation_rate,
        numeric_fields=numeric_fields,
        enum_fields=enum_fields,
        numeric_ranges=numeric_ranges,
    )


def mutate_tpsl_gene(gene: TPSLGene, mutation_rate: float = 0.1) -> TPSLGene:
    """TP/SL遺伝子の突然変異（ジェネリック関数使用）"""

    # 基本フィールド
    numeric_fields: List[str] = [
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

    # 各フィールドの許容範囲
    numeric_ranges: Dict[str, tuple[float, float]] = {
        "stop_loss_pct": (0.005, 0.15),  # 0.5%-15%
        "take_profit_pct": (0.01, 0.3),  # 1%-30%
        "risk_reward_ratio": (1.0, 10.0),  # 1:10まで
        "base_stop_loss": (0.01, 0.06),
        "atr_multiplier_sl": (0.5, 3.0),
        "atr_multiplier_tp": (1.0, 5.0),
        "confidence_threshold": (0.1, 0.9),
        "priority": (0.5, 1.5),
        "lookback_period": (50, 200),
        "atr_period": (10, 30),
    }

    # ジェネリック突然変異を実行
    mutated_gene = GeneticUtils.mutate_generic_gene(
        gene=gene,
        gene_class=TPSLGene,
        mutation_rate=mutation_rate,
        numeric_fields=numeric_fields,
        enum_fields=enum_fields,
        numeric_ranges=numeric_ranges,
    )

    # method_weightsの突然変異（辞書フィールドの特殊処理）
    if random.random() < mutation_rate:
        # method_weightsを乱数で調整
        for key in mutated_gene.method_weights:
            current_weight = mutated_gene.method_weights[key]
            # 現在の値を中心とした範囲で変動
            mutated_gene.method_weights[key] = current_weight * random.uniform(0.8, 1.2)

        # 合計が1.0になるよう正規化
        total_weight = sum(mutated_gene.method_weights.values())
        if total_weight > 0:
            for key in mutated_gene.method_weights:
                mutated_gene.method_weights[key] /= total_weight

    return mutated_gene

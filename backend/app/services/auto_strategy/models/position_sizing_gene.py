"""
ポジションサイジング遺伝子
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..utils.gene_utils import BaseGene
from ..config.enums import PositionSizingMethod


@dataclass
class PositionSizingGene(BaseGene):
    """
    ポジションサイジング遺伝子

    GA最適化対象としてのポジションサイジング設定を表現します。
    BaseGeneを継承して共通機能を活用します。
    """

    method: PositionSizingMethod = PositionSizingMethod.VOLATILITY_BASED
    lookback_period: int = 100
    optimal_f_multiplier: float = 0.5
    atr_period: int = 14
    atr_multiplier: float = 2.0
    risk_per_trade: float = 0.02
    fixed_ratio: float = 0.1
    fixed_quantity: float = 1.0
    min_position_size: float = 0.01
    max_position_size: float = 9999.0
    var_confidence: float = 0.95
    max_var_ratio: float = 0.02
    max_expected_shortfall_ratio: float = 0.03
    var_lookback: int = 100
    enabled: bool = True
    priority: float = 1.0

    def _validate_parameters(self, errors: List[str]) -> None:
        """パラメータ固有の検証を実装"""
        try:
            from ..config.constants import POSITION_SIZING_LIMITS

            lb_min, lb_max = POSITION_SIZING_LIMITS["lookback_period"]
            if not (lb_min <= self.lookback_period <= lb_max):
                errors.append(
                    f"lookback_periodは{lb_min}-{lb_max}の範囲である必要があります"
                )

            # 他のパラメータ検証も実装可能
            self._validate_range(
                self.risk_per_trade, 0.001, 0.1, "risk_per_trade", errors
            )
            self._validate_range(self.fixed_ratio, 0.001, 1.0, "fixed_ratio", errors)
            self._validate_range(
                self.atr_multiplier, 0.1, 5.0, "atr_multiplier", errors
            )
            self._validate_range(
                self.var_confidence,
                POSITION_SIZING_LIMITS["var_confidence"][0],
                POSITION_SIZING_LIMITS["var_confidence"][1],
                "var_confidence",
                errors,
            )
            self._validate_range(
                self.max_var_ratio,
                POSITION_SIZING_LIMITS["max_var_ratio"][0],
                POSITION_SIZING_LIMITS["max_var_ratio"][1],
                "max_var_ratio",
                errors,
            )
            self._validate_range(
                self.max_expected_shortfall_ratio,
                POSITION_SIZING_LIMITS["max_expected_shortfall_ratio"][0],
                POSITION_SIZING_LIMITS["max_expected_shortfall_ratio"][1],
                "max_expected_shortfall_ratio",
                errors,
            )
            self._validate_range(
                self.var_lookback,
                POSITION_SIZING_LIMITS["var_lookback"][0],
                POSITION_SIZING_LIMITS["var_lookback"][1],
                "var_lookback",
                errors,
            )

        except ImportError:
            # 定数が利用できない場合の基本検証
            if not (50 <= self.lookback_period <= 200):
                errors.append("lookback_periodは50-200の範囲である必要があります")

            # 基本的な範囲検証
            if not (0.001 <= self.risk_per_trade <= 0.1):
                errors.append("risk_per_tradeは0.001-0.1の範囲である必要があります")
            if not (0.001 <= self.fixed_ratio <= 1.0):
                errors.append("fixed_ratioは0.001-1.0の範囲である必要があります")
            if not (0.8 <= self.var_confidence <= 0.999):
                errors.append("var_confidenceは0.8-0.999の範囲である必要があります")
            if not (0.001 <= self.max_var_ratio <= 0.1):
                errors.append("max_var_ratioは0.001-0.1の範囲である必要があります")
            if not (0.001 <= self.max_expected_shortfall_ratio <= 0.2):
                errors.append(
                    "max_expected_shortfall_ratioは0.001-0.2の範囲である必要があります"
                )
            if not (20 <= self.var_lookback <= 1000):
                errors.append("var_lookbackは20-1000の範囲である必要があります")


def create_random_position_sizing_gene(config=None) -> PositionSizingGene:
    """ランダムなポジションサイジング遺伝子を生成"""
    import random

    method_choices = list(PositionSizingMethod)
    method = random.choice(method_choices)

    return PositionSizingGene(
        method=method,
        lookback_period=random.randint(50, 200),
        optimal_f_multiplier=random.uniform(0.25, 0.75),
        atr_period=random.randint(10, 30),
        atr_multiplier=random.uniform(1.0, 4.0),
        risk_per_trade=random.uniform(0.01, 0.05),
        fixed_ratio=random.uniform(0.05, 0.3),
        fixed_quantity=random.uniform(0.1, 10.0),
        min_position_size=random.uniform(0.01, 0.05),
        max_position_size=random.uniform(5.0, 50.0),
        enabled=True,
        priority=random.uniform(0.5, 1.5),
    )


def crossover_position_sizing_genes(
    parent1: PositionSizingGene, parent2: PositionSizingGene
) -> tuple[PositionSizingGene, PositionSizingGene]:
    """ポジションサイジング遺伝子の交叉（ジェネリック関数使用）"""
    from ..utils.gene_utils import GeneticUtils

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


def mutate_position_sizing_gene(
    gene: PositionSizingGene, mutation_rate: float = 0.1
) -> PositionSizingGene:
    """ポジションサイジング遺伝子の突然変異（ジェネリック関数使用）"""
    from typing import Dict
    from ..utils.gene_utils import GeneticUtils

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



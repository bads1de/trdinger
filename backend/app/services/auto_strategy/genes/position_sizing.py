"""
ポジションサイジング遺伝子
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, cast

from app.types import StrategyGeneDict

from ..config.constants import PositionSizingMethod
from .base_gene import BaseGene
from .gene_ranges import (
    POSITION_SIZING_GENERATION_RANGES,
    POSITION_SIZING_VALIDATION_RANGES,
)


def _ensure_position_size_bounds(params: dict[str, Any]) -> None:
    """min/max_position_size の論理整合性を補正する。"""
    min_size = params.get("min_position_size")
    max_size = params.get("max_position_size")
    if isinstance(min_size, (int, float)) and isinstance(max_size, (int, float)):
        if min_size > max_size:
            params["max_position_size"] = min_size


@dataclass(slots=True)
class PositionSizingGene(BaseGene):
    """
    ポジションサイジング遺伝子

    GA最適化対象としてのポジションサイジング設定を表現します。
    BaseGeneを継承して共通機能を活用します。
    """

    NUMERIC_FIELDS = [
        "lookback_period",
        "optimal_f_multiplier",
        "atr_multiplier",
        "risk_per_trade",
        "fixed_ratio",
        "fixed_quantity",
        "min_position_size",
        "max_position_size",
        "var_confidence",
        "max_var_ratio",
        "max_expected_shortfall_ratio",
        "var_lookback",
        "priority",
        "atr_period",
    ]
    ENUM_FIELDS = ["method"]
    CHOICE_FIELDS = ["enabled"]
    NUMERIC_RANGES = dict(POSITION_SIZING_VALIDATION_RANGES)

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

    @classmethod
    def from_dict(cls, data: StrategyGeneDict) -> PositionSizingGene:
        """辞書から復元。無効な method はデフォルトにフォールバック。"""
        cleaned = dict(data)
        method_raw = cleaned.get("method")
        if method_raw is not None:
            if isinstance(method_raw, str):
                try:
                    cleaned["method"] = PositionSizingMethod(method_raw)
                except ValueError:
                    cleaned["method"] = PositionSizingMethod.VOLATILITY_BASED
            else:
                cleaned["method"] = PositionSizingMethod.VOLATILITY_BASED
        from .base_gene import BaseGene

        base_from_dict = cast(Any, BaseGene.from_dict)
        return cast(PositionSizingGene, base_from_dict.__func__(cls, cleaned))

    def _validate_parameters(self, errors: list[str]) -> None:
        """パラメータ固有の検証を実装"""
        if not isinstance(self.method, PositionSizingMethod):
            errors.append(  # type: ignore[unreachable]
                "methodは有効なPositionSizingMethodである必要があります"
            )

        # NUMERIC_RANGESを使用して検証（config非依存）
        for field_name, (min_val, max_val) in self.NUMERIC_RANGES.items():
            value = getattr(self, field_name, None)
            if value is not None:
                self._validate_range(value, min_val, max_val, field_name, errors)

        if self.min_position_size > self.max_position_size:
            errors.append(
                "min_position_sizeはmax_position_size以下である必要があります"
            )

    def mutate(
        self,
        mutation_rate: float = 0.1,
    ) -> PositionSizingGene:
        """突然変異する。パラメータはGAが自動的に最適化する。"""
        from .genetic_utils import GeneticUtils

        mutated_gene = cast(
            PositionSizingGene,
            GeneticUtils.mutate_generic_gene(
                gene=self,
                gene_class=self.__class__,
                mutation_rate=mutation_rate,
                numeric_fields=self.NUMERIC_FIELDS,
                enum_fields=self.ENUM_FIELDS,
                choice_fields=self.CHOICE_FIELDS,
                numeric_ranges=self.NUMERIC_RANGES,
            ),
        )
        if mutated_gene.min_position_size > mutated_gene.max_position_size:
            mutated_gene.max_position_size = mutated_gene.min_position_size
        return mutated_gene

    def clone(self) -> PositionSizingGene:
        """軽量コピーを作成"""
        return PositionSizingGene(
            method=self.method,
            lookback_period=self.lookback_period,
            optimal_f_multiplier=self.optimal_f_multiplier,
            atr_period=self.atr_period,
            atr_multiplier=self.atr_multiplier,
            risk_per_trade=self.risk_per_trade,
            fixed_ratio=self.fixed_ratio,
            fixed_quantity=self.fixed_quantity,
            min_position_size=self.min_position_size,
            max_position_size=self.max_position_size,
            var_confidence=self.var_confidence,
            max_var_ratio=self.max_var_ratio,
            max_expected_shortfall_ratio=self.max_expected_shortfall_ratio,
            var_lookback=self.var_lookback,
            enabled=self.enabled,
            priority=self.priority,
        )


def create_random_position_sizing_gene() -> PositionSizingGene:
    """ランダムなポジションサイジング遺伝子を生成"""
    method = random.choice(list(PositionSizingMethod))
    ranges = POSITION_SIZING_GENERATION_RANGES

    gene_params = {
        "method": method,
        "lookback_period": random.randint(
            int(ranges["lookback_period"][0]),
            int(ranges["lookback_period"][1]),
        ),
        "optimal_f_multiplier": random.uniform(
            ranges["optimal_f_multiplier"][0],
            ranges["optimal_f_multiplier"][1],
        ),
        "atr_period": random.randint(
            int(ranges["atr_period"][0]),
            int(ranges["atr_period"][1]),
        ),
        "atr_multiplier": random.uniform(
            ranges["atr_multiplier"][0],
            ranges["atr_multiplier"][1],
        ),
        "risk_per_trade": random.uniform(
            ranges["risk_per_trade"][0],
            ranges["risk_per_trade"][1],
        ),
        "fixed_ratio": random.uniform(
            ranges["fixed_ratio"][0],
            ranges["fixed_ratio"][1],
        ),
        "fixed_quantity": random.uniform(
            ranges["fixed_quantity"][0],
            ranges["fixed_quantity"][1],
        ),
        "min_position_size": random.uniform(
            ranges["min_position_size"][0],
            ranges["min_position_size"][1],
        ),
        "max_position_size": random.uniform(
            ranges["max_position_size"][0],
            ranges["max_position_size"][1],
        ),
        "var_confidence": random.uniform(
            ranges["var_confidence"][0],
            ranges["var_confidence"][1],
        ),
        "max_var_ratio": random.uniform(
            ranges["max_var_ratio"][0],
            ranges["max_var_ratio"][1],
        ),
        "max_expected_shortfall_ratio": random.uniform(
            ranges["max_expected_shortfall_ratio"][0],
            ranges["max_expected_shortfall_ratio"][1],
        ),
        "var_lookback": random.randint(
            int(ranges["var_lookback"][0]),
            int(ranges["var_lookback"][1]),
        ),
        "enabled": True,
        "priority": random.uniform(ranges["priority"][0], ranges["priority"][1]),
    }

    _ensure_position_size_bounds(gene_params)
    return PositionSizingGene(**gene_params)  # type: ignore[arg-type]

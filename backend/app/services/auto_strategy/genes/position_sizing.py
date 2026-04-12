"""
ポジションサイジング遺伝子
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..config.constants import (
    GA_DEFAULT_POSITION_SIZING_METHOD_CONSTRAINTS,
    GA_POSITION_SIZING_ATR_MULTIPLIER_RANGE,
    GA_POSITION_SIZING_ATR_PERIOD_RANGE,
    GA_POSITION_SIZING_FIXED_QUANTITY_RANGE,
    GA_POSITION_SIZING_FIXED_RATIO_RANGE,
    GA_POSITION_SIZING_LOOKBACK_RANGE,
    GA_POSITION_SIZING_MAX_ES_RATIO_RANGE,
    GA_POSITION_SIZING_MAX_SIZE_RANGE,
    GA_POSITION_SIZING_MAX_VAR_RATIO_RANGE,
    GA_POSITION_SIZING_MIN_SIZE_RANGE,
    GA_POSITION_SIZING_OPTIMAL_F_MULTIPLIER_RANGE,
    GA_POSITION_SIZING_PRIORITY_RANGE,
    GA_POSITION_SIZING_RISK_PER_TRADE_RANGE,
    GA_POSITION_SIZING_VAR_CONFIDENCE_RANGE,
    GA_POSITION_SIZING_VAR_LOOKBACK_RANGE,
    PositionSizingMethod,
)
from .base_gene import BaseGene

_POSITION_SIZING_NUMERIC_RANGE_ATTRS = {
    "lookback_period": "position_sizing_lookback_range",
    "optimal_f_multiplier": "position_sizing_optimal_f_multiplier_range",
    "atr_period": "position_sizing_atr_period_range",
    "atr_multiplier": "position_sizing_atr_multiplier_range",
    "risk_per_trade": "position_sizing_risk_per_trade_range",
    "fixed_ratio": "position_sizing_fixed_ratio_range",
    "fixed_quantity": "position_sizing_fixed_quantity_range",
    "min_position_size": "position_sizing_min_size_range",
    "max_position_size": "position_sizing_max_size_range",
    "var_confidence": "position_sizing_var_confidence_range",
    "max_var_ratio": "position_sizing_max_var_ratio_range",
    "max_expected_shortfall_ratio": "position_sizing_max_expected_shortfall_ratio_range",
    "var_lookback": "position_sizing_var_lookback_range",
    "priority": "position_sizing_priority_range",
}

_DEFAULT_POSITION_SIZING_NUMERIC_RANGES = {
    "lookback_period": tuple(GA_POSITION_SIZING_LOOKBACK_RANGE),
    "optimal_f_multiplier": tuple(GA_POSITION_SIZING_OPTIMAL_F_MULTIPLIER_RANGE),
    "atr_period": tuple(GA_POSITION_SIZING_ATR_PERIOD_RANGE),
    "atr_multiplier": tuple(GA_POSITION_SIZING_ATR_MULTIPLIER_RANGE),
    "risk_per_trade": tuple(GA_POSITION_SIZING_RISK_PER_TRADE_RANGE),
    "fixed_ratio": tuple(GA_POSITION_SIZING_FIXED_RATIO_RANGE),
    "fixed_quantity": tuple(GA_POSITION_SIZING_FIXED_QUANTITY_RANGE),
    "min_position_size": tuple(GA_POSITION_SIZING_MIN_SIZE_RANGE),
    "max_position_size": tuple(GA_POSITION_SIZING_MAX_SIZE_RANGE),
    "var_confidence": tuple(GA_POSITION_SIZING_VAR_CONFIDENCE_RANGE),
    "max_var_ratio": tuple(GA_POSITION_SIZING_MAX_VAR_RATIO_RANGE),
    "max_expected_shortfall_ratio": tuple(GA_POSITION_SIZING_MAX_ES_RATIO_RANGE),
    "var_lookback": tuple(GA_POSITION_SIZING_VAR_LOOKBACK_RANGE),
    "priority": tuple(GA_POSITION_SIZING_PRIORITY_RANGE),
}


def _coerce_numeric_range(
    configured_range: Any,
    fallback_range: tuple[float, float],
) -> tuple[float, float]:
    """設定レンジを検証し、不正値はデフォルトへフォールバックする。"""
    if not isinstance(configured_range, (list, tuple)) or len(configured_range) != 2:
        return fallback_range

    lower, upper = configured_range
    if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
        return fallback_range
    if lower > upper:
        return fallback_range

    return (lower, upper)


def resolve_position_sizing_numeric_ranges(
    config: Any = None,
) -> Dict[str, tuple[float, float]]:
    """PositionSizingGene の探索レンジを config 付きで解決する。"""
    ranges = dict(_DEFAULT_POSITION_SIZING_NUMERIC_RANGES)
    if config is None:
        return ranges

    for field_name, attr_name in _POSITION_SIZING_NUMERIC_RANGE_ATTRS.items():
        configured_range = getattr(config, attr_name, None)
        if configured_range is None:
            continue
        ranges[field_name] = _coerce_numeric_range(
            configured_range,
            ranges[field_name],
        )

    return ranges


def resolve_position_sizing_methods(
    config: Any = None,
) -> List[PositionSizingMethod]:
    """PositionSizingGene の許可メソッド一覧を config から解決する。"""
    configured_methods = getattr(config, "position_sizing_method_constraints", None)
    if not isinstance(configured_methods, (list, tuple, set)) or not configured_methods:
        configured_methods = GA_DEFAULT_POSITION_SIZING_METHOD_CONSTRAINTS

    methods: List[PositionSizingMethod] = []
    for method in configured_methods:
        try:
            methods.append(
                method
                if isinstance(method, PositionSizingMethod)
                else PositionSizingMethod(method)
            )
        except ValueError:
            continue

    return methods or list(PositionSizingMethod)


def _ensure_position_size_bounds(params: Dict[str, Any]) -> None:
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
    NUMERIC_RANGES = dict(_DEFAULT_POSITION_SIZING_NUMERIC_RANGES)

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

            if not isinstance(self.method, PositionSizingMethod):
                errors.append(
                    "methodは有効なPositionSizingMethodである必要があります"
                )

            lb_min, lb_max = POSITION_SIZING_LIMITS["lookback_period"]
            if not (lb_min <= self.lookback_period <= lb_max):
                errors.append(
                    f"lookback_periodは{lb_min}-{lb_max}の範囲である必要があります"
                )

            # 他のパラメータ検証も実装可能
            self._validate_range(
                self.risk_per_trade,
                POSITION_SIZING_LIMITS["risk_per_trade"][0],
                POSITION_SIZING_LIMITS["risk_per_trade"][1],
                "risk_per_trade",
                errors,
            )
            self._validate_range(
                self.fixed_ratio,
                POSITION_SIZING_LIMITS["fixed_ratio"][0],
                POSITION_SIZING_LIMITS["fixed_ratio"][1],
                "fixed_ratio",
                errors,
            )
            self._validate_range(
                self.atr_multiplier,
                POSITION_SIZING_LIMITS["atr_multiplier"][0],
                POSITION_SIZING_LIMITS["atr_multiplier"][1],
                "atr_multiplier",
                errors,
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
            self._validate_range(
                self.optimal_f_multiplier,
                POSITION_SIZING_LIMITS["optimal_f_multiplier"][0],
                POSITION_SIZING_LIMITS["optimal_f_multiplier"][1],
                "optimal_f_multiplier",
                errors,
            )
            self._validate_range(
                self.atr_period,
                POSITION_SIZING_LIMITS["atr_period"][0],
                POSITION_SIZING_LIMITS["atr_period"][1],
                "atr_period",
                errors,
            )
            self._validate_range(
                self.fixed_quantity,
                POSITION_SIZING_LIMITS["fixed_quantity"][0],
                POSITION_SIZING_LIMITS["fixed_quantity"][1],
                "fixed_quantity",
                errors,
            )
            self._validate_range(
                self.min_position_size,
                POSITION_SIZING_LIMITS["min_position_size"][0],
                POSITION_SIZING_LIMITS["min_position_size"][1],
                "min_position_size",
                errors,
            )
            self._validate_range(
                self.max_position_size,
                POSITION_SIZING_LIMITS["max_position_size"][0],
                POSITION_SIZING_LIMITS["max_position_size"][1],
                "max_position_size",
                errors,
            )
            if self.min_position_size > self.max_position_size:
                errors.append(
                    "min_position_sizeはmax_position_size以下である必要があります"
                )

        except ImportError:
            # 定数が利用できない場合の基本検証
            if not isinstance(self.method, PositionSizingMethod):
                errors.append(
                    "methodは有効なPositionSizingMethodである必要があります"
                )

            if not (50 <= self.lookback_period <= 200):
                errors.append("lookback_periodは50-200の範囲である必要があります")

            # 基本的な範囲検証
            if not (0.001 <= self.risk_per_trade <= 0.1):
                errors.append("risk_per_tradeは0.001-0.1の範囲である必要があります")
            if not (0.01 <= self.fixed_ratio <= 10.0):
                errors.append("fixed_ratioは0.01-10.0の範囲である必要があります")
            if not (0.5 <= self.atr_multiplier <= 10.0):
                errors.append("atr_multiplierは0.5-10.0の範囲である必要があります")
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
            if not (0.1 <= self.optimal_f_multiplier <= 1.0):
                errors.append("optimal_f_multiplierは0.1-1.0の範囲である必要があります")
            if not (5 <= self.atr_period <= 50):
                errors.append("atr_periodは5-50の範囲である必要があります")
            if not (0.01 <= self.fixed_quantity <= 1000.0):
                errors.append("fixed_quantityは0.01-1000.0の範囲である必要があります")
            if not (0.001 <= self.min_position_size <= 1.0):
                errors.append("min_position_sizeは0.001-1.0の範囲である必要があります")
            if not (0.001 <= self.max_position_size <= 1000000000.0):
                errors.append(
                    "max_position_sizeは0.001-1000000000.0の範囲である必要があります"
                )
            if self.min_position_size > self.max_position_size:
                errors.append(
                    "min_position_sizeはmax_position_size以下である必要があります"
                )

    def mutate(
        self,
        mutation_rate: float = 0.1,
        config: Any = None,
    ) -> PositionSizingGene:
        """config 連動の探索レンジを使って突然変異する。"""
        import random

        from .genetic_utils import GeneticUtils

        numeric_ranges = resolve_position_sizing_numeric_ranges(config)
        allowed_methods = resolve_position_sizing_methods(config)
        mutated_params = GeneticUtils.extract_gene_params(self)

        for field_name in self.NUMERIC_FIELDS:
            if random.random() >= mutation_rate or field_name not in mutated_params:
                continue

            current_value = mutated_params[field_name]
            if not isinstance(current_value, (int, float)):
                continue

            min_val, max_val = numeric_ranges.get(field_name, (0.0, 100.0))
            new_value = current_value * random.uniform(0.8, 1.2)
            new_value = max(min_val, min(max_val, new_value))
            if isinstance(current_value, int):
                new_value = int(new_value)
            mutated_params[field_name] = new_value

        if random.random() < mutation_rate:
            mutated_params["method"] = random.choice(allowed_methods)

        _ensure_position_size_bounds(mutated_params)
        return PositionSizingGene(**mutated_params)  # type: ignore[arg-type]

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


def create_random_position_sizing_gene(config=None) -> PositionSizingGene:
    """ランダムなポジションサイジング遺伝子を生成"""
    import random

    method = random.choice(resolve_position_sizing_methods(config))
    ranges = resolve_position_sizing_numeric_ranges(config)

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
    return PositionSizingGene(
        **gene_params
    )

"""
fitness 値の抽出を共通化するユーティリティ。
"""

from __future__ import annotations

from typing import Any


def extract_individual_primary_fitness(
    individual: Any,
    *,
    default: float = float("-inf"),
) -> float:
    """個体から単一目的の主 fitness を取り出す。"""
    fitness = getattr(individual, "fitness", None)
    if fitness is None:
        return float(default)

    for attr_name in ("wvalues", "values"):
        values = getattr(fitness, attr_name, ())
        if not isinstance(values, (tuple, list)) or not values:
            continue
        try:
            return float(values[0])
        except (TypeError, ValueError):
            continue
    return float(default)


def extract_primary_fitness_from_result(
    result: Any,
    *,
    default: float = 0.0,
) -> float:
    """評価結果から主 fitness を抽出する。"""
    if isinstance(result, (tuple, list)) and result:
        try:
            return float(result[0])
        except (TypeError, ValueError):
            return float(default)
    if isinstance(result, (int, float)):
        return float(result)
    return float(default)


def extract_result_fitness(
    individual: Any,
    *,
    enable_multi_objective: bool,
    default_single: float = 0.0,
) -> Any:
    """結果出力用に個体の fitness を整形して返す。"""
    fitness = getattr(individual, "fitness", None)
    values = getattr(fitness, "values", ()) if fitness is not None else ()

    if enable_multi_objective:
        if isinstance(values, (tuple, list)):
            return tuple(values)
        return ()

    if isinstance(values, (tuple, list)) and values:
        try:
            return float(values[0])
        except (TypeError, ValueError):
            return float(default_single)
    return float(default_single)

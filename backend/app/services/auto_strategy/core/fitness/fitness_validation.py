"""
フィットネス共有の共通ユーティリティ
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from app.services.auto_strategy.config import objective_registry
from app.services.auto_strategy.config.constants import PENALTY_FITNESS_MAGNITUDE


def has_valid_fitness(individual: Any) -> bool:
    """個体が有効なフィットネス値を持つか判定する。"""
    return hasattr(individual, "fitness") and individual.fitness.valid


def resolve_minimize_flags(
    objectives: Sequence[str] | None,
    n_objectives: int,
) -> list[bool]:
    """目的ごとの最小化フラグ（True=最小化）を解決する。

    Args:
        objectives: 目的関数名のシーケンス。None の場合はレジストリを
            参照できないため、最大化扱い（False）とする。
        n_objectives: 適応度値の要素数。

    Returns:
        各要素が最小化目的かどうかのフラグ列。
    """
    if objectives is None:
        return [False] * n_objectives
    flags = [objective_registry.is_minimize_objective(obj) for obj in objectives]
    if len(flags) < n_objectives:
        flags.extend([False] * (n_objectives - len(flags)))
    return flags[:n_objectives]


def compute_worst_selection_values(
    population: Sequence[Any],
    minimize_flags: Sequence[bool],
) -> list[float | None]:
    """目的ごとに、選択空間（大きいほど良い）での集団内最悪値を求める。

    評価失敗ペナルティ値（±PENALTY_FITNESS_MAGNITUDE）と非有限値は
    基準値の計算から除外する（ペナルティを基準にすると通常個体の
    優位幅が巨大化し、共有調整が機能しなくなるため）。
    """
    n = len(minimize_flags)
    worst: list[float | None] = [None] * n
    for individual in population:
        if not has_valid_fitness(individual):
            continue
        values = getattr(individual.fitness, "values", ())
        for j in range(min(n, len(values))):
            value = values[j]
            if not math.isfinite(value) or abs(value) >= PENALTY_FITNESS_MAGNITUDE:
                continue
            selection_value = -value if minimize_flags[j] else value
            if worst[j] is None or selection_value < worst[j]:
                worst[j] = selection_value
    return worst


def shrink_advantage_toward_worst(
    value: float,
    worst_selection_value: float | None,
    is_minimize: bool,
    factor: float,
) -> float:
    """選択空間で「最悪値からの優位幅」を縮小する。

    生値の除算・乗算による共有調整は値の符号と目的の方向（最小化）の
    組み合わせによって、0 へ寄せる操作が「改善」になり混雑個体を
    報酬してしまう。そこで減衰は常に減算系で行う:

    ``adjusted = worst + (selection_value - worst) * factor``

    factor < 1 なら必ず元の値より悪化し、方向・符号に依存せず
    単調に機能する。

    Args:
        value: 元の（生の）適応度値。
        worst_selection_value: 同目的の集団内最悪値（選択空間）。
            None の場合は調整しない。
        is_minimize: 最小化目的かどうか。
        factor: 優位幅に掛ける係数（0 < factor <= 1）。

    Returns:
        調整後の生の適応度値。
    """
    if worst_selection_value is None or not math.isfinite(value):
        return value
    if abs(value) >= PENALTY_FITNESS_MAGNITUDE:
        # 評価失敗個体はすでに最下位。基準にも干渉させない。
        return value
    selection_value = -value if is_minimize else value
    adjusted = (
        worst_selection_value + (selection_value - worst_selection_value) * factor
    )
    return -adjusted if is_minimize else adjusted

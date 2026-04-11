"""Registry of objective direction metadata used across auto-strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

ObjectiveDirection = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class ObjectiveDefinition:
    """
    Metadata for a single objective.

    単一の目的関数のメタデータを定義します。
    最適化の方向（最大化・最小化）や動的スケーリングの有無を指定します。

    Attributes:
        name: 目的関数名
        direction: 最適化方向（'maximize' または 'minimize'）
        dynamic_scalar: 動的スケーリングを適用するか（デフォルト: False）
    """

    name: str
    direction: ObjectiveDirection = "maximize"
    dynamic_scalar: bool = False


OBJECTIVE_REGISTRY: Final[dict[str, ObjectiveDefinition]] = {
    "weighted_score": ObjectiveDefinition(name="weighted_score"),
    "total_return": ObjectiveDefinition(name="total_return"),
    "sharpe_ratio": ObjectiveDefinition(name="sharpe_ratio"),
    "max_drawdown": ObjectiveDefinition(
        name="max_drawdown",
        direction="minimize",
        dynamic_scalar=True,
    ),
    "win_rate": ObjectiveDefinition(name="win_rate"),
    "profit_factor": ObjectiveDefinition(name="profit_factor"),
    "sortino_ratio": ObjectiveDefinition(name="sortino_ratio"),
    "calmar_ratio": ObjectiveDefinition(name="calmar_ratio"),
    "balance_score": ObjectiveDefinition(name="balance_score"),
    "ulcer_index": ObjectiveDefinition(
        name="ulcer_index",
        direction="minimize",
        dynamic_scalar=True,
    ),
    "trade_frequency_penalty": ObjectiveDefinition(
        name="trade_frequency_penalty",
        direction="minimize",
        dynamic_scalar=True,
    ),
}

DEFAULT_OBJECTIVE_DEFINITION: Final[ObjectiveDefinition] = ObjectiveDefinition(
    name="unknown_objective"
)

MINIMIZE_OBJECTIVES: Final[frozenset[str]] = frozenset(
    name
    for name, definition in OBJECTIVE_REGISTRY.items()
    if definition.direction == "minimize"
)

DYNAMIC_SCALAR_OBJECTIVES: Final[frozenset[str]] = frozenset(
    name
    for name, definition in OBJECTIVE_REGISTRY.items()
    if definition.dynamic_scalar
)


def get_objective_definition(objective: Any) -> ObjectiveDefinition:
    """
    Return the registry entry for an objective, or the default definition.

    指定された目的関数の定義をレジストリから取得します。
    見つからない場合はデフォルト定義を返します。

    Args:
        objective: 目的関数名（文字列、またはNone）

    Returns:
        ObjectiveDefinition: 目的関数定義オブジェクト

    Note:
        Noneまたは空文字の場合はデフォルト定義を返します。
    """
    if objective in (None, ""):
        return DEFAULT_OBJECTIVE_DEFINITION
    return OBJECTIVE_REGISTRY.get(str(objective), DEFAULT_OBJECTIVE_DEFINITION)


def is_minimize_objective(objective: Any) -> bool:
    """
    Return True when the objective should be minimized.

    指定された目的関数が最小化すべき指標かどうかを確認します。

    Args:
        objective: 目的関数名

    Returns:
        bool: 最小化すべき場合はTrue、最大化すべき場合はFalse
    """
    return get_objective_definition(objective).direction == "minimize"


def is_dynamic_scalar_objective(objective: Any) -> bool:
    """
    Return True when dynamic scalar reweighting should apply.

    指定された目的関数に動的スケーリングを適用すべきかどうかを確認します。
    動的スケーリングは、目的関数の値の範囲に基づいて重みを調整するために使用されます。

    Args:
        objective: 目的関数名

    Returns:
        bool: 動的スケーリングを適用すべき場合はTrue、そうでない場合はFalse
    """
    return get_objective_definition(objective).dynamic_scalar


def to_selection_space(value: float, objective: Any) -> float:
    """
    Convert an objective value into the selection space used by ranking.

    目的関数の値をランキングに使用される選択空間に変換します。
    最小化すべき目的関数の場合は値を反転します。

    Args:
        value: 目的関数の値
        objective: 目的関数名

    Returns:
        float: 選択空間に変換された値

    変換ルール:
        - 最小化目的関数: -value
        - 最大化目的関数: value
    """
    numeric_value = float(value)
    return -numeric_value if is_minimize_objective(objective) else numeric_value

"""Registry of objective direction metadata used across auto-strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

ObjectiveDirection = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class ObjectiveDefinition:
    """Metadata for a single objective."""

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
    """Return the registry entry for an objective, or the default definition."""
    if objective in (None, ""):
        return DEFAULT_OBJECTIVE_DEFINITION
    return OBJECTIVE_REGISTRY.get(str(objective), DEFAULT_OBJECTIVE_DEFINITION)


def is_minimize_objective(objective: Any) -> bool:
    """Return True when the objective should be minimized."""
    return get_objective_definition(objective).direction == "minimize"


def is_dynamic_scalar_objective(objective: Any) -> bool:
    """Return True when dynamic scalar reweighting should apply."""
    return get_objective_definition(objective).dynamic_scalar


def to_selection_space(value: float, objective: Any) -> float:
    """Convert an objective value into the selection space used by ranking."""
    numeric_value = float(value)
    return -numeric_value if is_minimize_objective(objective) else numeric_value

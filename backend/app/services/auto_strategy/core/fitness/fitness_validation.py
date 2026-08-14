"""
フィットネス共有の共通ユーティリティ
"""

from __future__ import annotations

from typing import Any


def has_valid_fitness(individual: Any) -> bool:
    """個体が有効なフィットネス値を持つか判定する。"""
    return hasattr(individual, "fitness") and individual.fitness.valid

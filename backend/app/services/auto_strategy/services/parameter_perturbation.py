"""パラメータ摂動変異体の生成

合格候補の指標パラメータを微小に揺らした変異遺伝子を作る。
「パラメータの針の山」に過適合した戦略（±10% の変更で成績が崩壊する
戦略）を検出するための感度分析用。

設計:
- 数値パラメータ（int/float、bool除く）のみ対象
- 一度に1パラメータだけ変更する One-At-a-Time 方式（影響の切り分けが可能）
- 条件式から参照されない「死んだ」指標は揺らしても挙動が変わらないため除外
- 生成数は max_variants で打ち止め（均等に間引いて決定論的に選ぶ）
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from ..utils.indicator_references import (
    collect_indicator_references,
    is_dead_indicator,
)

logger = logging.getLogger(__name__)


def collect_alive_indicator_genes(gene: Any) -> list[Any]:
    """条件式から参照されている有効な指標遺伝子のみ返す。"""
    condition_groups = (
        getattr(gene, "long_entry_conditions", None),
        getattr(gene, "short_entry_conditions", None),
        getattr(gene, "long_exit_conditions", None),
        getattr(gene, "short_exit_conditions", None),
    )
    referenced: set[str] = set()
    for group in condition_groups:
        collect_indicator_references(group, referenced)

    return [
        indicator
        for indicator in getattr(gene, "indicators", None) or []
        if not is_dead_indicator(indicator, referenced)
    ]


def _perturb_value(value: Any, direction: int, delta: float) -> Any | None:
    """単一の数値パラメータを ±delta 比率で揺らす。対象外は None。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    perturbed = value * (1.0 + direction * delta)
    if isinstance(value, int):
        perturbed_int = max(1, round(perturbed))
        if perturbed_int == value:
            return None
        return int(perturbed_int)
    # float: 非正・無変化は除外
    if not perturbed > 0.0 or perturbed == value:
        return None
    return float(perturbed)


def _evenly_spread_indices(total: int, limit: int) -> list[int]:
    """0..total-1 から limit 個を決定論的かつ均等に選ぶ。"""
    if total <= limit:
        return list(range(total))
    step = total / limit
    return sorted({min(total - 1, int(i * step)) for i in range(limit)})


def build_perturbed_genes(
    gene: Any,
    delta: float,
    max_variants: int,
) -> list[tuple[str, Any]]:
    """摂動変異遺伝子のリストを返す。

    Returns:
        list[(label, variant_gene)]: label は人間可読な差分説明
            （例: "BBANDS.length: 20 -> 22"）。
    """
    if max_variants <= 0 or delta <= 0.0:
        return []

    alive = collect_alive_indicator_genes(gene)

    # (alive内の指標インデックス, パラメータキー, 現在値, 方向) の候補を収集
    candidates: list[tuple[int, str, Any, int]] = []
    for ind_index, indicator in enumerate(alive):
        parameters = getattr(indicator, "parameters", None)
        if not isinstance(parameters, dict):
            continue
        for key in sorted(parameters):
            for direction in (+1, -1):
                candidates.append((ind_index, key, parameters[key], direction))

    if not candidates:
        return []

    chosen = [
        candidates[i] for i in _evenly_spread_indices(len(candidates), max_variants)
    ]

    variants: list[tuple[str, Any]] = []
    seen_labels: set[str] = set()
    for ind_index, key, original, direction in chosen:
        new_value = _perturb_value(original, direction, delta)
        if new_value is None:
            continue
        variant = copy.deepcopy(gene)
        target = collect_alive_indicator_genes(variant)[ind_index]
        target.parameters[key] = new_value
        label = f"{getattr(target, 'type', '?')}.{key}: {original} -> {new_value}"
        if label in seen_labels:
            continue
        seen_labels.add(label)
        variants.append((label, variant))

    return variants

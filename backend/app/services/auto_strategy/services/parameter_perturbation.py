"""パラメータ摂動変異体の生成

合格候補の数値パラメータを微小に揺らした変異遺伝子を作る。
「パラメータの針の山」に過適合した戦略（±10% の変更で成績が崩壊する
戦略）を検出するための感度分析用。

設計:
- 数値パラメータ（int/float、bool除く）のみ対象
- 対象コンポーネント:
    * 条件式から参照される有効な指標遺伝子の parameters
    * TPSL 遺伝子（base / long / short、enabled のみ）
    * ポジションサイジング遺伝子（enabled のみ）
- 一度に1パラメータだけ変更する One-At-a-Time 方式（影響の切り分けが可能）
- 条件式から参照されない「死んだ」指標は揺らしても挙動が変わらないため除外
- 生成数は max_variants で打ち止め（全候補から均等に間引いて決定論的に選ぶ）
"""

from __future__ import annotations

import copy
import dataclasses
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


def _numeric_dataclass_fields(obj: Any) -> list[str]:
    """オブジェクトの数値フィールド名（bool/dict等は除外）をソート済みで返す。

    dataclass 以外（SimpleNamespace 等のテストダブル）は __dict__ に
    フォールバックする。
    """
    if obj is None:
        return []
    try:
        fields = dataclasses.fields(obj)
        names = [f.name for f in fields]
    except TypeError:
        names = list(getattr(obj, "__dict__", {}).keys())
    result: list[str] = []
    for name in sorted(names):
        value = getattr(obj, name, None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        result.append(name)
    return result


# (source_kind, locator, key, original_value, direction)
Candidate = tuple[str, Any, str, Any, int]


def _collect_candidates(gene: Any) -> tuple[list[Any], list[Candidate]]:
    """摂動候補（指標＋TPSL＋サイジング）と指標リストを返す。"""
    alive = collect_alive_indicator_genes(gene)
    candidates: list[Candidate] = []

    for ind_index, indicator in enumerate(alive):
        parameters = getattr(indicator, "parameters", None)
        if not isinstance(parameters, dict):
            continue
        for key in sorted(parameters):
            for direction in (+1, -1):
                candidates.append(
                    ("indicator", ind_index, key, parameters[key], direction)
                )

    sub_gene_specs = (
        ("tpsl_gene", "TPSL"),
        ("long_tpsl_gene", "TPSL_LONG"),
        ("short_tpsl_gene", "TPSL_SHORT"),
        ("position_sizing_gene", "SIZING"),
    )
    for attr_name, prefix in sub_gene_specs:
        sub = getattr(gene, attr_name, None)
        if sub is None or not getattr(sub, "enabled", True):
            continue
        for field_name in _numeric_dataclass_fields(sub):
            value = getattr(sub, field_name)
            for direction in (+1, -1):
                candidates.append((attr_name, prefix, field_name, value, direction))

    return alive, candidates


def _apply_perturbation(
    variant: Any,
    kind: str,
    locator: Any,
    key: str,
    new_value: Any,
) -> str | None:
    """変異遺伝子へ摂動を適用し、ラベルを返す（失敗時 None）。"""
    if kind == "indicator":
        target = collect_alive_indicator_genes(variant)[locator]
        original = target.parameters[key]
        target.parameters[key] = new_value
        label_prefix = str(getattr(target, "type", "?"))
    else:
        target = getattr(variant, kind, None)
        if target is None:
            return None
        original = getattr(target, key)
        setattr(target, key, new_value)
        label_prefix = str(locator)
    return f"{label_prefix}.{key}: {original} -> {new_value}"


def build_perturbed_genes(
    gene: Any,
    delta: float,
    max_variants: int,
) -> list[tuple[str, Any]]:
    """摂動変異遺伝子のリストを返す。

    Returns:
        list[(label, variant_gene)]: label は人間可読な差分説明
            （例: "BBANDS.length: 20 -> 22" / "TPSL.stop_loss_pct: 0.03 -> 0.027"）。
    """
    if max_variants <= 0 or delta <= 0.0:
        return []

    alive, candidates = _collect_candidates(gene)
    if not candidates:
        return []

    chosen = [
        candidates[i] for i in _evenly_spread_indices(len(candidates), max_variants)
    ]

    variants: list[tuple[str, Any]] = []
    seen_labels: set[str] = set()
    for kind, locator, key, original, direction in chosen:
        new_value = _perturb_value(original, direction, delta)
        if new_value is None:
            continue
        variant = copy.deepcopy(gene)
        label = _apply_perturbation(variant, kind, locator, key, new_value)
        if label is None or label in seen_labels:
            continue
        seen_labels.add(label)
        variants.append((label, variant))

    return variants

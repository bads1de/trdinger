"""
戦略遺伝子の構造的 重複排除ユーティリティ

GAの最終集団には同じ構造の個体（クローン）が複数残ることがある。
そのまま DB へ保存すると「まったく同じ戦略」が並んで表示されるため、
id・metadata を除いた構造シグネチャで同一個体をまとめ、
シグネチャごとに最高適応度の 1 件だけを残す。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from typing import Any

from app.services.auto_strategy.genes import ConditionGroup


def _condition_signature(condition: Any) -> tuple:
    """条件（葉／グループ）を正規化したタグ付きタプルへ変換する。"""
    if isinstance(condition, ConditionGroup):
        return (
            "group",
            str(condition.operator),
            tuple(
                sorted(_condition_signature(sub) for sub in condition.conditions or [])
            ),
        )
    right_operand = getattr(condition, "right_operand", None)
    return (
        "cond",
        str(getattr(condition, "left_operand", None)),
        str(getattr(condition, "operator", None)),
        type(right_operand).__name__,
        str(right_operand),
    )


def _indicator_signature(indicator: Any) -> tuple:
    parameters = getattr(indicator, "parameters", None) or {}
    return (
        str(getattr(indicator, "type", "")),
        str(getattr(indicator, "timeframe", None) or ""),
        tuple(sorted((str(k), str(v)) for k, v in parameters.items())),
    )


def _mapping_signature(value: Any) -> tuple:
    items = value.items() if isinstance(value, dict) else ()
    return tuple(sorted((str(k), str(v)) for k, v in items))


def _object_fields_signature(obj: Any) -> tuple:
    """dataclass 等のインスタンスフィールドを正規化する。

    TPSLGene 等のサブ遺伝子は ``@dataclass(slots=True)`` で定義されており
    ``__dict__`` を持たないため、dataclass は ``dataclasses.fields()``
    でフィールドを列挙する（``__dict__`` だけ見ると全サブ遺伝子が空に
    崩れ、TP/SL・サイジング違いの戦略が同一構造と誤判定される）。
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        values = {f.name: getattr(obj, f.name, None) for f in dataclasses.fields(obj)}
    else:
        values = dict(getattr(obj, "__dict__", None) or {})
    return tuple(sorted((str(k), str(v)) for k, v in values.items()))


def _optional_object_signature(obj: Any) -> tuple:
    return () if obj is None else _object_fields_signature(obj)


def strategy_structure_signature(strategy: Any) -> tuple:
    """id・metadata を除いた戦略の構造のみから一意シグネチャを構築する。

    指標構成（タイプ・タイムフレーム・パラメータ）、全条件ツリー、
    リスク管理・TP/SL（共通・ロング・ショート）・ポジションサイジング、
    エントリー/イグジット遺伝子、stateful条件、ツール遺伝子を対象とする。
    """
    indicators = tuple(
        sorted(
            _indicator_signature(ind)
            for ind in getattr(strategy, "indicators", None) or []
        )
    )

    condition_signatures: list[tuple] = []
    for attr in (
        "long_entry_conditions",
        "short_entry_conditions",
        "long_exit_conditions",
        "short_exit_conditions",
    ):
        conditions = getattr(strategy, attr, None) or []
        condition_signatures.append(
            tuple(sorted(_condition_signature(cond) for cond in conditions))
        )

    stateful_conditions = tuple(
        sorted(
            _object_fields_signature(cond)
            for cond in getattr(strategy, "stateful_conditions", None) or []
        )
    )
    tool_genes = tuple(
        sorted(
            _object_fields_signature(tool)
            for tool in getattr(strategy, "tool_genes", None) or []
        )
    )

    return (
        indicators,
        tuple(condition_signatures),
        stateful_conditions,
        tool_genes,
        _mapping_signature(getattr(strategy, "risk_management", None) or {}),
        _optional_object_signature(getattr(strategy, "tpsl_gene", None)),
        _optional_object_signature(getattr(strategy, "long_tpsl_gene", None)),
        _optional_object_signature(getattr(strategy, "short_tpsl_gene", None)),
        _optional_object_signature(getattr(strategy, "position_sizing_gene", None)),
        _optional_object_signature(getattr(strategy, "entry_gene", None)),
        _optional_object_signature(getattr(strategy, "long_entry_gene", None)),
        _optional_object_signature(getattr(strategy, "short_entry_gene", None)),
        _optional_object_signature(getattr(strategy, "exit_gene", None)),
    )


def deduplicate_strategies(
    strategies: Sequence[Any],
    fitness_scores: Sequence[float],
    *,
    exclude: Iterable[Any] = (),
    max_count: int | None = None,
) -> list[tuple[Any, float]]:
    """構造が同一の戦略をまとめ、シグネチャごとに最高適応度の 1 件だけ残す。

    Args:
        strategies: 保存対象の戦略リスト（適応度降順を想定）。
        fitness_scores: strategies と同じ順序の適応度。
        exclude: これと同一構造の戦略を保存対象から除外する
            （最良戦略のクローンが「その他戦略」として重複保存されるのを防ぐ）。
        max_count: dedup 後の最大件数。

    Returns:
        (戦略, 適応度) のリスト。順序は入力における初出順。
    """
    excluded_signatures = {strategy_structure_signature(s) for s in exclude}

    best: dict[tuple, tuple[float, Any]] = {}
    first_seen_order: list[tuple] = []
    for index, strategy in enumerate(strategies):
        raw_score = fitness_scores[index] if index < len(fitness_scores) else 0.0
        score = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0

        signature = strategy_structure_signature(strategy)
        if signature in excluded_signatures:
            continue
        if signature not in best:
            first_seen_order.append(signature)
            best[signature] = (score, strategy)
        elif score > best[signature][0]:
            best[signature] = (score, strategy)

    deduplicated = [
        (best[signature][1], best[signature][0]) for signature in first_seen_order
    ]
    if max_count is not None:
        return deduplicated[:max_count]
    return deduplicated

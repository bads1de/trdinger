"""
フィットネス共有の特徴ベクトル化ユーティリティ
"""

from typing import Any, Mapping, Sequence

import numpy as np

from app.services.auto_strategy.genes import ConditionGroup, StrategyGene


def vectorize_gene(
    gene: StrategyGene,
    indicator_types: Sequence[str],
    indicator_map: Mapping[str, int],
    operator_types: Sequence[str],
    operator_map: Mapping[str, int],
) -> np.ndarray:
    """
    戦略遺伝子を固定長の特徴ベクトルに変換する。
    """
    features: list[float] = []

    # 指標数
    features.append(float(len(gene.indicators)))

    # 条件数
    features.append(float(len(gene.long_entry_conditions)))
    features.append(float(len(gene.short_entry_conditions)))

    # リスク管理パラメータ
    if gene.risk_management:
        features.append(float(gene.risk_management.get("position_size", 0.1)))
    else:
        features.append(0.1)

    # TP/SLパラメータ
    if gene.tpsl_gene:
        features.append(float(gene.tpsl_gene.stop_loss_pct or 0.05))
        features.append(float(gene.tpsl_gene.take_profit_pct or 0.1))
    else:
        features.append(0.05)
        features.append(0.1)

    # ポジションサイジングパラメータ
    if gene.position_sizing_gene and hasattr(gene.position_sizing_gene, "risk_per_trade"):
        features.append(float(gene.position_sizing_gene.risk_per_trade or 0.01))
    else:
        features.append(0.01)

    # 指標タイプベクトル（Bag of Words）
    if indicator_types:
        indicator_vector = np.zeros(len(indicator_types))
        for ind in gene.indicators:
            if ind.type in indicator_map:
                idx = indicator_map[ind.type]
                indicator_vector[idx] += 1.0

        features.extend(indicator_vector.tolist())

    # オペレータタイプベクトル（Bag of Words）
    all_conditions = _collect_conditions(gene)
    if operator_types:
        operator_vector = np.zeros(len(operator_types))

        _count_operators(all_conditions, operator_map, operator_vector)

        features.extend(operator_vector.tolist())

    # 時間軸特性（指標パラメータから推定）
    period_values = []
    period_keys = [
        "period",
        "fast_period",
        "slow_period",
        "signal_period",
        "timeperiod",
        "k_period",
        "d_period",
    ]

    for ind in gene.indicators:
        parameters = getattr(ind, "parameters", {}) or {}
        for key in period_keys:
            if key in parameters and isinstance(parameters[key], (int, float)):
                period_values.append(float(parameters[key]))

    if period_values:
        features.append(float(np.mean(period_values)))
        features.append(float(np.max(period_values)))
    else:
        features.append(0.0)
        features.append(0.0)

    # オペランド特性（定数比較 vs 動的比較）
    numeric_operands, dynamic_operands = _count_operand_types(all_conditions)

    features.append(numeric_operands)
    features.append(dynamic_operands)

    return np.array(features)


def _collect_conditions(gene: StrategyGene) -> list[Any]:
    """全条件を1つのリストにまとめる。"""
    all_conditions: list[Any] = []
    if gene.long_entry_conditions:
        all_conditions.extend(gene.long_entry_conditions)
    if gene.short_entry_conditions:
        all_conditions.extend(gene.short_entry_conditions)
    return all_conditions


def _count_operators(
    conditions: list[Any],
    operator_map: Mapping[str, int],
    vector: np.ndarray,
) -> None:
    """条件リスト内のオペレータを再帰的にカウントする。"""
    for cond in conditions:
        if isinstance(cond, ConditionGroup):
            if cond.operator and cond.operator in operator_map:
                idx = operator_map[cond.operator]
                vector[idx] += 1.0

            if cond.conditions:
                _count_operators(cond.conditions, operator_map, vector)
        elif hasattr(cond, "operator"):
            op = cond.operator
            if op in operator_map:
                idx = operator_map[op]
                vector[idx] += 1.0


def _count_operand_types(conditions: list[Any]) -> tuple[float, float]:
    """
    オペランドのタイプ（数値/動的）をカウントする。
    """
    numeric = 0.0
    dynamic = 0.0

    for cond in conditions:
        if isinstance(cond, ConditionGroup):
            if cond.conditions:
                n, d = _count_operand_types(cond.conditions)
                numeric += n
                dynamic += d
        elif hasattr(cond, "right_operand"):
            op_val = cond.right_operand

            is_numeric = False
            if isinstance(op_val, (int, float)):
                is_numeric = True
            elif isinstance(op_val, str):
                try:
                    float(op_val)
                    is_numeric = True
                except ValueError:
                    is_numeric = False

            if is_numeric:
                numeric += 1.0
            else:
                dynamic += 1.0

    return numeric, dynamic

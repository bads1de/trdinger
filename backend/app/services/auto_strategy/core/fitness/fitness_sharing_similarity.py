"""
フィットネス共有の類似度計算ユーティリティ
"""

import logging
from typing import Any, Dict, List, Optional

from app.services.auto_strategy.genes import StrategyGene

logger = logging.getLogger(__name__)

# 定数
INDICATOR_WEIGHT = 0.2
LONG_CONDITION_WEIGHT = 0.2
SHORT_CONDITION_WEIGHT = 0.2
RISK_MANAGEMENT_WEIGHT = 0.2
TPSL_WEIGHT = 0.15
POSITION_SIZING_WEIGHT = 0.05
OPERATOR_SIMILARITY_SCORE = 0.5
OPERAND_SIMILARITY_SCORE = 0.5
METHOD_MATCH_SCORE = 0.5
TPSL_ATTRIBUTE_WEIGHT = 0.25
POSITION_SIZING_ATTRIBUTE_WEIGHT = 0.5
EPSILON = 1e-6


def calculate_similarity(gene1: StrategyGene, gene2: StrategyGene) -> float:
    """
    2つの戦略遺伝子間の類似度を計算する。
    """
    try:
        components = [
            (gene1.indicators, gene2.indicators, calculate_indicator_similarity, INDICATOR_WEIGHT),
            (
                gene1.long_entry_conditions,
                gene2.long_entry_conditions,
                calculate_condition_similarity,
                LONG_CONDITION_WEIGHT,
            ),
            (
                gene1.short_entry_conditions,
                gene2.short_entry_conditions,
                calculate_condition_similarity,
                SHORT_CONDITION_WEIGHT,
            ),
            (
                gene1.risk_management,
                gene2.risk_management,
                calculate_risk_management_similarity,
                RISK_MANAGEMENT_WEIGHT,
            ),
            (gene1.tpsl_gene, gene2.tpsl_gene, calculate_tpsl_similarity, TPSL_WEIGHT),
            (
                gene1.position_sizing_gene,
                gene2.position_sizing_gene,
                calculate_position_sizing_similarity,
                POSITION_SIZING_WEIGHT,
            ),
        ]

        total_similarity = 0.0
        for val1, val2, calc_func, weight in components:
            similarity = calc_func(val1, val2)  # type: ignore[arg-type]
            total_similarity += similarity * weight

        return max(0.0, min(1.0, total_similarity))
    except Exception as e:
        logger.error(f"類似度計算エラー: {e}")
        return 0.0


def check_none_similarity(val1: object, val2: object) -> Optional[float]:
    """
    None値に対する類似度チェックの共通処理。
    """
    if val1 is None and val2 is None:
        return 1.0
    if val1 is None or val2 is None:
        return 0.0
    return None


def calculate_indicator_similarity(
    indicators1: List[Any], indicators2: List[Any]
) -> float:
    """
    2つの指標セット間の類似度を計算する。
    """
    res = check_none_similarity(indicators1, indicators2)
    if res is not None:
        return res

    types1 = {ind.type for ind in indicators1}
    types2 = {ind.type for ind in indicators2}
    union = len(types1 | types2)
    return len(types1 & types2) / union if union > 0 else 0.0


def calculate_condition_similarity(
    conditions1: List[Any], conditions2: List[Any]
) -> float:
    """
    2つの条件リスト間の類似度を計算する。
    """
    res = check_none_similarity(conditions1, conditions2)
    if res is not None:
        return res

    similar_count = 0.0
    total = max(len(conditions1), len(conditions2))
    if total == 0:
        return 1.0

    for c1, c2 in zip(conditions1, conditions2):
        if getattr(c1, "operator", None) == getattr(c2, "operator", None):
            similar_count += OPERATOR_SIMILARITY_SCORE
        if str(type(getattr(c1, "left_operand", None))) == str(
            type(getattr(c2, "left_operand", None))
        ):
            similar_count += OPERAND_SIMILARITY_SCORE

    return similar_count / total


def calculate_risk_management_similarity(
    risk1: Dict[str, Any], risk2: Dict[str, Any]
) -> float:
    """
    リスク管理設定の類似度を計算する。
    """
    res = check_none_similarity(risk1, risk2)
    if res is not None:
        return res

    common_fields = set(risk1.keys()) & set(risk2.keys())
    if not common_fields:
        return 0.0

    score = 0.0
    for field in common_fields:
        v1, v2 = risk1[field], risk2[field]
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if v1 == v2:
                score += 1.0
            else:
                max_v = max(abs(v1), abs(v2))
                score += max(0.0, 1.0 - abs(v1 - v2) / max_v) if max_v > 0 else 1.0
        elif v1 == v2:
            score += 1.0

    return score / len(common_fields)


def calculate_tpsl_similarity(tpsl1: Any, tpsl2: Any) -> float:
    """TP/SL遺伝子の類似度を計算する。"""
    res = check_none_similarity(tpsl1, tpsl2)
    if res is not None:
        return res

    score = METHOD_MATCH_SCORE if tpsl1.method == tpsl2.method else 0.0
    for attr in ["stop_loss_pct", "take_profit_pct"]:
        v1, v2 = getattr(tpsl1, attr, None), getattr(tpsl2, attr, None)
        if v1 is not None and v2 is not None:
            diff = abs(v1 - v2)
            score += max(0.0, TPSL_ATTRIBUTE_WEIGHT * (1 - diff / max(v1, v2, EPSILON)))
    return min(1.0, score)


def calculate_position_sizing_similarity(ps1: Any, ps2: Any) -> float:
    """ポジションサイジング遺伝子の類似度を計算する。"""
    res = check_none_similarity(ps1, ps2)
    if res is not None:
        return res

    score = METHOD_MATCH_SCORE if ps1.method == ps2.method else 0.0
    v1, v2 = getattr(ps1, "risk_per_trade", None), getattr(ps2, "risk_per_trade", None)
    if v1 is not None and v2 is not None:
        diff = abs(v1 - v2)
        score += max(0.0, POSITION_SIZING_ATTRIBUTE_WEIGHT * (1 - diff / max(v1, v2, EPSILON)))
    return min(1.0, score)

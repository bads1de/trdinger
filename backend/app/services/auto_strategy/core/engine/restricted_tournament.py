"""
制限トーナメント置換 (Restricted Tournament Replacement, RTR)

次世代選択の前段として、各子個体を「構造的に最も近い」既存個体（ニッチ代表）
と局所競争させる生存選択。グローバルな適応度順の切り詰めでは、多数の類似した
子がわずかな適応度差で多様な親を一斉に置き換えてしまう。RTRは子が自分の
ニッチの代表を支配した場合のみ置換を許すため、指標構成の多様性が選択圧から
独立に守られる。

GDE3 (Kukkonen & Lampinen 2005) の生存選択を (mu+lambda) バッチ選択へ適応させた
もので、最終的な次世代の決定は既存の NSGA-II 二段階選抜に委ねる。
"""

from __future__ import annotations

import random
from typing import Any

from app.services.auto_strategy.utils.strategy_dedup import strategy_structure_signature

from ..fitness.fitness_validation import has_valid_fitness

# 指標構成の要素: (指標タイプ, タイムフレーム)
ElementSet = frozenset[tuple[str, str]]


def indicator_set_elements(strategy: Any) -> ElementSet:
    """有効な指標の (type, timeframe) 組の集合を返す。"""
    elements: set[tuple[str, str]] = set()
    for indicator in getattr(strategy, "indicators", None) or []:
        if not getattr(indicator, "enabled", True):
            continue
        elements.add(
            (
                str(getattr(indicator, "type", "")),
                str(getattr(indicator, "timeframe", None) or ""),
            )
        )
    return frozenset(elements)


def jaccard_distance(a: ElementSet, b: ElementSet) -> float:
    """2つの指標構成集合の Jaccard 距離（0.0=同一構成, 1.0=無共通）。"""
    union = len(a | b)
    if union == 0:
        return 0.0
    return 1.0 - len(a & b) / union


def _dominates(candidate: Any, competitor: Any) -> bool:
    """candidate が competitor を Pareto 支配するか（重み付き目的に従う）。"""
    candidate_fitness = getattr(candidate, "fitness", None)
    competitor_fitness = getattr(competitor, "fitness", None)
    if candidate_fitness is None or competitor_fitness is None:
        return False
    try:
        return bool(candidate_fitness.dominates(competitor_fitness))
    except Exception:  # fitness が DEAP 以外のモック等の場合は支配なしとみなす
        return False


def restricted_tournament_replace(
    parents: list[Any],
    offspring: list[Any],
    *,
    crowding_factor: int = 8,
    rng: random.Random | None = None,
) -> list[Any]:
    """子個体を構造的に最も近い既存個体と競争させて生存プールを組む。

    各子個体は生存プールから ``crowding_factor`` 個をサンプリングし、指標構成
    距離が最小の個体（ニッチ代表）と比較する:

    - 子が代表を支配 → 代表のスロットを子が置き換える（ニッチ内改善）
    - 代表が子を支配 → 子は棄却
    - 非支配のとき:
      - 構造シグネチャが完全同一（クローン）→ 子は棄却
      - 構造が異なる → 子を追加（多様性の純増）

    Args:
        parents: 現世代の個体群（変更されない）。
        offspring: 評価済みの子個体群。
        crowding_factor: 局所競争相手のサンプル数（1以上に丸める）。
        rng: 乱数生成器。None の場合はグローバル乱数を使用する。

    Returns:
        生存親 + 受理された子のリスト。NSGA-II 選抜の候補プールとして使う。
    """
    sampler: Any = rng if rng is not None else random
    sample_size = max(int(crowding_factor), 1)

    survivors = list(parents)
    slot_elements = [indicator_set_elements(individual) for individual in survivors]
    slot_signatures = [strategy_structure_signature(ind) for ind in survivors]

    for child in offspring:
        if not has_valid_fitness(child):
            continue

        if not survivors:
            survivors.append(child)
            slot_elements.append(indicator_set_elements(child))
            slot_signatures.append(strategy_structure_signature(child))
            continue

        child_elements = indicator_set_elements(child)
        indices = sampler.sample(
            range(len(survivors)), min(sample_size, len(survivors))
        )
        closest = min(
            indices,
            key=lambda index: jaccard_distance(child_elements, slot_elements[index]),
        )
        competitor = survivors[closest]

        if _dominates(child, competitor):
            survivors[closest] = child
            slot_elements[closest] = child_elements
            slot_signatures[closest] = strategy_structure_signature(child)
            continue

        if _dominates(competitor, child):
            continue

        child_signature = strategy_structure_signature(child)
        if child_signature == slot_signatures[closest]:
            # 構造クローン: 代表スロットが同じ情報を既に保持している
            continue
        survivors.append(child)
        slot_elements.append(child_elements)
        slot_signatures.append(child_signature)

    return survivors

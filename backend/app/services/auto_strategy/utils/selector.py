from typing import List, Dict, Any, Tuple

from .metrics import score_strategy_quality, passes_quality_threshold


def filter_and_rank_strategies(
    stats_list: List[Dict[str, Any]],
    min_trades: int = 1,
    require_quality_threshold: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    成立率>=60%の母集団から品質基準で二次選別し、スコアで降順ソート
    Returns: (selected, rejected)
    """
    # 成立: # Trades >= min_trades
    eligible = [s for s in stats_list if int(s.get("# Trades", 0)) >= min_trades]
    if not eligible:
        return [], stats_list

    # 成立率計算
    success_rate = len(eligible) / max(len(stats_list), 1)
    if success_rate < 0.6:
        # 要件に満たない場合、成立分をそのまま返し、rejectは非成立
        selected = sorted(eligible, key=score_strategy_quality, reverse=True)
        rejected = [s for s in stats_list if s not in eligible]
        return selected, rejected

    # 品質基準でフィルタ
    if require_quality_threshold:
        filtered = [s for s in eligible if passes_quality_threshold(s)]
    else:
        filtered = eligible

    # スコアでランク
    ranked = sorted(filtered, key=score_strategy_quality, reverse=True)
    rejected = [s for s in stats_list if s not in ranked]
    return ranked, rejected


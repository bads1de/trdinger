from __future__ import annotations

from typing import List

from app.services.auto_strategy.models.condition_group import ConditionGroup
from app.services.auto_strategy.models.gene_strategy import Condition
from .price_trend_policy import PriceTrendPolicy


class ConditionAssembly:
    """
    条件の正規化/組立ヘルパー：
    - フォールバック（価格 vs トレンド or open）の注入
    - 1件なら素条件のまま、2件以上なら OR グルーピング
    """

    @staticmethod
    def ensure_or_with_fallback(
        conds: List[Condition], side: str, indicators
    ) -> List[Condition | ConditionGroup]:
        # フォールバック
        trend_name = PriceTrendPolicy.pick_trend_name(indicators)
        fallback = Condition(
            left_operand="close",
            operator=">" if side == "long" else "<",
            right_operand=trend_name or "open",
        )
        if not conds:
            return [fallback]
        # 平坦化（既に OR グループがある場合は中身だけ取り出す）
        flat: List[Condition] = []
        for c in conds:
            if hasattr(c, "conditions") and isinstance(getattr(c, "conditions"), list):
                flat.extend(c.conditions)
            else:
                flat.append(c)
        # フォールバックの重複チェック
        exists = any(
            getattr(x, "left_operand", None) == fallback.left_operand
            and getattr(x, "operator", None) == fallback.operator
            and getattr(x, "right_operand", None) == fallback.right_operand
            for x in flat
        )
        if len(flat) == 1:
            return flat if exists else flat + [fallback]
        top_level: List[Condition | ConditionGroup] = [ConditionGroup(conditions=flat)]
        # 存在していてもトップレベルに1本は追加して可視化と成立性の底上げを図る
        top_level.append(fallback)
        return top_level

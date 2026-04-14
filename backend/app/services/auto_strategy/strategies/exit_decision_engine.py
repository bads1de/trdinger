"""
イグジット判定と決済実行を担当するモジュール。

UniversalStrategy.next() に集中していた決済の責務を分離する。
"""

from __future__ import annotations

import logging

from ..config.constants import ExitType
from ..genes import Condition, ConditionGroup

logger = logging.getLogger(__name__)


class ExitDecisionEngine:
    """イグジット方向の決定と決済実行を担当するクラス。"""

    def __init__(self, strategy):
        self.strategy = strategy

    def determine_exit_direction(self) -> float:
        """
        現在バーでイグジットすべき方向を返す。

        Returns:
            1.0 (ロング決済), -1.0 (ショート決済), 0.0 (決済なし)
        """
        position = self.strategy.position
        if not position:
            return 0.0

        direction = 1.0 if position.size > 0 else -1.0
        exit_gene = self.strategy._get_effective_exit_gene(direction)

        if not exit_gene or not exit_gene.enabled:
            return 0.0

        exit_conditions = self._get_exit_conditions(direction)
        if not exit_conditions:
            return 0.0

        if self._evaluate_exit_conditions(exit_conditions):
            return direction

        return 0.0

    def execute_exit(self, direction: float) -> bool:
        """
        指定方向での決済を実行する。

        Returns:
            実際に決済が実行された場合 True
        """
        if direction == 0.0:
            return False

        position = self.strategy.position
        if not position:
            return False

        exit_gene = self.strategy._get_effective_exit_gene(direction)
        if not exit_gene or not exit_gene.enabled:
            return False

        exit_type = exit_gene.exit_type

        if exit_type == ExitType.PARTIAL and exit_gene.partial_exit_enabled:
            return self._execute_partial_exit(direction, exit_gene)
        elif exit_type == ExitType.TRAILING and exit_gene.trailing_stop_activation:
            return self._activate_trailing_stop(direction, exit_gene)
        else:
            return self._execute_full_exit(direction)

    def _get_exit_conditions(
        self, direction: float
    ) -> list:
        """方向に応じたイグジット条件リストを取得。"""
        if direction > 0:
            return getattr(self.strategy.gene, "long_exit_conditions", [])
        else:
            return getattr(self.strategy.gene, "short_exit_conditions", [])

    def _evaluate_exit_conditions(
        self, conditions: list
    ) -> bool:
        """
        イグジット条件を評価する。

        Returns:
            条件が成立した場合 True
        """
        if not conditions:
            return False

        # ベクトル化キャッシュがあればそれを使用（高速パス）
        cached_signals = self._get_cached_exit_signals()
        if cached_signals is not None:
            current_bar = self.strategy._current_bar_index
            if 0 <= current_bar < len(cached_signals):
                return bool(cached_signals[current_bar])

        # フォールバック: 逐次評価
        evaluator = self.strategy.condition_evaluator

        for condition_group in conditions:
            if isinstance(condition_group, ConditionGroup):
                if self._evaluate_condition_group(condition_group, evaluator):
                    return True
            elif isinstance(condition_group, Condition):
                if evaluator.evaluate_single_condition(condition_group, self.strategy):
                    return True

        return False

    def _get_cached_exit_signals(self):
        """ベクトル化されたExit条件のキャッシュ信号を取得。"""
        position = self.strategy.position
        if not position:
            return None
        direction = 1.0 if position.size > 0 else -1.0
        cached = getattr(self.strategy, "_precomputed_exit_signals", {})
        return cached.get(direction)

    def _evaluate_condition_group(
        self, group: ConditionGroup, evaluator
    ) -> bool:
        """ConditionGroupを再帰的に評価。"""
        if not group.conditions:
            return False

        if group.operator == "AND":
            return all(
                self._evaluate_single_condition(cond, evaluator)
                for cond in group.conditions
            )
        else:  # OR
            return any(
                self._evaluate_single_condition(cond, evaluator)
                for cond in group.conditions
            )

    def _evaluate_single_condition(self, cond, evaluator) -> bool:
        """単一条件を評価。"""
        if isinstance(cond, ConditionGroup):
            return self._evaluate_condition_group(cond, evaluator)
        elif isinstance(cond, Condition):
            return evaluator.evaluate_single_condition(cond, self.strategy)
        return False

    def _execute_partial_exit(self, direction: float, exit_gene) -> bool:
        """部分決済を実行。"""
        position = self.strategy.position
        if not position:
            return False

        exit_pct = exit_gene.partial_exit_pct
        exit_size = abs(position.size) * exit_pct

        if exit_size <= 0:
            return False

        current_price = self.strategy.data.Close[-1]
        logger.info(
            f"部分決済: direction={'LONG' if direction > 0 else 'SHORT'}, "
            f"size={exit_size:.4f}, price={current_price}, pct={exit_pct:.2%}"
        )

        if direction > 0:
            self.strategy.sell(size=exit_size)
        else:
            self.strategy.buy(size=exit_size)

        return True

    def _execute_full_exit(self, direction: float) -> bool:
        """全ポジション決済を実行。"""
        position = self.strategy.position
        if not position:
            return False

        current_price = self.strategy.data.Close[-1]
        logger.info(
            f"全決済: direction={'LONG' if direction > 0 else 'SHORT'}, "
            f"size={abs(position.size):.4f}, price={current_price}"
        )

        if direction > 0:
            self.strategy.sell(size=abs(position.size))
        else:
            self.strategy.buy(size=abs(position.size))

        return True

    def _activate_trailing_stop(self, direction: float, exit_gene) -> bool:
        """
        トレーリングSLを起動する。
        決済は実行せず、PositionManagerにトレーリング開始を通知。
        """
        logger.info(
            f"トレーリングSL起動: direction={'LONG' if direction > 0 else 'SHORT'}"
        )
        self.strategy.position_manager.activate_trailing_stop()
        return True

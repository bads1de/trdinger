"""
ステートフル条件評価モジュール

UniversalStrategyのステートフル条件評価ロジックを担当します。
トリガーチェック、条件評価、エントリー方向取得などの機能を提供します。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class StatefulConditionsEvaluator:
    """
    ステートフル条件評価クラス

    UniversalStrategyのステートフル条件評価ロジックを分離したクラス。
    トリガーチェック、条件評価、エントリー方向取得などの機能を提供します。
    """

    def __init__(self, strategy):
        """
        初期化

        Args:
            strategy: UniversalStrategyインスタンス
        """
        self.strategy = strategy

    def process_stateful_triggers(self) -> None:
        """
        ステートフル条件のトリガーをチェックし、StateTrackerに記録

        各バーで呼ばれ、すべてのStatefulConditionのトリガー条件を評価します。
        成立していれば、StateTrackerにイベントとして記録します。
        """
        if not self.strategy.gene or not hasattr(
            self.strategy.gene, "stateful_conditions"
        ):
            return

        for stateful_cond in self.strategy.gene.stateful_conditions:
            if stateful_cond.enabled:
                self.strategy.condition_evaluator.check_and_record_trigger(
                    stateful_cond,
                    self.strategy,
                    self.strategy.state_tracker,
                    self.strategy._current_bar_index,
                )

    def check_stateful_conditions(self) -> bool:
        """
        ステートフル条件を評価

        いずれかのステートフル条件が成立していればTrueを返します。

        Returns:
            ステートフル条件成立ならTrue
        """
        if not self.strategy.gene or not hasattr(
            self.strategy.gene, "stateful_conditions"
        ):
            return False

        for stateful_cond in self.strategy.gene.stateful_conditions:
            if stateful_cond.enabled:
                result = self.strategy.condition_evaluator.evaluate_stateful_condition(
                    stateful_cond,
                    self.strategy,
                    self.strategy.state_tracker,
                    self.strategy._current_bar_index,
                )
                if result:
                    return True

        return False

    def get_stateful_entry_direction(self) -> Optional[float]:
        """
        成立したステートフル条件からエントリー方向を取得

        いずれかのステートフル条件が成立していれば、その条件に設定された
        direction を元にエントリー方向を返します。

        Returns:
            1.0 (Long), -1.0 (Short), または None（条件不成立時）
        """
        if not self.strategy.gene or not hasattr(
            self.strategy.gene, "stateful_conditions"
        ):
            return None

        for stateful_cond in self.strategy.gene.stateful_conditions:
            if stateful_cond.enabled:
                result = self.strategy.condition_evaluator.evaluate_stateful_condition(
                    stateful_cond,
                    self.strategy,
                    self.strategy.state_tracker,
                    self.strategy._current_bar_index,
                )
                if result:
                    # direction フィールドに基づいてエントリー方向を返す
                    direction = getattr(stateful_cond, "direction", "long")
                    return 1.0 if direction == "long" else -1.0

        return None

"""
1バー分の戦略実行サイクルを担当するモジュール。
"""

from __future__ import annotations

from typing import cast

import pandas as pd


class StrategyExecutionCycle:
    """バー単位の前処理、既存ポジション処理、新規エントリー判定をまとめる。"""

    def __init__(self, strategy):
        self.strategy = strategy

    def run_current_bar(self) -> None:
        """現在バーの実行サイクルを進める。"""
        if not self.strategy._is_evaluation_bar():
            return

        self.process_bar_setup()

        handled_open_position = self.strategy.position_exit_engine.handle_open_position()
        self.strategy._check_early_termination()
        if handled_open_position:
            return

        if self.strategy.position:
            return

        direction = self.strategy.entry_decision_engine.determine_entry_direction()
        if direction == 0.0:
            return

        self.strategy.entry_decision_engine.execute_entry(direction)

    def process_bar_setup(self) -> None:
        """保留注文とステートフルトリガーの前処理を実行する。"""
        if self.strategy._minute_data is not None:
            self.strategy.order_manager.check_pending_order_fills(
                cast(pd.DataFrame, self.strategy._minute_data),
                self.strategy.data.index[-1],
                self.strategy._current_bar_index,
            )

        self.strategy.order_manager.expire_pending_orders(
            self.strategy._current_bar_index
        )
        self.strategy.stateful_conditions_evaluator.process_stateful_triggers()

"""
エントリー判定と発注実行を担当するモジュール。

UniversalStrategy.next() に集中していた新規エントリーの責務を分離する。
"""

from __future__ import annotations

from ..config.constants import EntryType


class EntryDecisionEngine:
    """エントリー方向の決定と注文実行を担当するクラス。"""

    def __init__(self, strategy):
        self.strategy = strategy

    def determine_entry_direction(self) -> float:
        """
        現在バーでエントリーすべき方向を返す。

        優先順位:
        1. 通常ロング
        2. 通常ショート
        3. ステートフル条件
        """
        if self.strategy._tools_block_entry():
            return 0.0

        if self.strategy._check_entry_conditions(1.0):
            return 1.0
        if self.strategy._check_entry_conditions(-1.0):
            return -1.0

        stateful_dir = self.strategy.stateful_conditions_evaluator.get_stateful_entry_direction()
        return 0.0 if stateful_dir is None else stateful_dir

    def execute_entry(self, direction: float) -> bool:
        """
        指定方向での新規エントリーを実行する。

        Returns:
            実際に注文実行または保留注文作成まで進んだ場合 True
        """
        if direction == 0.0:
            return False

        if (
            getattr(self.strategy, "volatility_gate_enabled", False)
            and self.strategy.ml_predictor
            and not self.strategy._ml_allows_entry(direction)
        ):
            return False

        current_price = self.strategy.data.Close[-1]
        sl_price, tp_price = self.strategy._calculate_effective_tpsl_prices(
            direction, current_price
        )

        entry_gene = self.strategy._get_effective_entry_gene(direction)
        entry_params = self.strategy.entry_executor.calculate_entry_params(
            entry_gene, current_price, direction
        )
        position_size = self.strategy._calculate_position_size()

        is_market = (
            entry_gene is None
            or not entry_gene.enabled
            or entry_gene.entry_type == EntryType.MARKET
        )

        if is_market:
            if direction > 0:
                self.strategy.buy(size=position_size)
            else:
                self.strategy.sell(size=position_size)

            self.strategy.runtime_state.set_open_position(
                entry_price=current_price,
                sl_price=sl_price,
                tp_price=tp_price,
                direction=direction,
            )
            return True

        self.strategy.order_manager.create_pending_order(
            direction=direction,
            size=position_size,
            entry_params=entry_params,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_gene=entry_gene,
            current_bar_index=self.strategy._current_bar_index,
        )
        return True

"""
既存ポジションの exit 判定を担当するモジュール。
"""

from __future__ import annotations


class PositionExitEngine:
    """オープンポジション時の決済判定をまとめるクラス。"""

    def __init__(self, strategy):
        self.strategy = strategy

    def handle_open_position(self) -> bool:
        """
        既存ポジションに対する exit 判定を実行する。

        Returns:
            決済が行われ処理を終了すべき場合 True
        """
        if not self.strategy.position:
            return False
        if self.strategy.runtime_state.sl_price is None:
            return False
        return self.strategy.position_manager.check_pessimistic_exit()

"""
ポジション管理モジュール

UniversalStrategyのポジション管理と決済ロジックを担当します。
悲観的約定判定、トレーリングストップ、トレーリングTPなどの機能を提供します。
"""

import logging

logger = logging.getLogger(__name__)


class PositionManager:
    """
    ポジション管理クラス

    UniversalStrategyのポジション管理ロジックを分離したクラス。
    悲観的約定判定、トレーリングストップ、トレーリングTPなどの機能を提供します。
    """

    def __init__(self, strategy):
        """
        初期化

        Args:
            strategy: UniversalStrategyインスタンス
        """
        self.strategy = strategy

    def check_pessimistic_exit(self) -> bool:
        """
        悲観的約定ロジックによるSL/TP判定

        同一足内でSLとTPの両方に達した場合、SLを優先して決済します。
        これにより「幻の利益」を防ぎ、バックテスト結果を安全側に倒します。

        Returns:
            True: 決済が実行された場合
            False: 決済が実行されなかった場合
        """
        if self.strategy._sl_price is None:
            return False

        current_low = self.strategy.data.Low[-1]
        current_high = self.strategy.data.High[-1]

        # ロングポジションの場合
        if self.strategy._position_direction > 0:
            # トレーリングTP到達後モード: 利益確保ラインで決済判定
            if self.strategy._tp_reached and self.strategy._trailing_tp_sl is not None:
                if current_low <= self.strategy._trailing_tp_sl:
                    self.strategy.position.close()
                    self.reset_position_state()
                    return True
                # 利益確保ラインを更新（さらに上昇した場合）
                self.update_trailing_tp_sl()
                return False

            # 1. SL判定 [最優先]: Low <= SL価格
            if current_low <= self.strategy._sl_price:
                self.strategy.position.close()
                self.reset_position_state()
                return True

            # 2. TP判定 [次点]: High >= TP価格
            if (
                self.strategy._tp_price is not None
                and current_high >= self.strategy._tp_price
            ):
                # トレーリングTPが有効な場合は即時決済せず、利益確保モードへ
                if self.is_trailing_tp_enabled():
                    self.strategy._tp_reached = True
                    # 初期利益確保ライン = TP価格（ここから追従開始）
                    self.strategy._trailing_tp_sl = self.strategy._tp_price
                    self.update_trailing_tp_sl()
                    return False
                else:
                    self.strategy.position.close()
                    self.reset_position_state()
                    return True

        # ショートポジションの場合
        elif self.strategy._position_direction < 0:
            # トレーリングTP到達後モード: 利益確保ラインで決済判定
            if self.strategy._tp_reached and self.strategy._trailing_tp_sl is not None:
                if current_high >= self.strategy._trailing_tp_sl:
                    self.strategy.position.close()
                    self.reset_position_state()
                    return True
                # 利益確保ラインを更新（さらに下落した場合）
                self.update_trailing_tp_sl()
                return False

            # 1. SL判定 [最優先]: High >= SL価格 (ショートはSLが上側)
            if current_high >= self.strategy._sl_price:
                self.strategy.position.close()
                self.reset_position_state()
                return True

            # 2. TP判定 [次点]: Low <= TP価格 (ショートはTPが下側)
            if (
                self.strategy._tp_price is not None
                and current_low <= self.strategy._tp_price
            ):
                # トレーリングTPが有効な場合は即時決済せず、利益確保モードへ
                if self.is_trailing_tp_enabled():
                    self.strategy._tp_reached = True
                    # 初期利益確保ライン = TP価格（ここから追従開始）
                    self.strategy._trailing_tp_sl = self.strategy._tp_price
                    self.update_trailing_tp_sl()
                    return False
                else:
                    self.strategy.position.close()
                    self.reset_position_state()
                    return True

        # === トレーリングストップ更新 ===
        # 決済条件に達しなかった場合、トレーリングが有効ならSLを更新
        self.update_trailing_stop()

        return False

    def reset_position_state(self) -> None:
        """ポジション決済後に内部状態をリセット"""
        self.strategy._sl_price = None
        self.strategy._tp_price = None
        self.strategy._entry_price = None
        self.strategy._position_direction = 0.0
        self.strategy._tp_reached = False
        self.strategy._trailing_tp_sl = None

    def is_trailing_tp_enabled(self) -> bool:
        """トレーリングTPが有効かどうかを確認"""
        active_tpsl_gene = self.strategy._get_effective_tpsl_gene(
            self.strategy._position_direction
        )
        if not active_tpsl_gene:
            return False
        return getattr(active_tpsl_gene, "trailing_take_profit", False)

    def update_trailing_tp_sl(self) -> None:
        """
        トレーリングTP用の利益確保ラインを更新

        TP到達後、価格がさらに有利な方向に動いた場合、
        利益確保ライン（実質的なSL）を追従させます。
        """
        if not self.strategy._tp_reached or self.strategy._trailing_tp_sl is None:
            return

        active_tpsl_gene = self.strategy._get_effective_tpsl_gene(
            self.strategy._position_direction
        )
        if not active_tpsl_gene:
            return

        trailing_step = getattr(active_tpsl_gene, "trailing_step_pct", 0.01)
        current_close = self.strategy.data.Close[-1]

        # ロングポジションの場合: 終値ベースで新しい利益確保ラインを計算
        if self.strategy._position_direction > 0:
            new_trailing_sl = current_close * (1.0 - trailing_step)
            if new_trailing_sl > self.strategy._trailing_tp_sl:
                self.strategy._trailing_tp_sl = new_trailing_sl

        # ショートポジションの場合
        elif self.strategy._position_direction < 0:
            new_trailing_sl = current_close * (1.0 + trailing_step)
            if new_trailing_sl < self.strategy._trailing_tp_sl:
                self.strategy._trailing_tp_sl = new_trailing_sl

    def update_trailing_stop(self) -> None:
        """
        トレーリングストップの更新

        価格が有利な方向に動いた場合、SLを追従させます。
        SLは有利な方向にのみ移動し、不利な方向には絶対に戻しません。
        """
        # トレーリングが有効か確認
        active_tpsl_gene = self.strategy._get_effective_tpsl_gene(
            self.strategy._position_direction
        )
        if not active_tpsl_gene:
            return
        if not getattr(active_tpsl_gene, "trailing_stop", False):
            return
        if self.strategy._sl_price is None:
            return

        trailing_step = getattr(active_tpsl_gene, "trailing_step_pct", 0.01)
        current_close = self.strategy.data.Close[-1]

        # ロングポジションの場合: 終値ベースで新SLを計算し、現在SLより高ければ更新
        if self.strategy._position_direction > 0:
            new_sl = current_close * (1.0 - trailing_step)
            if new_sl > self.strategy._sl_price:
                self.strategy._sl_price = new_sl

        # ショートポジションの場合: 終値ベースで新SLを計算し、現在SLより低ければ更新
        elif self.strategy._position_direction < 0:
            new_sl = current_close * (1.0 + trailing_step)
            if new_sl < self.strategy._sl_price:
                self.strategy._sl_price = new_sl

"""
リスク管理基盤クラス

backtesting.pyの組み込みSL/TP機能を活用したMixinクラス
"""

import logging

from typing import Optional, Dict, Any

from .calculators import RiskCalculator, calculate_sl_tp_prices
from .validators import validate_risk_parameters
from backtesting import Strategy

logger = logging.getLogger(__name__)


class RiskManagementMixin(Strategy):
    """
    リスク管理機能を提供するMixinクラス

    backtesting.pyのStrategyクラスと組み合わせて使用し、
    統一されたリスク管理機能を提供します。

    Usage:
        class MyStrategy(RiskManagementMixin, Strategy):
            def init(self):
                self.setup_risk_management(sl_pct=0.02, tp_pct=0.05)

            def next(self):
                if self.should_buy():
                    self.buy_with_risk_management()
    """

    def setup_risk_management(
        self,
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
        min_risk_reward_ratio: Optional[float] = None,
        use_kelly_criterion: bool = False,
        position_sizing_method: str = "fixed_ratio",
    ):
        """
        リスク管理の初期設定

        Args:
            sl_pct: デフォルトストップロス率
            tp_pct: デフォルトテイクプロフィット率
            config: 詳細なリスク管理設定辞書
            min_risk_reward_ratio: 最小リスクリワード比率
            use_kelly_criterion: Kelly基準を使用するかどうか
            position_sizing_method: ポジションサイジング方法
        """
        try:
            # デフォルト値の設定
            self._default_sl_pct = sl_pct or 0.02  # 2%
            self._default_tp_pct = tp_pct or 0.05  # 5%
            self._min_risk_reward_ratio = min_risk_reward_ratio or 1.5  # 1.5:1
            self._use_kelly_criterion = use_kelly_criterion
            self._position_sizing_method = position_sizing_method

            # リスク計算機の初期化
            self._risk_calculator = RiskCalculator(
                default_sl_pct=self._default_sl_pct, default_tp_pct=self._default_tp_pct
            )

            # 詳細設定の処理
            if config:
                self._risk_config = validate_risk_parameters(config)
            else:
                self._risk_config = {}

            # 取引履歴の初期化（Kelly基準用）
            self._trade_history = []

            logger.info(
                f"リスク管理をセットアップしました: SL={self._default_sl_pct:.1%}, TP={self._default_tp_pct:.1%}"
            )
            logger.info(
                f"最小リスクリワード比率: {self._min_risk_reward_ratio}, Kelly基準: {self._use_kelly_criterion}"
            )

        except Exception as e:
            logger.error(f"リスク管理の設定中にエラーが発生しました: {e}")
            raise

    def buy_with_risk_management(
        self,
        size: float = 0.9999,
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        **kwargs,
    ):
        """
        リスク管理付きの買い注文

        Args:
            size: 注文サイズ
            sl_pct: ストップロス率（指定時は絶対価格より優先）
            tp_pct: テイクプロフィット率（指定時は絶対価格より優先）
            sl_price: ストップロス絶対価格
            tp_price: テイクプロフィット絶対価格
            **kwargs: その他のbuy()パラメータ
        """
        try:
            current_price = self.data.Close[-1]

            # SL/TP価格の計算
            final_sl, final_tp = self._calculate_sl_tp_for_order(
                current_price, True, sl_pct, tp_pct, sl_price, tp_price
            )

            # リスクリワード比率のチェック
            if final_sl and final_tp:
                rr_ratio = self._risk_calculator.calculate_risk_reward_ratio(
                    current_price, final_sl, final_tp, is_long=True
                )
                if rr_ratio and rr_ratio < self._min_risk_reward_ratio:
                    logger.warning(
                        f"リスクリワード比率 {rr_ratio:.2f} が最小値 {self._min_risk_reward_ratio} を下回っています。"
                    )
                    return None

            # ポジションサイズの最適化
            if (
                self._use_kelly_criterion
                or self._position_sizing_method != "fixed_ratio"
            ):
                optimal_size = self._calculate_optimal_position_size(
                    current_price, final_sl, True
                )
                if optimal_size:
                    size = optimal_size

            # 注文実行
            order = self.buy(size=size, sl=final_sl, tp=final_tp, **kwargs)

            if order:
                logger.info(
                    f"買い注文が実行されました: 価格={current_price:.2f}, サイズ={size:.4f}, SL={final_sl}, TP={final_tp}"
                )

            return order

        except Exception as e:
            logger.error(f"リスク管理付き買い注文の実行中にエラーが発生しました: {e}")
            return None

    def sell_with_risk_management(
        self,
        size: float = 0.9999,
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        **kwargs,
    ):
        """
        リスク管理付きの売り注文

        Args:
            size: 注文サイズ
            sl_pct: ストップロス率（指定時は絶対価格より優先）
            tp_pct: テイクプロフィット率（指定時は絶対価格より優先）
            sl_price: ストップロス絶対価格
            tp_price: テイクプロフィット絶対価格
            **kwargs: その他のsell()パラメータ
        """
        try:
            current_price = self.data.Close[-1]

            # SL/TP価格の計算
            final_sl, final_tp = self._calculate_sl_tp_for_order(
                current_price, False, sl_pct, tp_pct, sl_price, tp_price
            )

            # 注文実行
            order = self.sell(size=size, sl=final_sl, tp=final_tp, **kwargs)

            if order:
                logger.info(
                    f"売り注文が実行されました: 価格={current_price:.2f}, SL={final_sl}, TP={final_tp}"
                )

            return order

        except Exception as e:
            logger.error(f"リスク管理付き売り注文の実行中にエラーが発生しました: {e}")
            return None

    def update_trailing_stop(self, atr_multiplier: float = 2.0):
        """
        トレーリングストップの更新

        Args:
            atr_multiplier: ATR倍数（ATRが利用可能な場合）
        """
        try:
            if not self.position or not self.trades:
                return

            current_price = self.data.Close[-1]

            for trade in self.trades:
                if trade.is_long:
                    # ロングポジションのトレーリングストップ
                    new_sl = current_price * (1 - self._default_sl_pct)
                    if trade.sl is None or new_sl > trade.sl:
                        trade.sl = new_sl
                        logger.debug(
                            f"ロングトレードのトレーリングストップを更新しました: {new_sl:.2f}"
                        )

                elif trade.is_short:
                    # ショートポジションのトレーリングストップ
                    new_sl = current_price * (1 + self._default_sl_pct)
                    if trade.sl is None or new_sl < trade.sl:
                        trade.sl = new_sl
                        logger.debug(
                            f"ショートトレードのトレーリングストップを更新しました: {new_sl:.2f}"
                        )

        except Exception as e:
            logger.error(f"トレーリングストップの更新中にエラーが発生しました: {e}")

    def _calculate_sl_tp_for_order(
        self,
        entry_price: float,
        is_long: bool,
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
    ) -> tuple:
        """
        注文用のSL/TP価格を計算

        Args:
            entry_price: エントリー価格
            is_long: ロングポジションかどうか
            sl_pct: ストップロス率
            tp_pct: テイクプロフィット率
            sl_price: ストップロス絶対価格
            tp_price: テイクプロフィット絶対価格

        Returns:
            tuple: (sl_price, tp_price)
        """
        try:
            # パーセンテージが指定されている場合は優先
            if sl_pct is not None or tp_pct is not None:
                sl_pct = sl_pct or self._default_sl_pct
                tp_pct = tp_pct or self._default_tp_pct

                return calculate_sl_tp_prices(
                    entry_price, sl_pct, tp_pct, is_long, use_absolute=False
                )

            # 絶対価格が指定されている場合
            elif sl_price is not None or tp_price is not None:
                return calculate_sl_tp_prices(
                    entry_price, sl_price, tp_price, is_long, use_absolute=True
                )

            # デフォルト値を使用
            else:
                return calculate_sl_tp_prices(
                    entry_price,
                    self._default_sl_pct,
                    self._default_tp_pct,
                    is_long,
                    use_absolute=False,
                )

        except Exception as e:
            logger.error(f"注文のSL/TP計算中にエラーが発生しました: {e}")
            return None, None

    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        現在のリスク指標を取得

        Returns:
            リスク指標の辞書
        """
        try:
            metrics = {
                "default_sl_pct": self._default_sl_pct,
                "default_tp_pct": self._default_tp_pct,
                "active_trades": len(self.trades) if hasattr(self, "trades") else 0,
                "position_size": (
                    self.position.size
                    if hasattr(self, "position") and self.position
                    else 0
                ),
                "current_equity": self.equity if hasattr(self, "equity") else 0,
            }

            # アクティブな取引のリスク情報
            if hasattr(self, "trades") and self.trades:
                total_risk = 0
                total_reward = 0

                for trade in self.trades:
                    if trade.sl and trade.tp:
                        if trade.is_long:
                            risk = trade.entry_price - trade.sl
                            reward = trade.tp - trade.entry_price
                        else:
                            risk = trade.sl - trade.entry_price
                            reward = trade.entry_price - trade.tp

                        total_risk += risk * abs(trade.size)
                        total_reward += reward * abs(trade.size)

                if total_risk > 0:
                    metrics["total_risk_reward_ratio"] = total_reward / total_risk
                else:
                    metrics["total_risk_reward_ratio"] = 0

            return metrics

        except Exception as e:
            logger.error(f"リスク指標の取得中にエラーが発生しました: {e}")
            return {}

    def _calculate_optimal_position_size(
        self, entry_price: float, sl_price: Optional[float], is_long: bool
    ) -> Optional[float]:
        """
        最適なポジションサイズを計算

        Args:
            entry_price: エントリー価格
            sl_price: ストップロス価格
            is_long: ロングポジションかどうか

        Returns:
            最適なポジションサイズ（None if 計算不可）
        """
        try:
            current_equity = getattr(self, "equity", 10000)  # デフォルト値

            if self._position_sizing_method == "fixed_risk":
                return self._risk_calculator.calculate_optimal_position_size(
                    current_equity,
                    entry_price,
                    sl_price or entry_price * 0.98,
                    method="fixed_risk",
                    risk_amount=current_equity * 0.01,
                )

            elif self._position_sizing_method == "kelly" or self._use_kelly_criterion:
                # 取引履歴から統計を計算
                win_rate, avg_win, avg_loss = self._calculate_trade_statistics()

                return self._risk_calculator.calculate_optimal_position_size(
                    current_equity,
                    entry_price,
                    sl_price or entry_price * 0.98,
                    method="kelly",
                    win_rate=win_rate,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                )

            else:
                return self._risk_calculator.calculate_optimal_position_size(
                    current_equity,
                    entry_price,
                    sl_price or entry_price * 0.98,
                    method="fixed_ratio",
                    ratio=0.02,
                )

        except Exception as e:
            logger.error(f"最適なポジションサイズの計算中にエラーが発生しました: {e}")
            return None

    def _calculate_trade_statistics(self) -> tuple:
        """
        取引履歴から統計を計算

        Returns:
            tuple: (win_rate, avg_win, avg_loss)
        """
        try:
            if not self._trade_history or len(self._trade_history) < 10:
                # デフォルト値を返す
                return 0.5, 0.05, 0.02

            wins = [trade for trade in self._trade_history if trade > 0]
            losses = [abs(trade) for trade in self._trade_history if trade < 0]

            win_rate = len(wins) / len(self._trade_history)
            avg_win = sum(wins) / len(wins) if wins else 0.05
            avg_loss = sum(losses) / len(losses) if losses else 0.02

            return win_rate, avg_win, avg_loss

        except Exception as e:
            logger.error(f"取引統計の計算中にエラーが発生しました: {e}")
            return 0.5, 0.05, 0.02

    def update_trade_history(self, trade_result: float):
        """
        取引結果を履歴に追加

        Args:
            trade_result: 取引結果（利益率）
        """
        try:
            self._trade_history.append(trade_result)

            # 履歴を最新100件に制限
            if len(self._trade_history) > 100:
                self._trade_history = self._trade_history[-100:]

        except Exception as e:
            logger.error(f"取引履歴の更新中にエラーが発生しました: {e}")

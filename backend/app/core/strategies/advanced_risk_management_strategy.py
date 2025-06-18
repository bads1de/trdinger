"""
高度なリスク管理機能付き戦略

Kelly Criterion、Risk-Reward Ratio、Position Sizingなどの
高度なリスク管理機能を統合した戦略
"""

from backtesting import Strategy
from backtesting.lib import crossover
from .indicators import SMA, RSI, ATR
from .risk_management import RiskManagementMixin
import logging

logger = logging.getLogger(__name__)


class AdvancedRiskManagementStrategy(Strategy, RiskManagementMixin):
    """
    高度なリスク管理機能付き戦略

    以下の高度なリスク管理機能を統合:
    - Kelly Criterion（ケリー基準）による最適ポジションサイジング
    - Risk-Reward Ratio（リスクリワード比率）フィルタリング
    - 動的ポジションサイジング
    - ATRベースのストップロス・テイクプロフィット
    - 取引履歴に基づく適応的リスク管理

    パラメータ:
        n1: 短期SMAの期間（デフォルト: 20）
        n2: 長期SMAの期間（デフォルト: 50）
        rsi_period: RSI期間（デフォルト: 14）
        atr_period: ATR期間（デフォルト: 14）

        # リスク管理パラメータ
        sl_pct: ストップロス率（デフォルト: 0.02）
        tp_pct: テイクプロフィット率（デフォルト: 0.05）
        min_risk_reward_ratio: 最小リスクリワード比率（デフォルト: 2.0）
        use_kelly_criterion: Kelly基準を使用するか（デフォルト: True）
        position_sizing_method: ポジションサイジング方法（デフォルト: "kelly"）

        # ATRベースリスク管理
        use_atr_based_risk: ATRベースのSL/TPを使用するか（デフォルト: False）
        atr_sl_multiplier: ATRストップロス倍数（デフォルト: 2.0）
        atr_tp_multiplier: ATRテイクプロフィット倍数（デフォルト: 3.0）

        # フィルタリング
        rsi_oversold: RSI買われすぎレベル（デフォルト: 30）
        rsi_overbought: RSI売られすぎレベル（デフォルト: 70）
        use_rsi_filter: RSIフィルターを使用するか（デフォルト: True）
    """

    # テクニカル指標パラメータ
    n1 = 20
    n2 = 50
    rsi_period = 14
    atr_period = 14

    # リスク管理パラメータ
    sl_pct = 0.02
    tp_pct = 0.05
    min_risk_reward_ratio = 2.0
    use_kelly_criterion = True
    position_sizing_method = "kelly"

    # ATRベースリスク管理
    use_atr_based_risk = False
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    # フィルタリング
    rsi_oversold = 30
    rsi_overbought = 70
    use_rsi_filter = True

    # 追加パラメータ（backtesting.pyの要求）
    volatility = 0.02
    risk_percent = 0.01
    consecutive_losses = 0
    consecutive_wins = 0
    use_risk_management = True
    min_risk_reward_ratio = 2.0

    def init(self):
        """戦略の初期化"""
        # テクニカル指標の初期化
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.sma1 = self.I(SMA, close, self.n1)  # 短期SMA
        self.sma2 = self.I(SMA, close, self.n2)  # 長期SMA
        self.rsi = self.I(RSI, close, self.rsi_period)  # RSI
        self.atr = self.I(ATR, high, low, close, self.atr_period)  # ATR

        # 高度なリスク管理機能の初期化
        self.setup_risk_management(
            sl_pct=self.sl_pct,
            tp_pct=self.tp_pct,
            min_risk_reward_ratio=self.min_risk_reward_ratio,
            use_kelly_criterion=self.use_kelly_criterion,
            position_sizing_method=self.position_sizing_method,
        )

        logger.info("Advanced Risk Management Strategy initialized")
        logger.info(f"Kelly Criterion: {self.use_kelly_criterion}")
        logger.info(f"Min R:R Ratio: {self.min_risk_reward_ratio}")
        logger.info(f"Position Sizing: {self.position_sizing_method}")

    def next(self):
        """売買ロジック"""
        # 十分なデータが揃うまで待機
        if (
            len(self.sma1) < self.n2
            or len(self.sma2) < self.n2
            or len(self.rsi) < self.rsi_period
            or len(self.atr) < self.atr_period
        ):
            return

        current_price = self.data.Close[-1]
        current_rsi = self.rsi[-1]
        current_atr = self.atr[-1]

        # ゴールデンクロス + RSIフィルター
        if crossover(self.sma1, self.sma2):  # type: ignore
            # RSIフィルター（買われすぎでない場合のみエントリー）
            if self.use_rsi_filter and current_rsi > self.rsi_overbought:
                logger.debug(f"Buy signal ignored: RSI overbought ({current_rsi:.1f})")
                return

            # ATRベースまたはパーセンテージベースのリスク管理
            if self.use_atr_based_risk:
                sl_price = current_price - (current_atr * self.atr_sl_multiplier)
                tp_price = current_price + (current_atr * self.atr_tp_multiplier)

                # リスクリワード比率のチェック
                rr_ratio = self._risk_calculator.calculate_risk_reward_ratio(
                    current_price, sl_price, tp_price, is_long=True
                )

                if rr_ratio and rr_ratio >= self.min_risk_reward_ratio:
                    # ATRベースの注文
                    self.buy(sl=sl_price, tp=tp_price)
                    logger.info(
                        f"ATR-based buy: Price={current_price:.2f}, SL={sl_price:.2f}, TP={tp_price:.2f}, R:R={rr_ratio:.2f}"
                    )
                else:
                    logger.debug(
                        f"ATR-based buy rejected: R:R ratio {rr_ratio:.2f} below minimum"
                    )
            else:
                # パーセンテージベースのリスク管理
                self.buy_with_risk_management()

        # デッドクロス
        elif crossover(self.sma2, self.sma1):  # type: ignore
            if self.position:
                # 取引結果を履歴に記録
                if hasattr(self.position, "pl_pct"):
                    self.update_trade_history(self.position.pl_pct)

                self.position.close()
                logger.info("Position closed on dead cross")

    def get_strategy_performance_metrics(self) -> dict:
        """
        戦略のパフォーマンス指標を取得

        Returns:
            パフォーマンス指標の辞書
        """
        try:
            base_metrics = self.get_risk_metrics()

            # 取引統計の追加
            win_rate, avg_win, avg_loss = self._calculate_trade_statistics()
            kelly_ratio = self._risk_calculator.calculate_kelly_criterion(
                win_rate, avg_win, avg_loss
            )

            advanced_metrics = {
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "kelly_ratio": kelly_ratio,
                "trade_history_length": len(self._trade_history),
                "current_rsi": self.rsi[-1] if len(self.rsi) > 0 else None,
                "current_atr": self.atr[-1] if len(self.atr) > 0 else None,
                "sma_trend": (
                    "bullish"
                    if len(self.sma1) > 0
                    and len(self.sma2) > 0
                    and self.sma1[-1] > self.sma2[-1]
                    else "bearish"
                ),
            }

            return {**base_metrics, **advanced_metrics}

        except Exception as e:
            logger.error(f"Error getting strategy performance metrics: {e}")
            return {}

    def get_strategy_description(self) -> str:
        """戦略の詳細説明を取得"""
        return f"""
        高度なリスク管理機能付き戦略 (SMA {self.n1}/{self.n2})
        
        エントリー条件:
        - ゴールデンクロス: SMA({self.n1}) > SMA({self.n2})
        - RSIフィルター: RSI < {self.rsi_overbought} (有効時)
        
        リスク管理:
        - 最小リスクリワード比率: {self.min_risk_reward_ratio}:1
        - Kelly Criterion: {'有効' if self.use_kelly_criterion else '無効'}
        - ポジションサイジング: {self.position_sizing_method}
        - ATRベースSL/TP: {'有効' if self.use_atr_based_risk else '無効'}
        
        特徴:
        - 動的ポジションサイジング
        - 取引履歴に基づく適応的リスク管理
        - 複数のテクニカル指標によるフィルタリング
        - 高度な統計的リスク管理
        
        推奨市場:
        - トレンドが明確な市場
        - 十分な流動性がある市場
        - ボラティリティが適度な市場
        """


class ConservativeRiskManagementStrategy(AdvancedRiskManagementStrategy):
    """
    保守的なリスク管理戦略

    より保守的なパラメータを使用した安全重視の戦略
    """

    # より保守的なパラメータ
    sl_pct = 0.015  # 1.5%
    tp_pct = 0.045  # 4.5%
    min_risk_reward_ratio = 3.0  # 3:1
    use_kelly_criterion = True
    position_sizing_method = "fixed_risk"  # 固定リスクでより安全

    # より厳しいフィルタリング
    rsi_oversold = 25
    rsi_overbought = 75

    def init(self):
        """保守的戦略の初期化"""
        super().init()
        logger.info("Conservative Risk Management Strategy initialized")


class AggressiveRiskManagementStrategy(AdvancedRiskManagementStrategy):
    """
    アグレッシブなリスク管理戦略

    より積極的なパラメータを使用した高リターン狙いの戦略
    """

    # より積極的なパラメータ
    sl_pct = 0.025  # 2.5%
    tp_pct = 0.075  # 7.5%
    min_risk_reward_ratio = 1.5  # 1.5:1
    use_kelly_criterion = True
    position_sizing_method = "kelly"  # Kelly基準でより積極的

    # より緩いフィルタリング
    rsi_oversold = 35
    rsi_overbought = 65
    use_atr_based_risk = True  # ATRベースでより動的

    def init(self):
        """アグレッシブ戦略の初期化"""
        super().init()
        logger.info("Aggressive Risk Management Strategy initialized")

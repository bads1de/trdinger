"""
SMA + RSI複合戦略

移動平均線のトレンド判定とRSIのモメンタム判定を組み合わせた戦略
"""

from backtesting import Strategy
from backtesting.lib import crossover
from .indicators import SMA, RSI
from .risk_management.base import RiskManagementMixin
import logging

logger = logging.getLogger(__name__)


class SMARSIStrategy(RiskManagementMixin, Strategy):
    """
    SMA + RSI複合戦略

    エントリー条件：
    - 買い：短期SMA > 長期SMA かつ RSI < oversold_threshold（買われすぎでない）
    - 売り：短期SMA < 長期SMA かつ RSI > overbought_threshold（売られすぎでない）

    エグジット条件：
    - 買いポジション：短期SMA < 長期SMA または RSI > overbought_threshold
    - 売りポジション：短期SMA > 長期SMA または RSI < oversold_threshold

    パラメータ:
        sma_short: 短期SMAの期間（デフォルト: 20）
        sma_long: 長期SMAの期間（デフォルト: 50）
        rsi_period: RSIの期間（デフォルト: 14）
        oversold_threshold: 買いシグナルのRSI閾値（デフォルト: 30）
        overbought_threshold: 売りシグナルのRSI閾値（デフォルト: 70）
        sl_pct: ストップロス率（デフォルト: 0.02 = 2%）
        tp_pct: テイクプロフィット率（デフォルト: 0.05 = 5%）
        use_risk_management: リスク管理機能を使用するかどうか（デフォルト: True）
    """

    # デフォルトパラメータ
    sma_short = 20
    sma_long = 50
    rsi_period = 14
    oversold_threshold = 30
    overbought_threshold = 70
    sl_pct = 0.02
    tp_pct = 0.05
    use_risk_management = True

    def init(self):
        """戦略の初期化"""
        # 移動平均線の計算
        self.sma_short_line = self.I(SMA, self.data.Close, self.sma_short)
        self.sma_long_line = self.I(SMA, self.data.Close, self.sma_long)

        # RSIの計算
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)

        # リスク管理の初期化
        if self.use_risk_management:
            self.setup_risk_management(sl_pct=self.sl_pct, tp_pct=self.tp_pct)

        logger.info(
            f"SMA+RSI Strategy initialized: SMA({self.sma_short},{self.sma_long}), RSI({self.rsi_period})"
        )

    def next(self):
        """各バーでの戦略実行"""
        # 十分なデータがない場合はスキップ
        if (
            len(self.sma_short_line) < max(self.sma_short, self.sma_long)
            or len(self.rsi) < self.rsi_period
        ):
            return

        current_price = self.data.Close[-1]
        sma_short_current = self.sma_short_line[-1]
        sma_long_current = self.sma_long_line[-1]
        rsi_current = self.rsi[-1]

        # トレンド判定
        uptrend = sma_short_current > sma_long_current
        downtrend = sma_short_current < sma_long_current

        # ポジションがない場合のエントリー判定
        if not self.position:
            # 買いエントリー条件
            if uptrend and rsi_current < self.oversold_threshold:
                if self.use_risk_management:
                    self.buy_with_risk_management()
                    logger.debug(
                        f"Buy signal: Price={current_price:.2f}, SMA_short={sma_short_current:.2f}, SMA_long={sma_long_current:.2f}, RSI={rsi_current:.2f}"
                    )
                else:
                    self.buy()

            # 売りエントリー条件（ショート）
            elif downtrend and rsi_current > self.overbought_threshold:
                if self.use_risk_management:
                    self.sell_with_risk_management()
                    logger.debug(
                        f"Sell signal: Price={current_price:.2f}, SMA_short={sma_short_current:.2f}, SMA_long={sma_long_current:.2f}, RSI={rsi_current:.2f}"
                    )
                else:
                    self.sell()

        # ポジションがある場合のエグジット判定
        else:
            # 買いポジションのエグジット
            if self.position.is_long:
                if downtrend or rsi_current > self.overbought_threshold:
                    self.position.close()
                    logger.debug(
                        f"Close long: Price={current_price:.2f}, RSI={rsi_current:.2f}"
                    )

            # 売りポジションのエグジット
            elif self.position.is_short:
                if uptrend or rsi_current < self.oversold_threshold:
                    self.position.close()
                    logger.debug(
                        f"Close short: Price={current_price:.2f}, RSI={rsi_current:.2f}"
                    )


class SMARSIStrategyOptimized(SMARSIStrategy):
    """
    最適化されたSMA+RSI戦略

    基本戦略に追加の条件とフィルターを加えた改良版
    """

    # 追加パラメータ
    volume_filter = True
    volume_threshold = 1.5  # 平均出来高の倍数
    volume_period = 20

    # RSIの追加条件
    rsi_confirmation_bars = 2  # RSI条件の確認期間

    def init(self):
        """戦略の初期化（最適化版）"""
        super().init()

        # 出来高移動平均
        if self.volume_filter:
            self.volume_ma = self.I(SMA, self.data.Volume, self.volume_period)

        logger.info("Optimized SMA+RSI Strategy initialized")

    def next(self):
        """各バーでの戦略実行（最適化版）"""
        # 基本条件チェック
        if (
            len(self.sma_short_line) < max(self.sma_short, self.sma_long)
            or len(self.rsi) < self.rsi_period
        ):
            return

        current_price = self.data.Close[-1]
        sma_short_current = self.sma_short_line[-1]
        sma_long_current = self.sma_long_line[-1]
        rsi_current = self.rsi[-1]

        # 出来高フィルター
        volume_condition = True
        if self.volume_filter and len(self.volume_ma) >= self.volume_period:
            current_volume = self.data.Volume[-1]
            avg_volume = self.volume_ma[-1]
            volume_condition = current_volume > (avg_volume * self.volume_threshold)

        # RSI確認条件（複数バーでの確認）
        rsi_confirmation = True
        if len(self.rsi) >= self.rsi_confirmation_bars:
            for i in range(1, self.rsi_confirmation_bars + 1):
                if not (
                    self.rsi[-i] < self.oversold_threshold
                    or self.rsi[-i] > self.overbought_threshold
                ):
                    rsi_confirmation = False
                    break

        # トレンド判定
        uptrend = sma_short_current > sma_long_current
        downtrend = sma_short_current < sma_long_current

        # ポジションがない場合のエントリー判定
        if not self.position:
            # 買いエントリー条件（追加フィルター付き）
            if (
                uptrend
                and rsi_current < self.oversold_threshold
                and volume_condition
                and rsi_confirmation
            ):
                if self.use_risk_management:
                    self.buy_with_risk_management()
                else:
                    self.buy()

            # 売りエントリー条件（追加フィルター付き）
            elif (
                downtrend
                and rsi_current > self.overbought_threshold
                and volume_condition
                and rsi_confirmation
            ):
                if self.use_risk_management:
                    self.sell_with_risk_management()
                else:
                    self.sell()

        # ポジションがある場合のエグジット判定（基本戦略と同じ）
        else:
            if self.position.is_long:
                if downtrend or rsi_current > self.overbought_threshold:
                    self.position.close()
            elif self.position.is_short:
                if uptrend or rsi_current < self.oversold_threshold:
                    self.position.close()

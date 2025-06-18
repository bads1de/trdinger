"""
リスク管理機能付きSMAクロス戦略

backtesting.pyの組み込みSL/TP機能を活用した改良版SMAクロス戦略
"""

from backtesting import Strategy
from backtesting.lib import crossover
from .indicators import SMA
from .risk_management import RiskManagementMixin
import logging

logger = logging.getLogger(__name__)


class EnhancedSMACrossStrategy(RiskManagementMixin, Strategy):
    """
    リスク管理機能付きSMAクロス戦略

    従来のSMAクロス戦略にリスク管理機能を統合した改良版。
    backtesting.pyの組み込みSL/TP機能を活用します。

    パラメータ:
        n1: 短期SMAの期間（デフォルト: 20）
        n2: 長期SMAの期間（デフォルト: 50）
        sl_pct: ストップロス率（デフォルト: 0.02 = 2%）
        tp_pct: テイクプロフィット率（デフォルト: 0.05 = 5%）
        use_risk_management: リスク管理機能を使用するかどうか（デフォルト: True）
    """

    # デフォルトパラメータ
    n1 = 20
    n2 = 50
    sl_pct = 0.02  # 2%
    tp_pct = 0.05  # 5%
    use_risk_management = True
    min_risk_reward_ratio = 1.5

    def init(self):
        """
        戦略の初期化

        SMAインジケーターとリスク管理機能を初期化します。
        """
        # SMAインジケーターの初期化
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)  # 短期SMA
        self.sma2 = self.I(SMA, close, self.n2)  # 長期SMA

        # リスク管理機能の初期化
        if self.use_risk_management:
            self.setup_risk_management(sl_pct=self.sl_pct, tp_pct=self.tp_pct)
            logger.info(
                f"Risk management enabled: SL={self.sl_pct:.1%}, TP={self.tp_pct:.1%}"
            )
        else:
            logger.info("Risk management disabled")

    def next(self):
        """
        売買ロジック

        SMAクロスオーバーを検出し、リスク管理機能付きで売買を実行します。
        """
        # ゴールデンクロス: 短期SMAが長期SMAを上抜け → 買いシグナル
        if crossover(self.sma1, self.sma2):  # type: ignore
            if self.use_risk_management:
                # リスク管理機能付きの買い注文
                self.buy_with_risk_management()
            else:
                # 従来の買い注文
                self.buy()

        # デッドクロス: 短期SMAが長期SMAを下抜け → 売りシグナル
        elif crossover(self.sma2, self.sma1):  # type: ignore
            if self.position:
                # ポジションがある場合は決済
                self.position.close()

    def validate_parameters(self) -> bool:
        """
        パラメータの妥当性を検証

        Returns:
            パラメータが有効な場合True

        Raises:
            ValueError: パラメータが無効な場合
        """
        if self.n1 <= 0 or self.n2 <= 0:
            raise ValueError("SMA periods must be positive integers")

        if self.n1 >= self.n2:
            raise ValueError("Short period (n1) must be less than long period (n2)")

        if self.use_risk_management:
            if self.sl_pct <= 0 or self.sl_pct >= 1:
                raise ValueError("Stop loss percentage must be between 0 and 1")

            if self.tp_pct <= 0 or self.tp_pct >= 1:
                raise ValueError("Take profit percentage must be between 0 and 1")

        return True

    def get_strategy_description(self) -> str:
        """
        戦略の詳細説明を取得

        Returns:
            戦略の説明文
        """
        risk_info = ""
        if self.use_risk_management:
            risk_info = f"""
        リスク管理:
        - ストップロス: {self.sl_pct:.1%}
        - テイクプロフィット: {self.tp_pct:.1%}
        """

        return f"""
        リスク管理機能付きSMAクロス戦略 ({self.n1}/{self.n2})
        
        エントリー条件:
        - ゴールデンクロス: SMA({self.n1}) > SMA({self.n2}) → 買い
        - デッドクロス: SMA({self.n1}) < SMA({self.n2}) → 決済
        {risk_info}
        特徴:
        - トレンドフォロー戦略
        - 自動ストップロス・テイクプロフィット
        - backtesting.py組み込み機能を活用
        
        推奨市場:
        - トレンドが明確な市場
        - ボラティリティが適度な市場
        """


class EnhancedSMACrossStrategyWithTrailing(EnhancedSMACrossStrategy):
    """
    トレーリングストップ機能付きSMAクロス戦略

    基本のリスク管理機能に加えて、トレーリングストップ機能を追加した戦略
    """

    # 追加パラメータ
    use_trailing_stop = True
    trailing_update_frequency = 1  # 何バーごとにトレーリングストップを更新するか

    def init(self):
        """戦略の初期化（トレーリングストップ版）"""
        super().init()
        self.bar_count = 0

        if self.use_trailing_stop:
            logger.info("Trailing stop enabled")

    def next(self):
        """売買ロジック（トレーリングストップ版）"""
        # 基本の売買ロジック
        super().next()

        # トレーリングストップの更新
        if self.use_trailing_stop and self.use_risk_management:
            self.bar_count += 1

            # 指定された頻度でトレーリングストップを更新
            if self.bar_count % self.trailing_update_frequency == 0:
                self.update_trailing_stop()


class EnhancedSMACrossStrategyWithVolume(EnhancedSMACrossStrategy):
    """
    出来高フィルター付きSMAクロス戦略

    基本のリスク管理機能に加えて、出来高フィルターを追加した戦略
    """

    # 追加パラメータ
    volume_threshold = 1.5  # 平均出来高の倍数
    volume_period = 20  # 出来高移動平均の期間

    def init(self):
        """戦略の初期化（出来高フィルター版）"""
        super().init()

        # 出来高移動平均
        self.volume_ma = self.I(SMA, self.data.Volume, self.volume_period)

        logger.info(f"Volume filter enabled: threshold={self.volume_threshold}x")

    def next(self):
        """売買ロジック（出来高フィルター版）"""
        # 出来高フィルターの確認
        if len(self.volume_ma) > 0:
            current_volume = self.data.Volume[-1]
            avg_volume = self.volume_ma[-1]

            # 出来高が閾値を下回る場合はシグナルを無視
            if current_volume < avg_volume * self.volume_threshold:
                return

        # 基本の売買ロジック
        super().next()


class EnhancedSMACrossStrategyAdvanced(EnhancedSMACrossStrategy):
    """
    高度なSMAクロス戦略

    複数の改良機能を組み合わせた最上位版戦略
    """

    # 高度なパラメータ
    use_trailing_stop = True
    use_volume_filter = True
    volume_threshold = 1.5
    volume_period = 20
    trailing_update_frequency = 1

    # ATRベースのリスク管理
    use_atr_based_risk = False
    atr_period = 14
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    def init(self):
        """戦略の初期化（高度版）"""
        super().init()

        # 出来高移動平均
        if self.use_volume_filter:
            self.volume_ma = self.I(SMA, self.data.Volume, self.volume_period)

        # ATR（Average True Range）
        if self.use_atr_based_risk:
            from .indicators import ATR

            self.atr = self.I(
                ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period
            )

        self.bar_count = 0

        logger.info("Advanced SMA Cross Strategy initialized")

    def next(self):
        """売買ロジック（高度版）"""
        # 出来高フィルター
        if self.use_volume_filter and len(self.volume_ma) > 0:
            current_volume = self.data.Volume[-1]
            avg_volume = self.volume_ma[-1]

            if current_volume < avg_volume * self.volume_threshold:
                return

        # ゴールデンクロス
        if crossover(self.sma1, self.sma2):  # type: ignore
            if self.use_risk_management:
                if self.use_atr_based_risk and len(self.atr) > 0:
                    # ATRベースのリスク管理
                    current_price = self.data.Close[-1]
                    atr_value = self.atr[-1]

                    sl_price = current_price - (atr_value * self.atr_sl_multiplier)
                    tp_price = current_price + (atr_value * self.atr_tp_multiplier)

                    self.buy(sl=sl_price, tp=tp_price)
                else:
                    # パーセンテージベースのリスク管理
                    self.buy_with_risk_management()
            else:
                self.buy()

        # デッドクロス
        elif crossover(self.sma2, self.sma1):  # type: ignore
            if self.position:
                self.position.close()

        # トレーリングストップの更新
        if self.use_trailing_stop and self.use_risk_management:
            self.bar_count += 1

            if self.bar_count % self.trailing_update_frequency == 0:
                self.update_trailing_stop()

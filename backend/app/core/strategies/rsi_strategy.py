"""
RSI戦略

Relative Strength Index Strategy
RSIの過買い・過売りレベルを利用した逆張り戦略
"""

from backtesting import Strategy
from .indicators import RSI


class RSIStrategy(Strategy):
    """
    RSI戦略

    RSIが過売りレベル（通常30以下）で買い、
    RSIが過買いレベル（通常70以上）で売る逆張り戦略。

    パラメータ:
        period: RSIの計算期間（デフォルト: 14）
        oversold: 過売りレベル（デフォルト: 30）
        overbought: 過買いレベル（デフォルト: 70）
    """

    # デフォルトパラメータ（クラス変数として定義）
    period = 14
    oversold = 30
    overbought = 70

    def __init__(self, broker=None, data=None, params=None):
        """
        戦略の初期化

        Args:
            broker: ブローカーオブジェクト（backtesting.pyから渡される）
            data: データオブジェクト（backtesting.pyから渡される）
            params: パラメータ辞書（backtesting.pyから渡される）

        パラメータはクラス変数として設定されます。
        最適化時にはbacktesting.pyが自動的にクラス変数を変更します。
        """
        super().__init__(broker, data, params)

    def init(self):
        """
        指標の初期化

        RSIインジケーターを計算し、戦略で使用できるように設定します。
        """
        # 終値データを取得
        close = self.data.Close

        # RSIインジケーターを初期化
        self.rsi = self.I(RSI, close, self.period)

    def next(self):
        """
        売買ロジック

        各バーで実行される売買判定ロジック。
        RSIの過買い・過売りレベルを検出して売買シグナルを生成します。
        """
        current_rsi = self.rsi[-1]

        # 過売り状態からの回復 → 買いシグナル
        if current_rsi <= self.oversold and not self.position:
            self.buy()

        # 過買い状態への到達 → 売りシグナル
        elif current_rsi >= self.overbought and self.position:
            self.sell()

    def validate_parameters(self) -> bool:
        """
        パラメータの妥当性を検証

        Returns:
            パラメータが有効な場合True

        Raises:
            ValueError: パラメータが無効な場合
        """
        if self.period <= 0:
            raise ValueError("RSI期間は正の整数である必要があります。")

        if not (0 <= self.oversold <= 100):
            raise ValueError("売られすぎレベルは0から100の間である必要があります。")

        if not (0 <= self.overbought <= 100):
            raise ValueError("買われすぎレベルは0から100の間である必要があります。")

        if self.oversold >= self.overbought:
            raise ValueError(
                "売られすぎレベルは買われすぎレベルより小さくする必要があります。"
            )

        return True

    def get_strategy_description(self) -> str:
        """
        戦略の詳細説明を取得

        Returns:
            戦略の説明文
        """
        return f"""
        RSI戦略 (期間: {self.period})
        
        エントリー条件:
        - 過売り: RSI <= {self.oversold} → 買い
        - 過買い: RSI >= {self.overbought} → 売り
        
        特徴:
        - 逆張り戦略
        - レンジ相場で効果的
        - トレンド相場では逆行する可能性
        
        推奨市場:
        - ボラティリティが高い市場
        - レンジ相場
        """

    def get_current_signals(self) -> dict:
        """
        現在のシグナル状況を取得

        Returns:
            現在のシグナル情報を含む辞書
        """
        if len(self.rsi) == 0:
            return {"error": "シグナル計算のためのデータが不足しています。"}

        current_rsi = self.rsi[-1]

        # シグナルの判定
        if current_rsi <= self.oversold:
            signal = "BUY"
            signal_strength = (self.oversold - current_rsi) / self.oversold
        elif current_rsi >= self.overbought:
            signal = "SELL"
            signal_strength = (current_rsi - self.overbought) / (100 - self.overbought)
        else:
            signal = "HOLD"
            signal_strength = 0.0

        return {
            "signal": signal,
            "signal_strength": round(signal_strength, 4),
            "rsi_value": round(current_rsi, 2),
            "oversold_level": self.oversold,
            "overbought_level": self.overbought,
            "current_price": round(self.data.Close[-1], 2),
        }


class RSIStrategyAdvanced(RSIStrategy):
    """
    高度なRSI戦略

    基本のRSI戦略に追加の条件を加えた改良版。
    """

    # 追加パラメータ
    rsi_smoothing = 3  # RSIの平滑化期間
    volume_confirmation = True  # 出来高確認フラグ
    trend_filter = True  # トレンドフィルター

    def init(self):
        """指標の初期化（拡張版）"""
        super().init()

        # 追加指標
        from .indicators import SMA

        # RSIの平滑化
        if self.rsi_smoothing > 1:
            self.rsi_smooth = self.I(SMA, self.rsi, self.rsi_smoothing)
        else:
            self.rsi_smooth = self.rsi

        # トレンドフィルター用のSMA
        if self.trend_filter:
            self.trend_sma = self.I(SMA, self.data.Close, 50)

        # 出来高移動平均
        if self.volume_confirmation:
            self.volume_ma = self.I(SMA, self.data.Volume, 20)

    def next(self):
        """売買ロジック（拡張版）"""
        current_rsi = self.rsi_smooth[-1]
        current_price = self.data.Close[-1]

        # トレンドフィルター
        if self.trend_filter:
            trend_direction = "up" if current_price > self.trend_sma[-1] else "down"
        else:
            trend_direction = "neutral"

        # 出来高確認
        volume_ok = True
        if self.volume_confirmation and len(self.volume_ma) > 0:
            volume_ok = self.data.Volume[-1] > self.volume_ma[-1]

        # 買いシグナル（過売り + 上昇トレンド + 出来高確認）
        if (
            current_rsi <= self.oversold
            and not self.position
            and volume_ok
            and (trend_direction in ["up", "neutral"])
        ):
            self.buy()

        # 売りシグナル（過買い + 下降トレンド + 出来高確認）
        elif (
            current_rsi >= self.overbought
            and self.position
            and volume_ok
            and (trend_direction in ["down", "neutral"])
        ):
            self.sell()


class RSIMeanReversionStrategy(RSIStrategy):
    """
    RSI平均回帰戦略

    RSIの極端な値からの平均回帰を狙う戦略
    """

    # 極端なレベル
    extreme_oversold = 20
    extreme_overbought = 80

    # 利確・損切りレベル
    take_profit_rsi = 50
    stop_loss_pct = 0.05  # 5%

    def init(self):
        """指標の初期化"""
        super().init()
        self.entry_price = None

    def next(self):
        """売買ロジック（平均回帰版）"""
        current_rsi = self.rsi[-1]
        current_price = self.data.Close[-1]

        # 極端な過売りからの買い
        if current_rsi <= self.extreme_oversold and not self.position:
            self.buy()
            self.entry_price = current_price

        # 極端な過買いからの売り
        elif current_rsi >= self.extreme_overbought and not self.position:
            self.sell()
            self.entry_price = current_price

        # 利確・損切り
        if self.position and self.entry_price:
            # RSIが中央値に戻った場合の利確
            if abs(current_rsi - 50) < 10:  # RSIが40-60の範囲
                self.position.close()
                self.entry_price = None

            # 損切り
            elif self.position.is_long:
                if current_price <= self.entry_price * (1 - self.stop_loss_pct):
                    self.position.close()
                    self.entry_price = None

            elif self.position.is_short:
                if current_price >= self.entry_price * (1 + self.stop_loss_pct):
                    self.position.close()
                    self.entry_price = None

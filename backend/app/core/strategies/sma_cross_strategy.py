"""
SMAクロス戦略

Simple Moving Average Crossover Strategy
短期移動平均と長期移動平均のクロスオーバーを利用した戦略
"""

from backtesting import Strategy
from backtesting.lib import crossover
from .indicators import SMA


class SMACrossStrategy(Strategy):
    """
    SMAクロス戦略

    短期SMAが長期SMAを上抜けした時に買い、
    短期SMAが長期SMAを下抜けした時に売る戦略。

    パラメータ:
        n1: 短期SMAの期間（デフォルト: 20）
        n2: 長期SMAの期間（デフォルト: 50）
    """

    # デフォルトパラメータ（クラス変数として定義）
    n1 = 20
    n2 = 50

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

        短期・長期のSMAを計算し、戦略で使用できるように設定します。
        """
        # 終値データを取得
        close = self.data.Close

        # SMAインジケーターを初期化
        self.sma1 = self.I(SMA, close, self.n1)  # 短期SMA
        self.sma2 = self.I(SMA, close, self.n2)  # 長期SMA

    def next(self):
        """
        売買ロジック

        各バーで実行される売買判定ロジック。
        SMAのクロスオーバーを検出して売買シグナルを生成します。
        """
        # ゴールデンクロス: 短期SMAが長期SMAを上抜け → 買いシグナル
        if crossover(self.sma1, self.sma2):
            self.buy()

        # デッドクロス: 短期SMAが長期SMAを下抜け → 売りシグナル
        elif crossover(self.sma2, self.sma1):
            self.sell()

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

        return True

    def get_strategy_description(self) -> str:
        """
        戦略の詳細説明を取得

        Returns:
            戦略の説明文
        """
        return f"""
        SMAクロス戦略 ({self.n1}/{self.n2})
        
        エントリー条件:
        - ゴールデンクロス: SMA({self.n1}) > SMA({self.n2}) → 買い
        - デッドクロス: SMA({self.n1}) < SMA({self.n2}) → 売り
        
        特徴:
        - トレンドフォロー戦略
        - シンプルで理解しやすい
        - レンジ相場では多くのダマシが発生する可能性
        
        推奨市場:
        - トレンドが明確な市場
        - ボラティリティが適度な市場
        """

    def get_current_signals(self) -> dict:
        """
        現在のシグナル状況を取得

        Returns:
            現在のシグナル情報を含む辞書
        """
        if len(self.sma1) == 0 or len(self.sma2) == 0:
            return {"error": "Insufficient data for signal calculation"}

        current_sma1 = self.sma1[-1]
        current_sma2 = self.sma2[-1]

        # シグナルの判定
        if current_sma1 > current_sma2:
            signal = "BULLISH"
            signal_strength = (current_sma1 - current_sma2) / current_sma2 * 100
        elif current_sma1 < current_sma2:
            signal = "BEARISH"
            signal_strength = (current_sma2 - current_sma1) / current_sma2 * 100
        else:
            signal = "NEUTRAL"
            signal_strength = 0.0

        return {
            "signal": signal,
            "signal_strength": round(signal_strength, 4),
            "sma_short": round(current_sma1, 2),
            "sma_long": round(current_sma2, 2),
            "current_price": round(self.data.Close[-1], 2),
        }


class SMACrossStrategyOptimized(SMACrossStrategy):
    """
    最適化されたSMAクロス戦略

    基本のSMAクロス戦略に追加の条件を加えた改良版。
    """

    # 追加パラメータ
    volume_threshold = 1.5  # 出来高フィルター（平均出来高の倍数）
    atr_multiplier = 2.0  # ATRベースのストップロス

    def init(self):
        """指標の初期化（拡張版）"""
        super().init()

        # 追加指標
        from .indicators import ATR

        # 出来高移動平均
        self.volume_ma = self.I(SMA, self.data.Volume, 20)

        # ATR（Average True Range）
        self.atr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, 14)

    def next(self):
        """売買ロジック（拡張版）"""
        # 出来高フィルター
        if self.data.Volume[-1] < self.volume_ma[-1] * self.volume_threshold:
            return  # 出来高が不十分な場合はシグナルを無視

        # 基本のSMAクロス判定
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()

        # ATRベースのストップロス（ポジションがある場合）
        if self.position:
            current_price = self.data.Close[-1]
            atr_value = self.atr[-1]

            if self.position.is_long:
                stop_loss = current_price - (atr_value * self.atr_multiplier)
                if current_price <= stop_loss:
                    self.position.close()

            elif self.position.is_short:
                stop_loss = current_price + (atr_value * self.atr_multiplier)
                if current_price >= stop_loss:
                    self.position.close()

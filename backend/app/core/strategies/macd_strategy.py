"""
MACD戦略

Moving Average Convergence Divergence Strategy
MACDラインとシグナルラインのクロスオーバーを利用したトレンドフォロー戦略
"""

from backtesting import Strategy
from backtesting.lib import crossover
from .indicators import MACD, SMA
from .base_strategy import BaseStrategy


class MACDStrategy(Strategy):
    """
    MACD戦略
    
    MACDラインがシグナルラインを上抜けした時に買い、
    MACDラインがシグナルラインを下抜けした時に売る戦略。
    
    パラメータ:
        fast_period: 短期EMAの期間（デフォルト: 12）
        slow_period: 長期EMAの期間（デフォルト: 26）
        signal_period: シグナルラインの期間（デフォルト: 9）
    """

    # パラメータ（最適化可能）
    fast_period = 12
    slow_period = 26
    signal_period = 9

    def init(self):
        """
        指標の初期化
        
        MACDインジケーターを計算し、戦略で使用できるように設定します。
        """
        # 終値データを取得
        close = self.data.Close
        
        # MACDインジケーターを初期化
        self.macd_line, self.signal_line, self.histogram = self.I(
            MACD, close, self.fast_period, self.slow_period, self.signal_period
        )

    def next(self):
        """
        売買ロジック
        
        各バーで実行される売買判定ロジック。
        MACDとシグナルラインのクロスオーバーを検出して売買シグナルを生成します。
        """
        # MACDラインがシグナルラインを上抜け → 買いシグナル
        if crossover(self.macd_line, self.signal_line):
            self.buy()
        
        # MACDラインがシグナルラインを下抜け → 売りシグナル
        elif crossover(self.signal_line, self.macd_line):
            self.sell()

    def validate_parameters(self) -> bool:
        """
        パラメータの妥当性を検証
        
        Returns:
            パラメータが有効な場合True
            
        Raises:
            ValueError: パラメータが無効な場合
        """
        if self.fast_period <= 0 or self.slow_period <= 0 or self.signal_period <= 0:
            raise ValueError("All periods must be positive integers")
        
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        return True

    def get_strategy_description(self) -> str:
        """
        戦略の詳細説明を取得
        
        Returns:
            戦略の説明文
        """
        return f"""
        MACD戦略 ({self.fast_period}/{self.slow_period}/{self.signal_period})
        
        エントリー条件:
        - ブルクロス: MACD > Signal → 買い
        - ベアクロス: MACD < Signal → 売り
        
        特徴:
        - トレンドフォロー戦略
        - 中期的なトレンド転換を捉える
        - レンジ相場では多くのダマシが発生する可能性
        
        推奨市場:
        - トレンドが明確な市場
        - 中期的な投資期間
        """

    def get_current_signals(self) -> dict:
        """
        現在のシグナル状況を取得
        
        Returns:
            現在のシグナル情報を含む辞書
        """
        if len(self.macd_line) == 0 or len(self.signal_line) == 0:
            return {"error": "Insufficient data for signal calculation"}
        
        current_macd = self.macd_line[-1]
        current_signal = self.signal_line[-1]
        current_histogram = self.histogram[-1]
        
        # シグナルの判定
        if current_macd > current_signal:
            signal = "BULLISH"
            signal_strength = abs(current_histogram) / abs(current_macd) if current_macd != 0 else 0
        elif current_macd < current_signal:
            signal = "BEARISH"
            signal_strength = abs(current_histogram) / abs(current_macd) if current_macd != 0 else 0
        else:
            signal = "NEUTRAL"
            signal_strength = 0.0
        
        return {
            "signal": signal,
            "signal_strength": round(signal_strength, 4),
            "macd_line": round(current_macd, 4),
            "signal_line": round(current_signal, 4),
            "histogram": round(current_histogram, 4),
            "current_price": round(self.data.Close[-1], 2)
        }


class MACDDivergenceStrategy(MACDStrategy):
    """
    MACDダイバージェンス戦略
    
    価格とMACDの乖離（ダイバージェンス）を検出する高度な戦略
    """

    # ダイバージェンス検出パラメータ
    lookback_period = 20  # ダイバージェンス検出期間
    min_divergence_strength = 0.5  # 最小ダイバージェンス強度

    def init(self):
        """指標の初期化（拡張版）"""
        super().init()
        
        # 価格とMACDの高値・安値を追跡するための変数
        self.price_highs = []
        self.price_lows = []
        self.macd_highs = []
        self.macd_lows = []

    def next(self):
        """売買ロジック（ダイバージェンス版）"""
        current_price = self.data.Close[-1]
        current_macd = self.macd_line[-1]
        
        # 基本のMACDシグナル
        if crossover(self.macd_line, self.signal_line):
            # ブルダイバージェンスの確認
            if self._detect_bullish_divergence():
                self.buy()
        
        elif crossover(self.signal_line, self.macd_line):
            # ベアダイバージェンスの確認
            if self._detect_bearish_divergence():
                self.sell()

    def _detect_bullish_divergence(self) -> bool:
        """ブルダイバージェンスの検出"""
        if len(self.data.Close) < self.lookback_period:
            return False
        
        # 最近の安値を検出
        recent_prices = self.data.Close[-self.lookback_period:]
        recent_macd = self.macd_line[-self.lookback_period:]
        
        # 価格が下降トレンドでMACDが上昇トレンドの場合
        price_trend = recent_prices[-1] - recent_prices[0]
        macd_trend = recent_macd[-1] - recent_macd[0]
        
        return price_trend < 0 and macd_trend > 0

    def _detect_bearish_divergence(self) -> bool:
        """ベアダイバージェンスの検出"""
        if len(self.data.Close) < self.lookback_period:
            return False
        
        # 最近の高値を検出
        recent_prices = self.data.Close[-self.lookback_period:]
        recent_macd = self.macd_line[-self.lookback_period:]
        
        # 価格が上昇トレンドでMACDが下降トレンドの場合
        price_trend = recent_prices[-1] - recent_prices[0]
        macd_trend = recent_macd[-1] - recent_macd[0]
        
        return price_trend > 0 and macd_trend < 0


class MACDTrendStrategy(MACDStrategy):
    """
    MACDトレンド戦略
    
    MACDのゼロライン突破とトレンドフィルターを組み合わせた戦略
    """

    # トレンドフィルターパラメータ
    trend_sma_period = 50
    use_zero_line_filter = True  # ゼロライン突破フィルター

    def init(self):
        """指標の初期化（トレンド版）"""
        super().init()
        
        # トレンドフィルター用のSMA
        self.trend_sma = self.I(SMA, self.data.Close, self.trend_sma_period)

    def next(self):
        """売買ロジック（トレンド版）"""
        current_price = self.data.Close[-1]
        current_macd = self.macd_line[-1]
        current_signal = self.signal_line[-1]
        
        # トレンド方向の判定
        trend_direction = "up" if current_price > self.trend_sma[-1] else "down"
        
        # ゼロライン突破フィルター
        macd_above_zero = current_macd > 0 if self.use_zero_line_filter else True
        macd_below_zero = current_macd < 0 if self.use_zero_line_filter else True
        
        # 買いシグナル（上昇トレンド + MACDクロス + ゼロライン上）
        if (crossover(self.macd_line, self.signal_line) and 
            trend_direction == "up" and 
            macd_above_zero):
            self.buy()
        
        # 売りシグナル（下降トレンド + MACDクロス + ゼロライン下）
        elif (crossover(self.signal_line, self.macd_line) and 
              trend_direction == "down" and 
              macd_below_zero):
            self.sell()


class MACDScalpingStrategy(MACDStrategy):
    """
    MACDスキャルピング戦略
    
    短期的なMACDシグナルを利用した高頻度取引戦略
    """

    # スキャルピングパラメータ
    fast_period = 5   # より短期
    slow_period = 13  # より短期
    signal_period = 5 # より短期
    
    # 利確・損切り
    take_profit_pct = 0.02  # 2%利確
    stop_loss_pct = 0.01    # 1%損切り

    def init(self):
        """指標の初期化（スキャルピング版）"""
        super().init()
        self.entry_price = None

    def next(self):
        """売買ロジック（スキャルピング版）"""
        current_price = self.data.Close[-1]
        
        # エントリーシグナル
        if crossover(self.macd_line, self.signal_line) and not self.position:
            self.buy()
            self.entry_price = current_price
        
        elif crossover(self.signal_line, self.macd_line) and not self.position:
            self.sell()
            self.entry_price = current_price
        
        # 利確・損切り
        if self.position and self.entry_price:
            if self.position.is_long:
                # 利確
                if current_price >= self.entry_price * (1 + self.take_profit_pct):
                    self.position.close()
                    self.entry_price = None
                # 損切り
                elif current_price <= self.entry_price * (1 - self.stop_loss_pct):
                    self.position.close()
                    self.entry_price = None
            
            elif self.position.is_short:
                # 利確
                if current_price <= self.entry_price * (1 - self.take_profit_pct):
                    self.position.close()
                    self.entry_price = None
                # 損切り
                elif current_price >= self.entry_price * (1 + self.stop_loss_pct):
                    self.position.close()
                    self.entry_price = None

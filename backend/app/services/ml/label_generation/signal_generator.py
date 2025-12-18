"""
SignalGenerator - メタラベリング用のシグナル生成クラス

Primary Model として機能するルールベースのシグナルジェネレーター。
ML モデル（Secondary Model）が判定するための「候補イベント」を生成します。
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    メタラベリング用のシグナル生成クラス

    ボリンジャーバンド、ドンチャンチャネル、出来高急増などの
    テクニカル指標に基づいて、潜在的なトレード候補（イベント）を検出します。

    これらのシグナルは「高Recall」を目指し、多くの候補を広く拾います。
    その後、ML モデル（Secondary Model）が「高Precision」で
    真のトレード機会をフィルタリングします。

    Examples:
        >>> generator = SignalGenerator()
        >>> bb_events = generator.get_bb_breakout_events(ohlcv_df)
        >>> combined_events = generator.get_combined_events(
        ...     ohlcv_df,
        ...     use_bb=True,
        ...     use_volume=True
        ... )
    """

    def _validate_df(self, df: pd.DataFrame, window: int) -> bool:
        """データの最小数チェック"""
        if len(df) < window:
            logger.warning(f"データ数（{len(df)}）が不足（{window}必要）ため空の結果を返します")
            return False
        return True

    def get_bb_breakout_events(
        self,
        df: pd.DataFrame,
        window: int = 20,
        dev: float = 2.0,
        price_column: str = "close",
    ) -> pd.DatetimeIndex:
        """ボリンジャーバンド ブレイクアウトイベントを検出"""
        if not self._validate_df(df, window):
            return pd.DatetimeIndex([])

        prices = df[price_column]
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        up, dn = sma + (dev * std), sma - (dev * std)
        up_mask, dn_mask = prices > up, prices < dn
        
        # 新規ブレイクアウトのみ
        mask = (up_mask & ~up_mask.shift(1, fill_value=False)) | \
               (dn_mask & ~dn_mask.shift(1, fill_value=False))

        return df.index[mask]

    def get_donchian_breakout_events(
        self,
        df: pd.DataFrame,
        window: int = 20,
        price_column_high: str = "high",
        price_column_low: str = "low",
    ) -> pd.DatetimeIndex:
        """ドンチャンチャネル ブレイクアウトイベントを検出"""
        if not self._validate_df(df, window):
            return pd.DatetimeIndex([])

        h, low_p = df[price_column_high], df[price_column_low]
        ph, pl = h.rolling(window).max().shift(1), low_p.rolling(window).min().shift(1)

        mask = ((h > ph) & ph.notna()) | ((low_p < pl) & pl.notna())
        return df.index[mask]

    def get_volume_spike_events(
        self,
        df: pd.DataFrame,
        window: int = 20,
        multiplier: float = 2.0,
        volume_column: str = "volume",
    ) -> pd.DatetimeIndex:
        """出来高急増イベントを検出"""
        if not self._validate_df(df, window):
            return pd.DatetimeIndex([])

        v = df[volume_column]
        avg_v = v.rolling(window).mean().shift(1)
        return df.index[(v >= avg_v * multiplier) & avg_v.notna()]

    def get_combined_events(
        self,
        df: pd.DataFrame,
        use_bb: bool = True,
        use_donchian: bool = True,
        use_volume: bool = True,
        bb_window: int = 20,
        bb_dev: float = 2.0,
        donchian_window: int = 20,
        volume_window: int = 20,
        volume_multiplier: float = 2.0,
    ) -> pd.DatetimeIndex:
        """複数のシグナルを組み合わせてイベントを取得"""
        events = []
        if use_bb:
            events.append(self.get_bb_breakout_events(df, bb_window, bb_dev))
        if use_donchian:
            events.append(self.get_donchian_breakout_events(df, donchian_window))
        if use_volume:
            events.append(self.get_volume_spike_events(df, volume_window, volume_multiplier))

        if not events:
            return pd.DatetimeIndex([])

        combined = events[0]
        for e in events[1:]:
            combined = combined.union(e)
        return combined.sort_values()




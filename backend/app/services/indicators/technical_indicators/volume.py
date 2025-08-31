"""
出来高系テクニカル指標

登録してあるテクニカルの一覧:
- AD (Accumulation/Distribution Line)
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)
- MFI (Money Flow Index)
- EOM (Ease of Movement)
- NVI (Negative Volume Index)
- PVI (Positive Volume Index)
- AOBV (Archer On-Balance Volume)
- EFI (Elder's Force Index)
- PVOL (Price-Volume)
- PVR (Price Volume Rank)
- VP (Volume Price)
"""

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import handle_pandas_ta_errors


class VolumeIndicators:
    """
    出来高系指標クラス
    """

    @staticmethod
    @handle_pandas_ta_errors
    def ad(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """チャイキンA/Dライン"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        result = ta.ad(
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        return result if result is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def adosc(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        fast: int = 3,
        slow: int = 10,
    ) -> pd.Series:
        """チャイキンA/Dオシレーター"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        result = ta.adosc(
            high=high,
            low=low,
            close=close,
            volume=volume,
            fast=fast,
            slow=slow,
        )
        return result if result is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def obv(
        close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """オンバランスボリューム"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        result = ta.obv(close=close, volume=volume)
        return result if result is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def nvi(
        close: pd.Series, volume: pd.Series, length: int = 13, initial: int = 1000
    ) -> pd.Series:
        """Negative Volume Index"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")
        df = ta.nvi(close=close, volume=volume, length=length, initial=initial)
        return df if df is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def pvi(
        close: pd.Series, volume: pd.Series, length: int = 13, initial: int = 1000
    ) -> pd.Series:
        """Positive Volume Index"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")
        df = ta.pvi(close=close, volume=volume, length=length, initial=initial)
        return df if df is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        anchor: str | None = None,
    ) -> pd.Series:
        """Volume Weighted Average Price"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        # pandas-taのVWAP関数は時系列インデックスが必要なため、独自実装を使用
        try:
            # pandas-taのVWAPを試行（時系列インデックスがある場合）
            if hasattr(high.index, "to_period"):
                df = ta.vwap(high=high, low=low, close=close, volume=volume, anchor=anchor)
                return df if df is not None else pd.Series([], dtype=float)
        except Exception:
            pass

        # フォールバック: 独自VWAP実装
        # 典型価格 = (H + L + C) / 3
        typical_price = (high + low + close) / 3

        # VWAP = Σ(典型価格 × 出来高) / Σ(出来高)
        # 累積で計算
        cumulative_pv = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()

        # ゼロ除算を避ける - pandas使用
        vwap = np.where(
            cumulative_volume != 0, cumulative_pv / cumulative_volume, typical_price
        )

        return pd.Series(vwap, index=high.index if hasattr(high, 'index') else None)

    @staticmethod
    @handle_pandas_ta_errors
    def eom(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        length: int = 14,
        divisor: int = 100000000,
        drift: int = 1,
    ) -> pd.Series:
        """Ease of Movement"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")
        df = ta.eom(high=high, low=low, close=close, volume=volume, length=length, divisor=divisor, drift=drift)
        return df if df is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def kvo(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        fast: int = 34,
        slow: int = 55,
    ) -> tuple[pd.Series, pd.Series]:
        """Klinger Volume Oscillator"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")
        df = ta.kvo(high=high, low=low, close=close, volume=volume, fast=fast, slow=slow)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.Series([], dtype=float), pd.Series([], dtype=float)
        # 2つの列を返す
        return df.iloc[:, 0], df.iloc[:, 1]  # KVO_34_55_13, KVOs_34_55_13

    @staticmethod
    @handle_pandas_ta_errors
    def pvt(
        close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """Price Volume Trend"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")
        df = ta.pvt(close=close, volume=volume)
        return df if df is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def cmf(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        length: int = 20,
    ) -> pd.Series:
        """Chaikin Money Flow"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")
        df = ta.cmf(high=high, low=low, close=close, volume=volume, length=length)
        return df if df is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def aobv(
        close: pd.Series,
        volume: pd.Series,
        fast: int = 5,
        slow: int = 10,
        max_lookback: int = 2,
        min_lookback: int = 2,
        mamode: str = "ema",
    ) -> tuple[pd.Series, ...]:
        """Archer On-Balance Volume"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")
        df = ta.aobv(close=close, volume=volume, fast=fast, slow=slow, max_lookback=max_lookback, min_lookback=min_lookback, mamode=mamode)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            # 7つの空のシリーズを返す
            empty = pd.Series([], dtype=float)
            return empty, empty, empty, empty, empty, empty, empty
        # DataFrameの各列をシリーズとして返す
        return tuple(df.iloc[:, i] for i in range(df.shape[1]))

    @staticmethod
    @handle_pandas_ta_errors
    def efi(
        close: pd.Series, volume: pd.Series, length: int = 13, mamode: str = "ema", drift: int = 1
    ) -> pd.Series:
        """Elder's Force Index"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")
        df = ta.efi(close=close, volume=volume, length=length, mamode=mamode, drift=drift)
        return df if df is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def pvol(
        close: pd.Series, volume: pd.Series, signed: bool = True
    ) -> pd.Series:
        """Price-Volume"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")
        df = ta.pvol(close=close, volume=volume, signed=signed)
        return df if df is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def pvr(
        close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """Price Volume Rank"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")
        df = ta.pvr(close=close, volume=volume)
        return df if df is not None else pd.Series([], dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def vp(
        close: pd.Series, volume: pd.Series, width: int = 10
    ) -> tuple[pd.Series, ...]:
        """Volume Price Confirmation (Volume Profile)"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")
        df = ta.vp(close=close, volume=volume, width=width)

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            # 空の結果を返す
            return pd.Series([], dtype=float), pd.Series([], dtype=float), pd.Series([], dtype=float), pd.Series([], dtype=float), pd.Series([], dtype=float), pd.Series([], dtype=float)

        # DataFrameであることを確認し、各列を個別のpandas Seriesとして返す
        if isinstance(df, pd.DataFrame) and not df.empty:
            return (
                df.iloc[:, 0],  # low_0 (価格範囲下限) - pd.Series
                df.iloc[:, 1],  # mean_0 (価格範囲平均) - pd.Series
                df.iloc[:, 2],  # high_0 (価格範囲上限) - pd.Series
                df.iloc[:, 3],  # pos_1 (陽線出来高) - pd.Series
                df.iloc[:, 4],  # neg_1 (陰線出来高) - pd.Series
                df.iloc[:, 5],  # total_1 (総出来高) - pd.Series
            )
        else:
            # 空の結果を返す
            return pd.Series([], dtype=float), pd.Series([], dtype=float), pd.Series([], dtype=float), pd.Series([], dtype=float), pd.Series([], dtype=float), pd.Series([], dtype=float)

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

from typing import Union

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
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        volume: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """チャイキンA/Dライン"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close
        volume_series = pd.Series(volume) if isinstance(volume, np.ndarray) else volume

        result = ta.ad(
            high=high_series,
            low=low_series,
            close=close_series,
            volume=volume_series,
        )
        return result.values if result is not None and hasattr(result, "values") else np.asarray(result) if result is not None else np.array([])

    @staticmethod
    @handle_pandas_ta_errors
    def adosc(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        volume: Union[np.ndarray, pd.Series],
        fast: int = 3,
        slow: int = 10,
    ) -> np.ndarray:
        """チャイキンA/Dオシレーター"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close
        volume_series = pd.Series(volume) if isinstance(volume, np.ndarray) else volume

        result = ta.adosc(
            high=high_series,
            low=low_series,
            close=close_series,
            volume=volume_series,
            fast=fast,
            slow=slow,
        )
        return result.values if result is not None and hasattr(result, "values") else np.asarray(result) if result is not None else np.array([])

    @staticmethod
    @handle_pandas_ta_errors
    def obv(
        close: Union[np.ndarray, pd.Series], volume: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """オンバランスボリューム"""
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close
        volume_series = pd.Series(volume) if isinstance(volume, np.ndarray) else volume

        result = ta.obv(close=close_series, volume=volume_series)
        return result.values if result is not None and hasattr(result, "values") else np.asarray(result) if result is not None else np.array([])

    @staticmethod
    @handle_pandas_ta_errors
    def nvi(
        close: Union[np.ndarray, pd.Series], volume: Union[np.ndarray, pd.Series], length: int = 13, initial: int = 1000
    ) -> np.ndarray:
        """Negative Volume Index"""
        c = pd.Series(close) if isinstance(close, np.ndarray) else close
        v = pd.Series(volume) if isinstance(volume, np.ndarray) else volume
        df = ta.nvi(close=c, volume=v, length=length, initial=initial)
        return np.asarray(df) if df is not None else np.array([])

    @staticmethod
    @handle_pandas_ta_errors
    def pvi(
        close: Union[np.ndarray, pd.Series], volume: Union[np.ndarray, pd.Series], length: int = 13, initial: int = 1000
    ) -> np.ndarray:
        """Positive Volume Index"""
        c = pd.Series(close) if isinstance(close, np.ndarray) else close
        v = pd.Series(volume) if isinstance(volume, np.ndarray) else volume
        df = ta.pvi(close=c, volume=v, length=length, initial=initial)
        return np.asarray(df) if df is not None else np.array([])

    @staticmethod
    @handle_pandas_ta_errors
    def vwap(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        volume: Union[np.ndarray, pd.Series],
        anchor: str | None = None,
    ) -> np.ndarray:
        """Volume Weighted Average Price"""
        h = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        c = pd.Series(close) if isinstance(close, np.ndarray) else close
        v = pd.Series(volume) if isinstance(volume, np.ndarray) else volume

        # pandas-taのVWAP関数は時系列インデックスが必要なため、独自実装を使用
        try:
            # pandas-taのVWAPを試行（時系列インデックスがある場合）
            if hasattr(h.index, "to_period"):
                df = ta.vwap(high=h, low=low_series, close=c, volume=v, anchor=anchor)
                return np.asarray(df) if df is not None else np.array([])
        except Exception:
            pass

        # フォールバック: 独自VWAP実装
        # 典型価格 = (H + L + C) / 3
        typical_price = (h + low_series + c) / 3

        # VWAP = Σ(典型価格 × 出来高) / Σ(出来高)
        # 累積で計算
        cumulative_pv = (typical_price * v).cumsum()
        cumulative_volume = v.cumsum()

        # ゼロ除算を避ける
        vwap = np.where(
            cumulative_volume != 0, cumulative_pv / cumulative_volume, typical_price
        )

        return np.asarray(vwap)

    @staticmethod
    @handle_pandas_ta_errors
    def eom(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        volume: Union[np.ndarray, pd.Series],
        length: int = 14,
        divisor: int = 100000000,
        drift: int = 1,
    ) -> np.ndarray:
        """Ease of Movement"""
        h = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        c = pd.Series(close) if isinstance(close, np.ndarray) else close
        v = pd.Series(volume) if isinstance(volume, np.ndarray) else volume
        df = ta.eom(high=h, low=low_series, close=c, volume=v, length=length, divisor=divisor, drift=drift)
        return np.asarray(df) if df is not None else np.array([])

    @staticmethod
    @handle_pandas_ta_errors
    def kvo(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        volume: Union[np.ndarray, pd.Series],
        fast: int = 34,
        slow: int = 55,
    ) -> np.ndarray:
        """Klinger Volume Oscillator"""
        h = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        c = pd.Series(close) if isinstance(close, np.ndarray) else close
        v = pd.Series(volume) if isinstance(volume, np.ndarray) else volume
        df = ta.kvo(high=h, low=low_series, close=c, volume=v, fast=fast, slow=slow)
        return np.asarray(df) if df is not None else np.array([])

    @staticmethod
    @handle_pandas_ta_errors
    def pvt(
        close: Union[np.ndarray, pd.Series], volume: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Price Volume Trend"""
        c = pd.Series(close) if isinstance(close, np.ndarray) else close
        v = pd.Series(volume) if isinstance(volume, np.ndarray) else volume
        df = ta.pvt(close=c, volume=v)
        return np.asarray(df) if df is not None else np.array([])

    @staticmethod
    @handle_pandas_ta_errors
    def cmf(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        volume: Union[np.ndarray, pd.Series],
        length: int = 20,
    ) -> np.ndarray:
        """Chaikin Money Flow"""
        h = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        c = pd.Series(close) if isinstance(close, np.ndarray) else close
        v = pd.Series(volume) if isinstance(volume, np.ndarray) else volume
        df = ta.cmf(high=h, low=low_series, close=c, volume=v, length=length)
        return np.asarray(df) if df is not None else np.array([])

    @staticmethod
    @handle_pandas_ta_errors
    def aobv(
        close: Union[np.ndarray, pd.Series],
        volume: Union[np.ndarray, pd.Series],
        fast: int = 5,
        slow: int = 10,
        max_lookback: int = 2,
        min_lookback: int = 2,
        mamode: str = "ema",
    ) -> np.ndarray:
        """Archer On-Balance Volume"""
        c = pd.Series(close) if isinstance(close, np.ndarray) else close
        v = pd.Series(volume) if isinstance(volume, np.ndarray) else volume
        df = ta.aobv(close=c, volume=v, fast=fast, slow=slow, max_lookback=max_lookback, min_lookback=min_lookback, mamode=mamode)
        return np.asarray(df) if df is not None else np.array([])

    @staticmethod
    @handle_pandas_ta_errors
    def efi(
        close: Union[np.ndarray, pd.Series], volume: Union[np.ndarray, pd.Series], length: int = 13, mamode: str = "ema", drift: int = 1
    ) -> np.ndarray:
        """Elder's Force Index"""
        c = pd.Series(close) if isinstance(close, np.ndarray) else close
        v = pd.Series(volume) if isinstance(volume, np.ndarray) else volume
        df = ta.efi(close=c, volume=v, length=length, mamode=mamode, drift=drift)
        return np.asarray(df) if df is not None else np.array([])

    @staticmethod
    @handle_pandas_ta_errors
    def pvol(
        close: Union[np.ndarray, pd.Series], volume: Union[np.ndarray, pd.Series], signed: bool = True
    ) -> np.ndarray:
        """Price-Volume"""
        c = pd.Series(close) if isinstance(close, np.ndarray) else close
        v = pd.Series(volume) if isinstance(volume, np.ndarray) else volume
        df = ta.pvol(close=c, volume=v, signed=signed)
        return np.asarray(df) if df is not None else np.array([])

    @staticmethod
    @handle_pandas_ta_errors
    def pvr(
        close: Union[np.ndarray, pd.Series], volume: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Price Volume Rank"""
        c = pd.Series(close) if isinstance(close, np.ndarray) else close
        v = pd.Series(volume) if isinstance(volume, np.ndarray) else volume
        df = ta.pvr(close=c, volume=v)
        return np.asarray(df) if df is not None else np.array([])

    @staticmethod
    @handle_pandas_ta_errors
    def vp(
        close: Union[np.ndarray, pd.Series], volume: Union[np.ndarray, pd.Series], width: int = 10
    ) -> tuple[np.ndarray, ...]:
        """Volume Price Confirmation (Volume Profile)"""
        c = pd.Series(close) if isinstance(close, np.ndarray) else close
        v = pd.Series(volume) if isinstance(volume, np.ndarray) else volume
        df = ta.vp(close=c, volume=v, width=width)

        if df is None:
            # 空の結果を返す
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        # DataFrameであることを確認し、各列を個別のnumpy arrayとして返す
        if isinstance(df, pd.DataFrame) and not df.empty:
            return (
                df.iloc[:, 0].values,  # low_0 (価格範囲下限)
                df.iloc[:, 1].values,  # mean_0 (価格範囲平均)
                df.iloc[:, 2].values,  # high_0 (価格範囲上限)
                df.iloc[:, 3].values,  # pos_1 (陽線出来高)
                df.iloc[:, 4].values,  # neg_1 (陰線出来高)
                df.iloc[:, 5].values,  # total_1 (総出来高)
            )
        else:
            # 空の結果を返す
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

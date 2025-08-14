"""
出来高系テクニカル指標（pandas-ta移行版）

このモジュールはpandas-taライブラリを使用し、
backtesting.pyとの完全な互換性を提供します。
numpy配列ベースのインターフェースを維持しています。
"""

from typing import Union

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import handle_pandas_ta_errors


class VolumeIndicators:
    """
    出来高系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
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
        return result.values

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
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def obv(
        close: Union[np.ndarray, pd.Series], volume: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """オンバランスボリューム"""
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close
        volume_series = pd.Series(volume) if isinstance(volume, np.ndarray) else volume

        result = ta.obv(close=close_series, volume=volume_series)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def nvi(
        close: Union[np.ndarray, pd.Series], volume: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Negative Volume Index"""
        c = pd.Series(close) if isinstance(close, np.ndarray) else close
        v = pd.Series(volume) if isinstance(volume, np.ndarray) else volume
        df = ta.nvi(close=c, volume=v)
        return df.values if hasattr(df, "values") else np.asarray(df)

    @staticmethod
    @handle_pandas_ta_errors
    def pvi(
        close: Union[np.ndarray, pd.Series], volume: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Positive Volume Index"""
        c = pd.Series(close) if isinstance(close, np.ndarray) else close
        v = pd.Series(volume) if isinstance(volume, np.ndarray) else volume
        df = ta.pvi(close=c, volume=v)
        return df.values if hasattr(df, "values") else np.asarray(df)

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
        df = ta.vwap(high=h, low=low_series, close=c, volume=v, anchor=anchor)
        return df.values if hasattr(df, "values") else np.asarray(df)

    @staticmethod
    @handle_pandas_ta_errors
    def eom(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        volume: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """Ease of Movement"""
        h = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        c = pd.Series(close) if isinstance(close, np.ndarray) else close
        v = pd.Series(volume) if isinstance(volume, np.ndarray) else volume
        df = ta.eom(high=h, low=low_series, close=c, volume=v, length=length)
        return df.values if hasattr(df, "values") else np.asarray(df)

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
        return df.values if hasattr(df, "values") else np.asarray(df)

    @staticmethod
    @handle_pandas_ta_errors
    def pvt(
        close: Union[np.ndarray, pd.Series], volume: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Price Volume Trend"""
        c = pd.Series(close) if isinstance(close, np.ndarray) else close
        v = pd.Series(volume) if isinstance(volume, np.ndarray) else volume
        df = ta.pvt(close=c, volume=v)
        return df.values if hasattr(df, "values") else np.asarray(df)

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
        return df.values if hasattr(df, "values") else np.asarray(df)

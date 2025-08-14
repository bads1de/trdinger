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

from ..utils import (
    handle_pandas_ta_errors,
    ensure_series_minimal_conversion,
    validate_series_data,
)


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
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)
        volume_series = ensure_series_minimal_conversion(volume)

        validate_series_data(high_series, 1)
        validate_series_data(low_series, 1)
        validate_series_data(close_series, 1)
        validate_series_data(volume_series, 1)

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
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)
        volume_series = ensure_series_minimal_conversion(volume)

        validate_series_data(high_series, slow)
        validate_series_data(low_series, slow)
        validate_series_data(close_series, slow)
        validate_series_data(volume_series, slow)

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
        close_series = ensure_series_minimal_conversion(close)
        volume_series = ensure_series_minimal_conversion(volume)

        validate_series_data(close_series, 1)
        validate_series_data(volume_series, 1)

        result = ta.obv(close=close_series, volume=volume_series)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def nvi(
        close: Union[np.ndarray, pd.Series], volume: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Negative Volume Index"""
        c = ensure_series_minimal_conversion(close)
        v = ensure_series_minimal_conversion(volume)
        validate_series_data(c, 2)
        validate_series_data(v, 2)
        df = ta.nvi(close=c, volume=v)
        return df.values if hasattr(df, "values") else np.asarray(df)

    @staticmethod
    @handle_pandas_ta_errors
    def pvi(
        close: Union[np.ndarray, pd.Series], volume: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Positive Volume Index"""
        c = ensure_series_minimal_conversion(close)
        v = ensure_series_minimal_conversion(volume)
        validate_series_data(c, 2)
        validate_series_data(v, 2)
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
        h = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        c = ensure_series_minimal_conversion(close)
        v = ensure_series_minimal_conversion(volume)
        validate_series_data(c, 2)
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
        h = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        c = ensure_series_minimal_conversion(close)
        v = ensure_series_minimal_conversion(volume)
        validate_series_data(c, length)
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
        h = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        c = ensure_series_minimal_conversion(close)
        v = ensure_series_minimal_conversion(volume)
        validate_series_data(c, slow)
        df = ta.kvo(high=h, low=low_series, close=c, volume=v, fast=fast, slow=slow)
        return df.values if hasattr(df, "values") else np.asarray(df)

    @staticmethod
    @handle_pandas_ta_errors
    def pvt(
        close: Union[np.ndarray, pd.Series], volume: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Price Volume Trend"""
        c = ensure_series_minimal_conversion(close)
        v = ensure_series_minimal_conversion(volume)
        validate_series_data(c, 2)
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
        h = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        c = ensure_series_minimal_conversion(close)
        v = ensure_series_minimal_conversion(volume)
        validate_series_data(c, length)
        df = ta.cmf(high=h, low=low_series, close=c, volume=v, length=length)
        return df.values if hasattr(df, "values") else np.asarray(df)

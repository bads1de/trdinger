"""
ボラティリティ系テクニカル指標（pandas-ta移行版）

このモジュールはpandas-taライブラリを使用し、
backtesting.pyとの完全な互換性を提供します。
numpy配列ベースのインターフェースを維持しています。
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import (
    handle_pandas_ta_errors,
    ensure_series_minimal_conversion,
    validate_series_data,
)


class VolatilityIndicators:
    """
    ボラティリティ系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    @handle_pandas_ta_errors
    def atr(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """平均真の値幅"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)
        validate_series_data(close_series, length)

        result = ta.atr(
            high=high_series, low=low_series, close=close_series, length=length
        )
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def natr(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """正規化平均実効値幅"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)
        validate_series_data(close_series, length)

        result = ta.natr(
            high=high_series, low=low_series, close=close_series, length=length
        )
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def trange(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """真の値幅"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        validate_series_data(high_series, 1)
        validate_series_data(low_series, 1)
        validate_series_data(close_series, 1)

        result = ta.true_range(high=high_series, low=low_series, close=close_series)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def bbands(
        data: Union[np.ndarray, pd.Series], length: int = 20, std: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ボリンジャーバンド"""
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, length)
        result = ta.bbands(series, length=length, std=std)

        # 列名を動的に取得（pandas-taのバージョンによって異なる可能性がある）
        columns = result.columns.tolist()

        # 上位、中位、下位バンドを特定
        upper_col = [col for col in columns if "BBU" in col][0]
        middle_col = [col for col in columns if "BBM" in col][0]
        lower_col = [col for col in columns if "BBL" in col][0]

        return (
            result[upper_col].values,
            result[middle_col].values,
            result[lower_col].values,
        )

    @staticmethod
    @handle_pandas_ta_errors
    def keltner(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 20,
        scalar: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keltner Channels: returns (upper, middle, lower)"""
        h = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        c = ensure_series_minimal_conversion(close)
        validate_series_data(c, length)
        df = ta.kc(high=h, low=low_series, close=c, length=length, scalar=scalar)
        cols = list(df.columns)
        upper = df[next((c for c in cols if "KCu" in c), cols[0])].values
        middle = df[
            next(
                (c for c in cols if "KCe" in c or "KCM" in c or "mid" in c.lower()),
                cols[1 if len(cols) > 1 else 0],
            )
        ].values
        lower = df[next((c for c in cols if "KCl" in c), cols[-1])].values
        return upper, middle, lower

    @staticmethod
    @handle_pandas_ta_errors
    def donchian(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        length: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Donchian Channels: returns (upper, middle, lower)"""
        h = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        validate_series_data(h, length)
        validate_series_data(low_series, length)
        df = ta.donchian(high=h, low=low_series, length=length)
        cols = list(df.columns)
        upper = df[
            next((c for c in cols if "DCHU" in c or "upper" in c.lower()), cols[0])
        ].values
        middle = df[
            next(
                (c for c in cols if "DCHM" in c or "mid" in c.lower()),
                cols[1 if len(cols) > 1 else 0],
            )
        ].values
        lower = df[
            next((c for c in cols if "DCHL" in c or "lower" in c.lower()), cols[-1])
        ].values
        return upper, middle, lower

    @staticmethod
    @handle_pandas_ta_errors
    def supertrend(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 10,
        multiplier: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Supertrend: returns (supertrend, direction)"""
        h = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        c = ensure_series_minimal_conversion(close)
        validate_series_data(c, length)
        df = ta.supertrend(
            high=h, low=low_series, close=c, length=length, multiplier=multiplier
        )
        cols = list(df.columns)
        st_col = next(
            (c for c in cols if "SUPERT_" in c.upper() or "supertrend" in c.lower()),
            cols[0],
        )
        dir_col = next(
            (c for c in cols if "SUPERTd_" in c.upper() or "direction" in c.lower()),
            None,
        )
        st = df[st_col].values
        if dir_col is None:
            direction = np.where(c.to_numpy() >= st, 1.0, -1.0)
        else:
            direction = df[dir_col].fillna(0).to_numpy()
            if np.all(np.isnan(direction)):
                direction = np.where(c.to_numpy() >= st, 1.0, -1.0)
        return st, direction

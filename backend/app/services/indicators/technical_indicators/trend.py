"""
トレンド系テクニカル指標（pandas-ta移行版）

このモジュールはpandas-taライブラリを使用し、
backtesting.pyとの完全な互換性を提供します。
numpy配列ベースのインターフェースを維持しています。
"""

from typing import cast, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import (
    PandasTAError,
    handle_pandas_ta_errors,
    to_pandas_series,
    validate_series_data,
    validate_indicator_parameters,
)


class TrendIndicators:
    """
    トレンド系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    @handle_pandas_ta_errors
    def sma(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
        """単純移動平均"""
        validate_indicator_parameters(length)
        series = to_pandas_series(data)
        validate_series_data(series, length)
        result = ta.sma(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def ema(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
        """指数移動平均"""
        validate_indicator_parameters(length)
        series = to_pandas_series(data)
        validate_series_data(series, length)
        result = ta.ema(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def tema(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
        """三重指数移動平均"""
        series = to_pandas_series(data)
        validate_series_data(series, length)
        result = ta.tema(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def dema(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
        """二重指数移動平均"""
        series = to_pandas_series(data)
        validate_series_data(series, length)
        result = ta.dema(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def wma(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
        """加重移動平均"""
        series = to_pandas_series(data)
        validate_series_data(series, length)
        result = ta.wma(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def trima(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
        """三角移動平均"""
        series = to_pandas_series(data)
        validate_series_data(series, length)
        result = ta.trima(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def kama(data: Union[np.ndarray, pd.Series], length: int = 30) -> np.ndarray:
        """カウフマン適応移動平均"""
        series = to_pandas_series(data)
        validate_series_data(series, length)
        result = ta.kama(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def t3(
        data: Union[np.ndarray, pd.Series], length: int = 5, a: float = 0.7
    ) -> np.ndarray:
        """T3移動平均"""
        series = to_pandas_series(data)
        validate_series_data(series, length)
        result = ta.t3(series, length=length, a=a)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def sar(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        af: float = 0.02,
        max_af: float = 0.2,
    ) -> np.ndarray:
        """パラボリックSAR"""
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)

        validate_series_data(high_series, 2)
        validate_series_data(low_series, 2)

        result = ta.psar(high=high_series, low=low_series, af0=af, af=af, max_af=max_af)
        return result[f"PSARl_{af}_{max_af}"].fillna(result[f"PSARs_{af}_{max_af}"]).values

    @staticmethod
    @handle_pandas_ta_errors
    def sarext(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        startvalue: float = 0.0,
        offsetonreverse: float = 0.0,
        accelerationinitlong: float = 0.02,
        accelerationlong: float = 0.02,
        accelerationmaxlong: float = 0.2,
        accelerationinitshort: float = 0.02,
        accelerationshort: float = 0.02,
        accelerationmaxshort: float = 0.2,
    ) -> np.ndarray:
        """Extended Parabolic SAR (approximation using pandas-ta psar)"""
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)

        validate_series_data(high_series, 2)
        validate_series_data(low_series, 2)

        # Map extended parameters to pandas-ta psar arguments (approximate)
        result = ta.psar(
            high=high_series,
            low=low_series,
            af0=accelerationinitlong,
            af=accelerationlong,
            max_af=accelerationmaxlong,
        )

        af = accelerationlong
        max_af = accelerationmaxlong
        return result[f"PSARl_{af}_{max_af}"].fillna(result[f"PSARs_{af}_{max_af}"]).values

    @staticmethod
    @handle_pandas_ta_errors
    def ht_trendline(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Hilbert Transform - Instantaneous Trendline"""
        series = to_pandas_series(data)
        validate_series_data(series, 2)

        # pandas-ta exposes Hilbert transform utilities; use ht_trendline if available
        if hasattr(ta, "ht_trendline"):
            result = ta.ht_trendline(series)
            return result.values
        else:
            raise PandasTAError("pandas-ta does not provide ht_trendline in this version")

    @staticmethod
    def ma(data: np.ndarray, period: int, matype: int = 0) -> np.ndarray:
        """Moving Average (移動平均 - タイプ指定可能) - pandas-ta版"""
        # matypeに応じて適切な移動平均を選択
        if matype == 0:  # SMA
            return TrendIndicators.sma(data, period)
        elif matype == 1:  # EMA
            return TrendIndicators.ema(data, period)
        elif matype == 2:  # WMA
            return TrendIndicators.wma(data, period)
        elif matype == 3:  # DEMA
            return TrendIndicators.dema(data, period)
        elif matype == 4:  # TEMA
            return TrendIndicators.tema(data, period)
        elif matype == 5:  # TRIMA
            return TrendIndicators.trima(data, period)
        elif matype == 6:  # KAMA
            return TrendIndicators.kama(data, period)
        elif matype == 8:  # T3
            return TrendIndicators.t3(data, period)
        else:
            # デフォルトはSMA
            return TrendIndicators.sma(data, period)

    @staticmethod
    def mavp(
        data: np.ndarray,
        periods: np.ndarray,
        minperiod: int = 2,
        maxperiod: int = 30,
        matype: int = 0,
    ) -> np.ndarray:
        """Moving Average with Variable Period (可変期間移動平均)"""
        data = ensure_numpy_array(data)
        periods = ensure_numpy_array(periods)
        if len(data) != len(periods):
            raise PandasTAError(
                f"データと期間の長さが一致しません。Data: {len(data)}, Periods: {len(periods)}"
            )
        validate_input(data, minperiod)
        # MAVP has no direct pandas-ta equivalent; raise error to flag manual handling
        raise NotImplementedError(
            "MAVP is not implemented in pandas-ta and requires custom implementation"
        )

    @staticmethod
    @handle_pandas_ta_errors
    def midpoint(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
        """MidPoint over period"""
        series = to_pandas_series(data)
        validate_series_data(series, length)
        result = ta.midpoint(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def midprice(
        high: Union[np.ndarray, pd.Series], low: Union[np.ndarray, pd.Series], length: int
    ) -> np.ndarray:
        """Midpoint Price over period"""
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)

        result = ta.midprice(high=high_series, low=low_series, length=length)
        return result.values

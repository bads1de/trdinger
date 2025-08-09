"""
トレンド系テクニカル指標（pandas-ta移行版）

このモジュールはpandas-taライブラリを使用し、
backtesting.pyとの完全な互換性を提供します。
numpy配列ベースのインターフェースを維持しています。
"""

from typing import cast

import numpy as np

from ..utils import (
    PandasTAError,
    ensure_numpy_array,
    format_indicator_result,
    handle_talib_errors,
    validate_input,
    validate_multi_input,
)
from ..pandas_ta_utils import (
    sma as pandas_ta_sma,
    ema as pandas_ta_ema,
    tema as pandas_ta_tema,
    dema as pandas_ta_dema,
    wma as pandas_ta_wma,
    trima as pandas_ta_trima,
    kama as pandas_ta_kama,
    t3 as pandas_ta_t3,
    sar as pandas_ta_sar,
    sarext as pandas_ta_sarext,
    ht_trendline as pandas_ta_ht_trendline,
    midpoint as pandas_ta_midpoint,
    midprice as pandas_ta_midprice,
)


class TrendIndicators:
    """
    トレンド系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    def sma(data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average (単純移動平均) - pandas-ta版"""
        return pandas_ta_sma(data, period)

    @staticmethod
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average (指数移動平均) - pandas-ta版"""
        return pandas_ta_ema(data, period)

    @staticmethod
    def tema(data: np.ndarray, period: int) -> np.ndarray:
        """Triple Exponential Moving Average (三重指数移動平均) - pandas-ta版"""
        return pandas_ta_tema(data, period)

    @staticmethod
    def dema(data: np.ndarray, period: int) -> np.ndarray:
        """Double Exponential Moving Average (二重指数移動平均) - pandas-ta版"""
        return pandas_ta_dema(data, period)

    @staticmethod
    def wma(data: np.ndarray, period: int) -> np.ndarray:
        """Weighted Moving Average (加重移動平均) - pandas-ta版"""
        return pandas_ta_wma(data, period)

    @staticmethod
    def trima(data: np.ndarray, period: int) -> np.ndarray:
        """Triangular Moving Average (三角移動平均) - pandas-ta版"""
        return pandas_ta_trima(data, period)

    @staticmethod
    def kama(data: np.ndarray, period: int = 30) -> np.ndarray:
        """Kaufman Adaptive Moving Average (カウフマン適応移動平均) - pandas-ta版"""
        return pandas_ta_kama(data, period)

    @staticmethod
    def t3(data: np.ndarray, period: int = 5, vfactor: float = 0.7) -> np.ndarray:
        """Triple Exponential Moving Average (T3) - pandas-ta版"""
        return pandas_ta_t3(data, period, vfactor)

    @staticmethod
    def sar(
        high: np.ndarray,
        low: np.ndarray,
        acceleration: float = 0.02,
        maximum: float = 0.2,
    ) -> np.ndarray:
        """Parabolic SAR (パラボリックSAR) - pandas-ta版"""
        return pandas_ta_sar(high, low, acceleration, maximum)

    @staticmethod
    @handle_talib_errors
    def sarext(
        high: np.ndarray,
        low: np.ndarray,
        startvalue: float = 0.0,
        offsetonreverse: float = 0.0,
        accelerationinitlong: float = 0.02,
        accelerationlong: float = 0.02,
        accelerationmaxlong: float = 0.2,
        accelerationinitshort: float = 0.02,
        accelerationshort: float = 0.02,
        accelerationmaxshort: float = 0.2,
    ) -> np.ndarray:
        """Parabolic SAR - Extended (拡張パラボリックSAR)"""
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        validate_multi_input(high, low, high, 2)
        params = {
            "startvalue": startvalue,
            "offsetonreverse": offsetonreverse,
            "accelerationinitlong": accelerationinitlong,
            "accelerationlong": accelerationlong,
            "accelerationmaxlong": accelerationmaxlong,
            "accelerationinitshort": accelerationinitshort,
            "accelerationshort": accelerationshort,
            "accelerationmaxshort": accelerationmaxshort,
        }
        result = pandas_ta_sarext(high, low, **params)
        return cast(np.ndarray, format_indicator_result(result, "SAREXT"))

    @staticmethod
    @handle_talib_errors
    def ht_trendline(data: np.ndarray) -> np.ndarray:
        """Hilbert Transform - Instantaneous Trendline"""
        data = ensure_numpy_array(data)
        validate_input(data, 2)
        result = pandas_ta_ht_trendline(data)
        return cast(np.ndarray, format_indicator_result(result, "HT_TRENDLINE"))

    @staticmethod
    def ma(data: np.ndarray, period: int, matype: int = 0) -> np.ndarray:
        """Moving Average (移動平均 - タイプ指定可能) - pandas-ta版"""
        # matypeに応じて適切な移動平均を選択
        if matype == 0:  # SMA
            return pandas_ta_sma(data, period)
        elif matype == 1:  # EMA
            return pandas_ta_ema(data, period)
        elif matype == 2:  # WMA
            return pandas_ta_wma(data, period)
        elif matype == 3:  # DEMA
            return pandas_ta_dema(data, period)
        elif matype == 4:  # TEMA
            return pandas_ta_tema(data, period)
        elif matype == 5:  # TRIMA
            return pandas_ta_trima(data, period)
        elif matype == 6:  # KAMA
            return pandas_ta_kama(data, period)
        elif matype == 8:  # T3
            return pandas_ta_t3(data, period)
        else:
            # デフォルトはSMA
            return pandas_ta_sma(data, period)

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
    @handle_talib_errors
    def midpoint(data: np.ndarray, period: int) -> np.ndarray:
        """MidPoint over period (期間中点)"""
        data = ensure_numpy_array(data)
        validate_input(data, period)
        result = pandas_ta_midpoint(data, period)
        return cast(np.ndarray, format_indicator_result(result, "MIDPOINT"))

    @staticmethod
    @handle_talib_errors
    def midprice(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
        """Midpoint Price over period (期間中値価格)"""
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        validate_multi_input(high, low, high, period)
        result = pandas_ta_midprice(high, low, period)
        return cast(np.ndarray, format_indicator_result(result, "MIDPRICE"))

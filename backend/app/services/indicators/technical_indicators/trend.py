"""
トレンド系テクニカル指標

登録してあるテクニカルの一覧:
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- WMA (Weighted Moving Average)
- DEMA (Double Exponential Moving Average)
- TEMA (Triple Exponential Moving Average)
- T3 (Tillson's T3 Moving Average)
- KAMA (Kaufman's Adaptive Moving Average)
- HMA (Hull Moving Average)
- VWMA (Volume Weighted Moving Average)
- ALMA (Arnaud Legoux Moving Average)
- TRIMA (Triangular Moving Average)
- ZLMA (Zero Lag Moving Average)
- SAR (Parabolic SAR)
- AMAT (Archer Moving Averages Trends)
- RMA (Wilde's Moving Average)
- DPO (Detrended Price Oscillator)
- VORTEX (Vortex Indicator)
"""

import warnings
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import handle_pandas_ta_errors


logger = logging.getLogger(__name__)


# PandasのSeries位置アクセス警告を抑制 (pandas-taとの互換性のため)
warnings.filterwarnings(
    "ignore",
    message="Series.__getitem__ treating keys as positions is deprecated",
    category=FutureWarning,
)


class TrendIndicators:
    """
    トレンド系指標クラス

    移動平均線、Parabolic SARなどのトレンド系テクニカル指標を提供。
    トレンドの方向性と強さの分析に使用します。
    """

    @staticmethod
    @handle_pandas_ta_errors
    def sma(data: pd.Series, length: int) -> pd.Series:
        # パラメータ型チェック
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        # lengthパラメータ妥当性チェック
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        # データ長チェック（オプション）
        if len(data) == 0:
            return pd.Series(np.full(0, np.nan), index=data.index)

        return ta.sma(data, length=length)

    @staticmethod
    @handle_pandas_ta_errors
    def ema(data: pd.Series, length: int) -> pd.Series:
        """指数移動平均"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError(f"period must be positive: {length}")
        if len(data) == 0:
            return pd.Series(np.full(0, np.nan), index=data.index)
        return ta.ema(data, window=length, adjust=False, sma=True)

    @staticmethod
    @handle_pandas_ta_errors
    def wma(
        data: pd.Series = None,
        length: int = 14,
        close: pd.Series = None,
    ) -> pd.Series:
        """加重移動平均"""
        # dataが提供されない場合はcloseを使用
        if data is None and close is not None:
            data = close
        elif data is None:
            raise ValueError("Either 'data' or 'close' must be provided")

        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        return ta.wma(data, window=length)

    @staticmethod
    @handle_pandas_ta_errors
    def trima(data: pd.Series, length: int = 10, talib: bool | None = None) -> pd.Series:
        """三角移動平均"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        result = ta.trima(data, length=length, talib=talib)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def zlma(
        data: pd.Series,
        length: int = 10,
        mamode: str = "ema",
        offset: int = 0,
    ) -> pd.Series:
        """Zero Lag移動平均"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        result = ta.zlma(data, length=length, mamode=mamode, offset=offset)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def alma(
        data: pd.Series,
        length: int = 10,
        sigma: float = 6.0,
        distribution_offset: float = 0.85,
        offset: int = 0,
    ) -> pd.Series:
        """Arnaud Legoux Moving Average"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive: {sigma}")
        if not 0.0 <= distribution_offset <= 1.0:
            raise ValueError(
                f"distribution_offset must be between 0.0 and 1.0: {distribution_offset}"
            )

        result = ta.alma(
            data,
            length=length,
            sigma=sigma,
            distribution_offset=distribution_offset,
            offset=offset,
        )
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def dema(data: pd.Series, length: int) -> pd.Series:
        """二重指数移動平均"""
        # logger.debug(f"DEMA calculation started: data_length={len(data)}, length={length}")
        if not isinstance(data, pd.Series):
            logger.error("DEMA: Invalid data type - must be pandas Series")
            raise TypeError("data must be pandas Series")
        if length <= 0:
            logger.error(f"DEMA: Invalid length parameter: {length}")
            raise ValueError(f"length must be positive: {length}")
        if len(data) == 0:
            # logger.debug("DEMA: Empty data series, returning empty NaN series")
            return pd.Series(np.full(0, np.nan), index=data.index)

        # DEMAは2つのEMAを使用するため、最低length * 2のデータが必要
        min_required_length = length * 2
        if len(data) < min_required_length:
            # logger.debug(f"DEMA: Data length {len(data)} < minimum required {min_required_length}, returning NaN series")
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # logger.debug("DEMA: Calling pandas-ta.dema()")
        result = ta.dema(data, window=length)
        # logger.debug(f"DEMA: Calculation completed, result_length={len(result) if result is not None else 'None'}")
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def tema(data: pd.Series, length: int) -> pd.Series:
        """三重指数移動平均"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")
        if len(data) == 0:
            return pd.Series(np.full(0, np.nan), index=data.index)

        # TEMA requires sufficient data length (approximately length * 3 for 3-stage EMA)
        min_data_length = length * 3
        if len(data) < min_data_length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        return ta.tema(data, window=length)

    @staticmethod
    @handle_pandas_ta_errors
    def t3(data: pd.Series, length: int, a: float = 0.7) -> pd.Series:
        """T3移動平均"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError(f"period must be positive: {length}")
        if not (0.0 <= a <= 1.0):
            raise ValueError(f"a must be between 0.0 and 1.0: {a}")
        if len(data) == 0:
            return pd.Series(np.full(0, np.nan), index=data.index)

        # T3 requires sufficient data length (approximately length * 6 for 6-stage EMA)
        min_data_length = length * 6
        if len(data) < min_data_length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # Use pandas-ta directly
        result = ta.t3(data, window=length, a=a)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def kama(data: pd.Series, length: int = 30) -> pd.Series:
        """カウフマン適応移動平均"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")
        if len(data) == 0:
            return pd.Series(np.full(0, np.nan), index=data.index)
        return ta.kama(data, window=length)

    @staticmethod
    @handle_pandas_ta_errors
    def hma(data: pd.Series, length: int = 20) -> pd.Series:
        """Hull移動平均"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")
        if len(data) == 0:
            return pd.Series(np.full(0, np.nan), index=data.index)

        result = ta.hma(data, length=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def vwma(
        close: pd.Series,
        volume: pd.Series,
        length: int = 20,
    ) -> pd.Series:
        """出来高加重移動平均"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")
        if len(close) != len(volume):
            raise ValueError("close and volume series must share the same length")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")
        if len(close) == 0:
            return pd.Series(np.full(0, np.nan), index=close.index)

        result = ta.vwma(close=close, volume=volume, length=length)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def linreg(
        data: pd.Series,
        length: int = 14,
        scalar: float = 1.0,
        intercept: bool = False,
    ) -> pd.Series:
        """線形回帰 (pandas-ta)"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError("length must be positive")
        if scalar == 0:
            raise ValueError("scalar must be non-zero")

        if len(data) < length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        values = [np.nan] * (length - 1)

        for i in range(length - 1, len(data)):
            window = data[i-length+1:i+1]
            x = np.arange(length)
            coeffs = np.polyfit(x, window, 1)  # [slope, intercept]
            if intercept:
                value = coeffs[1]  # y切片
            else:
                # 中心点の値を計算
                mid_x = (length - 1) / 2
                value = coeffs[0] * mid_x + coeffs[1]
            values.append(value * scalar)

        return pd.Series(values, index=data.index)

    @staticmethod
    @handle_pandas_ta_errors
    def linregslope(data: pd.Series, length: int = 14, scalar: float = 1.0) -> pd.Series:
        """線形回帰スロープ"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError("length must be positive")
        if scalar == 0:
            raise ValueError("scalar must be non-zero")

        if len(data) < length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        slopes = [np.nan] * (length - 1)

        for i in range(length - 1, len(data)):
            window = data[i-length+1:i+1]
            x = np.arange(length)
            slope = np.polyfit(x, window, 1)[0]  # 1次多項式の係数（スロープ）
            slopes.append(slope * scalar)  # scalarを適用

        return pd.Series(slopes, index=data.index)

    @staticmethod
    @handle_pandas_ta_errors
    def sar(
        high: pd.Series,
        low: pd.Series,
        af: float = 0.02,
        max_af: float = 0.2,
    ) -> pd.Series:
        """パラボリックSAR"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        result = ta.psar(high=high, low=low, af0=af, af=af, max_af=max_af)
        # PSARl と PSARs を結合
        psar_long = result[f"PSARl_{af}_{max_af}"]
        psar_short = result[f"PSARs_{af}_{max_af}"]
        return psar_long.fillna(psar_short)

    @staticmethod
    @handle_pandas_ta_errors
    def amat(
        data: pd.Series,
        fast: int = 3,
        slow: int = 30,
        signal: int = 10
    ) -> pd.Series:
        """Archer Moving Averages Trends"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        # AMAT特有のデータ検証
        min_length = max(fast, slow, signal) + 10
        if len(data) < min_length:
            raise ValueError(f"Insufficient data for AMAT calculation. Need at least {min_length} points, got {len(data)}")

        result = ta.amat(data, fast=fast, slow=slow, signal=signal)
        if result is None or (hasattr(result, "empty") and result.empty):
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        # AMAT returns DataFrame, get the main series
        if hasattr(result, 'iloc'):
            return result.iloc[:, 0] if len(result.shape) > 1 else result
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def rma(data: pd.Series, length: int = 10) -> pd.Series:
        """Wilde's Moving Average"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError("length must be positive")

        result = ta.rma(data, length=length)
        if result is None or (hasattr(result, "empty") and result.empty):
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def dpo(
        data: pd.Series,
        length: int = 20,
        centered: bool = True,
        offset: int = 0,
    ) -> pd.Series:
        """Detrended Price Oscillator"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        result = ta.dpo(
            close=data,
            length=length,
            centered=centered,
            offset=offset,
        )
        if result is None or (hasattr(result, "empty") and result.empty):
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def vortex(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
        drift: int = 1,
        offset: int = 0,
    ) -> Tuple[pd.Series, pd.Series]:
        """Vortex Indicator"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")
        if drift <= 0:
            raise ValueError(f"drift must be positive: {drift}")

        result = ta.vortex(
            high=high,
            low=low,
            close=close,
            length=length,
            drift=drift,
            offset=offset,
        )

        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series

        return result.iloc[:, 0], result.iloc[:, 1]


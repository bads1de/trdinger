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
- SAR (Parabolic SAR)
"""

import warnings
import logging

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


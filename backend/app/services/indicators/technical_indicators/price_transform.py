"""
価格変換系テクニカル指標

登録してあるテクニカルの一覧:
- AvgPrice (Average Price)
- MedPrice (Median Price)
- TypPrice (Typical Price)
- WclPrice (Weighted Close Price)
- HA_Close (Heikin Ashi Close)
- HA_OHLC (Heikin Ashi OHLC)
"""

from typing import Union

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import handle_pandas_ta_errors


class PriceTransformIndicators:
    """
    価格変換系指標クラス
    """

    @staticmethod
    @handle_pandas_ta_errors
    def avgprice(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """平均価格"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        # pandas-taの引数名はopen_の場合がある
        try:
            result = ta.ohlc4(
                open=open_series, high=high_series, low=low_series, close=close_series
            )
        except TypeError:
            result = ta.ohlc4(
                open_=open_series, high=high_series, low=low_series, close=close_series
            )
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def medprice(
        high: Union[np.ndarray, pd.Series], low: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """中央値価格"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low

        result = ta.hl2(high=high_series, low=low_series)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def typprice(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """典型価格"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.hlc3(high=high_series, low=low_series, close=close_series)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def wclprice(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """加重終値価格"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.wcp(high=high_series, low=low_series, close=close_series)
        return result.values

    @staticmethod
    def ha_close(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Heikin Ashi Close（平均足終値）
        直接計算によりpandasの警告を回避。
        """
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        # Heikin-Ashiの計算式:
        # HA_Close = (Open + High + Low + Close) / 4
        ha_close = (open_series + high_series + low_series + close_series) / 4.0

        return ha_close.values

    @staticmethod
    def ha_ohlc(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Heikin Ashi の OHLC をタプルで返す (open, high, low, close)
        直接計算によりpandasの警告を回避。
        """
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        # Heikin-Ashiの計算式:
        # HA_Close = (Open + High + Low + Close) / 4
        ha_close = (open_series + high_series + low_series + close_series) / 4.0

        # HA_Open = (前のHA_Open + 前のHA_Close) / 2
        # 最初の値は元のOpenを使用
        ha_open = pd.Series(index=open_series.index, dtype=float)
        ha_open.iloc[0] = open_series.iloc[0]  # 最初の値は元のOpen

        # 2番目以降のHA_Openを計算
        for i in range(1, len(ha_open)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2.0

        # HA_High = max(High, HA_Open, HA_Close)
        ha_high = pd.concat([high_series, ha_open, ha_close], axis=1).max(axis=1)

        # HA_Low = min(Low, HA_Open, HA_Close)
        ha_low = pd.concat([low_series, ha_open, ha_close], axis=1).min(axis=1)

        return ha_open.values, ha_high.values, ha_low.values, ha_close.values

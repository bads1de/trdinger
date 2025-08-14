"""
価格変換系テクニカル指標

このモジュールはpandas-taライブラリを使用し、
backtesting.pyとの完全な互換性を提供します。
numpy配列ベースのインターフェースを維持しています。
"""

from typing import Union

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import handle_pandas_ta_errors


class PriceTransformIndicators:
    """
    価格変換系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、pandas-taの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
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
        open_series = pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
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
    @handle_pandas_ta_errors
    def ha_close(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Heikin Ashi Close（平均足終値）
        pandas-ta の ha を用いてHAの終値のみ返す。
        """
        open_series = pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        try:
            ha_df = ta.ha(
                open=open_series, high=high_series, low=low_series, close=close_series
            )
        except TypeError:
            ha_df = ta.ha(
                open_=open_series, high=high_series, low=low_series, close=close_series
            )

        # 列名はバージョンにより異なり得るため末尾一致で選択
        close_col = [c for c in ha_df.columns if str(c).lower().endswith("close")]
        col = close_col[0] if close_col else ha_df.columns[-1]
        return ha_df[col].values

    @staticmethod
    @handle_pandas_ta_errors
    def ha_ohlc(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Heikin Ashi の OHLC をタプルで返す (open, high, low, close)"""
        open_series = pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        try:
            ha_df = ta.ha(
                open=open_series, high=high_series, low=low_series, close=close_series
            )
        except TypeError:
            ha_df = ta.ha(
                open_=open_series, high=high_series, low=low_series, close=close_series
            )

        # 列並びを open, high, low, close 順に選択
        def pick_col(suffix: str):
            candidates = [c for c in ha_df.columns if str(c).lower().endswith(suffix)]
            return (
                ha_df[candidates[0]].values if candidates else ha_df.iloc[:, 0].values
            )

        ha_open = pick_col("open")
        ha_high = pick_col("high")
        ha_low = pick_col("low")
        ha_close = pick_col("close")
        return ha_open, ha_high, ha_low, ha_close

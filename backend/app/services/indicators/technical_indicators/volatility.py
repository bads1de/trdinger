"""
ボラティリティ系テクニカル指標

登録してあるテクニカルの一覧:
- ATR (Average True Range)
- NATR (Normalized Average True Range)
- True Range
- Bollinger Bands
- Keltner Channels
- Donchian Channels
- Supertrend
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import handle_pandas_ta_errors


class VolatilityIndicators:
    """
    ボラティリティ系指標クラス
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
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

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
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

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
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.true_range(high=high_series, low=low_series, close=close_series)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def bbands(
        data: Union[np.ndarray, pd.Series], length: int = 20, std: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ボリンジャーバンド"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
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
        std_dev: float = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keltner Channels: returns (upper, middle, lower)"""
        h = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        c = pd.Series(close) if isinstance(close, np.ndarray) else close

        # std_dev パラメータは使用しないが、互換性のため受け入れる
        _ = std_dev

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
        h = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
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
        h = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        c = pd.Series(close) if isinstance(close, np.ndarray) else close
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

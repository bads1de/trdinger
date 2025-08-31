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
- Aberration
- Acceleration Bands
- Holt-Winter Channel
- Mass Index
- Price Distance
- Elder's Thermometer
- Ulcer Index
"""

from typing import Tuple, Union, Optional

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
        if result is None:
            return np.full(len(high_series), np.nan)
        assert result is not None  # for type checker
        return result.to_numpy()

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
        if result is None:
            return np.full(len(high_series), np.nan)
        assert result is not None  # for type checker
        return result.to_numpy()

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
        if result is None:
            return np.full(len(high_series), np.nan)
        assert result is not None  # for type checker
        return result.to_numpy()

    @staticmethod
    @handle_pandas_ta_errors
    def bbands(
        data: Union[np.ndarray, pd.Series], length: int = 20, std: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ボリンジャーバンド"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.bbands(series, length=length, std=std)

        if result is None:
            # ta.bbandsがNoneを返す場合のフォールバック
            return (
                np.full(len(series), np.nan),
                np.full(len(series), np.nan),
                np.full(len(series), np.nan),
            )

        assert result is not None  # for type checker
        # 列名を動的に取得（pandas-taのバージョンによって異なる可能性がある）
        columns = result.columns.tolist()

        # 上位、中位、下位バンドを特定
        upper_col = [col for col in columns if "BBU" in col][0]
        middle_col = [col for col in columns if "BBM" in col][0]
        lower_col = [col for col in columns if "BBL" in col][0]

        return (
            result[upper_col].to_numpy(),
            result[middle_col].to_numpy(),
            result[lower_col].to_numpy(),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def keltner(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 20,
        scalar: float = 2.0,
        std_dev: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keltner Channels: returns (upper, middle, lower)"""
        h = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        c = pd.Series(close) if isinstance(close, np.ndarray) else close

        # std_dev パラメータは使用しないが、互換性のため受け入れる
        _ = std_dev

        df = ta.kc(high=h, low=low_series, close=c, length=length, scalar=scalar)
        if df is None:
            # ta.kcがNoneを返す場合のフォールバック
            upper = np.full(len(c), np.nan)
            middle = np.full(len(c), np.nan)
            lower = np.full(len(c), np.nan)
            return upper, middle, lower

        assert df is not None  # for type checker
        cols = list(df.columns)
        upper = df[next((c for c in cols if "KCu" in c), cols[0])].to_numpy()
        middle = df[
            next(
                (c for c in cols if "KCe" in c or "KCM" in c or "mid" in c.lower()),
                cols[1 if len(cols) > 1 else 0],
            )
        ].to_numpy()
        lower = df[next((c for c in cols if "KCl" in c), cols[-1])].to_numpy()
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
        if df is None:
            # ta.donchianがNoneを返す場合のフォールバック
            upper = np.full(len(h), np.nan)
            middle = np.full(len(h), np.nan)
            lower = np.full(len(h), np.nan)
            return upper, middle, lower

        assert df is not None  # for type checker
        cols = list(df.columns)
        upper = df[
            next((c for c in cols if "DCHU" in c or "upper" in c.lower()), cols[0])
        ].to_numpy()
        middle = df[
            next(
                (c for c in cols if "DCHM" in c or "mid" in c.lower()),
                cols[1 if len(cols) > 1 else 0],
            )
        ].to_numpy()
        lower = df[
            next((c for c in cols if "DCHL" in c or "lower" in c.lower()), cols[-1])
        ].to_numpy()
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
        if df is None:
            # ta.supertrendがNoneを返す場合のフォールバック
            st = np.full(len(c), np.nan)
            direction = np.full(len(c), np.nan)
            return st, direction

        assert df is not None  # for type checker
        cols = list(df.columns)
        st_col = next(
            (c for c in cols if "SUPERT_" in c.upper() or "supertrend" in c.lower()),
            cols[0],
        )
        dir_col = next(
            (c for c in cols if "SUPERTd_" in c.upper() or "direction" in c.lower()),
            None,
        )
        st = df[st_col].to_numpy()
        if dir_col is None:
            direction = np.where(c.to_numpy() >= st, 1.0, -1.0)
        else:
            direction = df[dir_col].fillna(0).to_numpy()
            if np.all(np.isnan(direction)):
                direction = np.where(c.to_numpy() >= st, 1.0, -1.0)
        return st, direction

    @staticmethod
    @handle_pandas_ta_errors
    def aberration(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 5,
    ) -> np.ndarray:
        """Aberration"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.aberration(
            high=high_series, low=low_series, close=close_series, length=length
        )
        if result is None:
            return np.full(len(high_series), np.nan)
        return result.to_numpy()

    @staticmethod
    @handle_pandas_ta_errors
    def accbands(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Acceleration Bands: returns (upper, middle, lower)"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.accbands(high=high_series, low=low_series, close=close_series, length=length)
        if result is None:
            # ta.accbandsがNoneを返す場合のフォールバック
            upper = np.full(len(close_series), np.nan)
            middle = np.full(len(close_series), np.nan)
            lower = np.full(len(close_series), np.nan)
            return upper, middle, lower

        assert result is not None  # for type checker
        cols = list(result.columns)
        upper = result[next((c for c in cols if "ACCBU" in c or "upper" in c.lower()), cols[0])].to_numpy()
        middle = result[
            next(
                (c for c in cols if "ACCBM" in c or "mid" in c.lower()),
                cols[1 if len(cols) > 1 else 0],
            )
        ].to_numpy()
        lower = result[next((c for c in cols if "ACCBL" in c or "lower" in c.lower()), cols[-1])].to_numpy()
        return upper, middle, lower

    @staticmethod
    @handle_pandas_ta_errors
    def hwc(
        close: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Holt-Winter Channel: returns (upper, middle, lower)"""
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        # Extract parameters from kwargs (needed for parameter compatibility)
        na = kwargs.get('na', 0.2)
        nb = kwargs.get('nb', 0.1)
        nc = kwargs.get('nc', 3.0)
        nd = kwargs.get('nd', 0.3)
        scalar = kwargs.get('scalar', 2.0)

        result = ta.hwc(close=close_series, na=na, nb=nb, nc=nc, nd=nd, scalar=scalar)
        if result is None:
            # ta.hwcがNoneを返す場合のフォールバック
            upper = np.full(len(close_series), np.nan)
            middle = np.full(len(close_series), np.nan)
            lower = np.full(len(close_series), np.nan)
            return upper, middle, lower

        assert result is not None  # for type checker
        cols = list(result.columns)
        upper = result[next((c for c in cols if "HWU" in c or "upper" in c.lower()), cols[0])].to_numpy()
        middle = result[
            next(
                (c for c in cols if "HWM" in c or "mid" in c.lower()),
                cols[1 if len(cols) > 1 else 0],
            )
        ].to_numpy()
        lower = result[next((c for c in cols if "HWL" in c or "lower" in c.lower()), cols[-1])].to_numpy()
        return upper, middle, lower

    @staticmethod
    @handle_pandas_ta_errors
    def massi(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        length: int = 25,
        fast: int = 9,
        slow: int = 25,
    ) -> np.ndarray:
        """Mass Index"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low

        result = ta.massi(high=high_series, low=low_series, length=length, fast=fast, slow=slow)
        if result is None:
            return np.full(len(high_series), np.nan)
        return result.to_numpy()

    @staticmethod
    @handle_pandas_ta_errors
    def pdist(
        open: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 10,
    ) -> np.ndarray:
        """Price Distance"""
        open_series = pd.Series(open) if isinstance(open, np.ndarray) else open
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.pdist(open_=open_series, high=high_series, low=low_series, close=close_series, length=length)
        if result is None:
            return np.full(len(close_series), np.nan)
        return result.to_numpy()

    @staticmethod
    @handle_pandas_ta_errors
    def thermo(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        length: int = 25,
        long_length: int = 14,
    ) -> np.ndarray:
        """Elder's Thermometer"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low

        result = ta.thermo(high=high_series, low=low_series, length=length, long_length=long_length)
        if result is None:
            return np.full(len(high_series), np.nan)
        return result.to_numpy()

    @staticmethod
    @handle_pandas_ta_errors
    def ui(
        data: Union[np.ndarray, pd.Series], length: int = 14
    ) -> np.ndarray:
        """Ulcer Index"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        result = ta.ui(series, length=length)
        if result is None:
            return np.full(len(series), np.nan)
        return result.to_numpy()

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

from typing import Tuple, Optional

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
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """平均真の値幅"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.atr(
            high=high, low=low, close=close, length=length
        )
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        assert result is not None  # for type checker
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def natr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """正規化平均実効値幅"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.natr(
            high=high, low=low, close=close, length=length
        )
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        assert result is not None  # for type checker
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def trange(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """真の値幅"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.true_range(high=high, low=low, close=close)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        assert result is not None  # for type checker
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def bbands(
        data: pd.Series, length: int = 20, std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ボリンジャーバンド"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        result = ta.bbands(data, length=length, std=std)

        if result is None:
            # ta.bbandsがNoneを返す場合のフォールバック
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return (nan_series, nan_series, nan_series)

        assert result is not None  # for type checker
        # 列名を動的に取得（pandas-taのバージョンによって異なる可能性がある）
        columns = result.columns.tolist()

        # 上位、中位、下位バンドを特定
        upper_col = [col for col in columns if "BBU" in col][0]
        middle_col = [col for col in columns if "BBM" in col][0]
        lower_col = [col for col in columns if "BBL" in col][0]

        return (
            result[upper_col],
            result[middle_col],
            result[lower_col],
        )

    @staticmethod
    @handle_pandas_ta_errors
    def keltner(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 20,
        scalar: float = 2.0,
        std_dev: Optional[float] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels: returns (upper, middle, lower)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # std_dev パラメータは使用しないが、互換性のため受け入れる
        _ = std_dev

        df = ta.kc(high=high, low=low, close=close, length=length, scalar=scalar)
        if df is None:
            # ta.kcがNoneを返す場合のフォールバック
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series

        assert df is not None  # for type checker
        cols = list(df.columns)
        upper = df[next((c for c in cols if "KCu" in c), cols[0])]
        middle = df[
            next(
                (c for c in cols if "KCe" in c or "KCM" in c or "mid" in c.lower()),
                cols[1 if len(cols) > 1 else 0],
            )
        ]
        lower = df[next((c for c in cols if "KCl" in c), cols[-1])]
        return upper, middle, lower

    @staticmethod
    @handle_pandas_ta_errors
    def donchian(
        high: pd.Series,
        low: pd.Series,
        length: int = 20,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels: returns (upper, middle, lower)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        df = ta.donchian(high=high, low=low, length=length)
        if df is None:
            # ta.donchianがNoneを返す場合のフォールバック
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series, nan_series

        assert df is not None  # for type checker
        cols = list(df.columns)
        upper = df[
            next((c for c in cols if "DCHU" in c or "upper" in c.lower()), cols[0])
        ]
        middle = df[
            next(
                (c for c in cols if "DCHM" in c or "mid" in c.lower()),
                cols[1 if len(cols) > 1 else 0],
            )
        ]
        lower = df[
            next((c for c in cols if "DCHL" in c or "lower" in c.lower()), cols[-1])
        ]
        return upper, middle, lower

    @staticmethod
    @handle_pandas_ta_errors
    def supertrend(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 10,
        multiplier: float = 3.0,
    ) -> Tuple[pd.Series, pd.Series]:
        """Supertrend: returns (supertrend, direction)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        df = ta.supertrend(
            high=high, low=low, close=close, length=length, multiplier=multiplier
        )
        if df is None:
            # ta.supertrendがNoneを返す場合のフォールバック
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series

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
        st = df[st_col]
        if dir_col is None:
            direction = pd.Series(np.where(close.to_numpy() >= st.to_numpy(), 1.0, -1.0), index=close.index)
        else:
            direction = df[dir_col].fillna(0)
            if direction.isna().all():
                direction = pd.Series(np.where(close.to_numpy() >= st.to_numpy(), 1.0, -1.0), index=close.index)
        return st, direction

    @staticmethod
    @handle_pandas_ta_errors
    def aberration(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 5,
    ) -> pd.Series:
        """Aberration"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.aberration(
            high=high, low=low, close=close, length=length
        )
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def accbands(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 20,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Acceleration Bands: returns (upper, middle, lower)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.accbands(high=high, low=low, close=close, length=length)
        if result is None:
            # ta.accbandsがNoneを返す場合のフォールバック
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series

        assert result is not None  # for type checker
        cols = list(result.columns)
        upper = result[next((c for c in cols if "ACCBU" in c or "upper" in c.lower()), cols[0])]
        middle = result[
            next(
                (c for c in cols if "ACCBM" in c or "mid" in c.lower()),
                cols[1 if len(cols) > 1 else 0],
            )
        ]
        lower = result[next((c for c in cols if "ACCBL" in c or "lower" in c.lower()), cols[-1])]
        return upper, middle, lower

    @staticmethod
    @handle_pandas_ta_errors
    def hwc(
        close: pd.Series,
        **kwargs
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Holt-Winter Channel: returns (upper, middle, lower)"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # Extract parameters from kwargs (needed for parameter compatibility)
        na = kwargs.get('na', 0.2)
        nb = kwargs.get('nb', 0.1)
        nc = kwargs.get('nc', 3.0)
        nd = kwargs.get('nd', 0.3)
        scalar = kwargs.get('scalar', 2.0)

        result = ta.hwc(close=close, na=na, nb=nb, nc=nc, nd=nd, scalar=scalar)
        if result is None:
            # ta.hwcがNoneを返す場合のフォールバック
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series

        assert result is not None  # for type checker
        cols = list(result.columns)
        upper = result[next((c for c in cols if "HWU" in c or "upper" in c.lower()), cols[0])]
        middle = result[
            next(
                (c for c in cols if "HWM" in c or "mid" in c.lower()),
                cols[1 if len(cols) > 1 else 0],
            )
        ]
        lower = result[next((c for c in cols if "HWL" in c or "lower" in c.lower()), cols[-1])]
        return upper, middle, lower

    @staticmethod
    @handle_pandas_ta_errors
    def massi(
        high: pd.Series,
        low: pd.Series,
        length: int = 25,
        fast: int = 9,
        slow: int = 25,
    ) -> pd.Series:
        """Mass Index"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        result = ta.massi(high=high, low=low, length=length, fast=fast, slow=slow)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def pdist(
        open: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 10,
    ) -> pd.Series:
        """Price Distance"""
        if not isinstance(open, pd.Series):
            raise TypeError("open must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.pdist(open_=open, high=high, low=low, close=close, length=length)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def thermo(
        high: pd.Series,
        low: pd.Series,
        length: int = 25,
        long_length: int = 14,
    ) -> pd.Series:
        """Elder's Thermometer"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        result = ta.thermo(high=high, low=low, length=length, long_length=long_length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def ui(
        data: pd.Series, length: int = 14
    ) -> pd.Series:
        """Ulcer Index"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        result = ta.ui(data, length=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

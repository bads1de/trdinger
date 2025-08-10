"""
追加のモメンタム指標（pandas-taに準拠）
AO, KDJ, RVGI, QQE, SMI, KST, STC など
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import (
    PandasTAError,
    handle_pandas_ta_errors,
    ensure_series_minimal_conversion,
    validate_series_data,
)


class MoreMomentumIndicators:
    @staticmethod
    @handle_pandas_ta_errors
    def ao(
        high: Union[np.ndarray, pd.Series], low: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Awesome Oscillator"""
        high_s = ensure_series_minimal_conversion(high)
        low_s = ensure_series_minimal_conversion(low)
        validate_series_data(high_s, 5)
        validate_series_data(low_s, 5)
        result = ta.ao(high=high_s, low=low_s)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def kdj(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        k: int = 14,
        d: int = 3,
        j_scalar: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """KDJ: pandas-taではstochから計算"""
        high_s = ensure_series_minimal_conversion(high)
        low_s = ensure_series_minimal_conversion(low)
        close_s = ensure_series_minimal_conversion(close)
        validate_series_data(close_s, k + d)
        stoch_df = ta.stoch(high=high_s, low=low_s, close=close_s, k=k, d=d, smooth_k=3)
        # pandas-taが先頭NaN区間で短縮する場合に備え、インデックスを合わせる
        stoch_df = stoch_df.reindex(close_s.index)
        # 列検出
        k_col = next((c for c in stoch_df.columns if "k" in c.lower()), None)
        d_col = next((c for c in stoch_df.columns if "d" in c.lower()), None)
        if k_col is None or d_col is None:
            raise PandasTAError("KDJの元となるstoch列が見つかりません")
        k_vals = stoch_df[k_col].values
        d_vals = stoch_df[d_col].values
        j_vals = j_scalar * k_vals - 2 * d_vals
        return k_vals, d_vals, j_vals

    @staticmethod
    @handle_pandas_ta_errors
    def rvgi(
        open_: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Relative Vigor Index"""
        o = ensure_series_minimal_conversion(open_)
        h = ensure_series_minimal_conversion(high)
        l = ensure_series_minimal_conversion(low)
        c = ensure_series_minimal_conversion(close)
        validate_series_data(c, length + 1)
        df = ta.rvgi(open_=o, high=h, low=l, close=c, length=length)
        r_col = next(
            (c for c in df.columns if c.lower().endswith("rvi")), df.columns[0]
        )
        s_col = next(
            (c for c in df.columns if c.lower().endswith("signal")), df.columns[-1]
        )
        return df[r_col].values, df[s_col].values

    @staticmethod
    @handle_pandas_ta_errors
    def qqe(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
        """Qualitative Quantitative Estimation"""
        s = ensure_series_minimal_conversion(data)
        validate_series_data(s, length + 1)
        df = ta.qqe(s, length=length)
        # 単列
        return df.values if hasattr(df, "values") else np.asarray(df)

    @staticmethod
    @handle_pandas_ta_errors
    def smi(
        data: Union[np.ndarray, pd.Series],
        fast: int = 13,
        slow: int = 25,
        signal: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Momentum Index"""
        s = ensure_series_minimal_conversion(data)
        validate_series_data(s, fast + slow + signal)
        df = ta.smi(s, fast=fast, slow=slow, signal=signal)
        # 2列想定
        cols = list(df.columns)
        return df[cols[0]].values, df[cols[1]].values

    @staticmethod
    @handle_pandas_ta_errors
    def kst(
        data: Union[np.ndarray, pd.Series],
        r1: int = 10,
        r2: int = 15,
        r3: int = 20,
        r4: int = 30,
        n1: int = 10,
        n2: int = 10,
        n3: int = 10,
        n4: int = 15,
        signal: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Know Sure Thing"""
        s = ensure_series_minimal_conversion(data)
        validate_series_data(s, max(r1, r2, r3, r4, signal))
        df = ta.kst(
            s, r1=r1, r2=r2, r3=r3, r4=r4, n1=n1, n2=n2, n3=n3, n4=n4, signal=signal
        )
        k_col = next((c for c in df.columns if c.lower().endswith("kst")), None)
        s_col = next((c for c in df.columns if c.lower().endswith("signal")), None)
        if k_col is None or s_col is None:
            cols = list(df.columns)
            return df[cols[0]].values, df[cols[-1]].values
        return df[k_col].values, df[s_col].values

    @staticmethod
    @handle_pandas_ta_errors
    def stc(
        data: Union[np.ndarray, pd.Series],
        tclength: int = 10,
        fast: int = 23,
        slow: int = 50,
        factor: float = 0.5,
    ) -> np.ndarray:
        """Schaff Trend Cycle"""
        s = ensure_series_minimal_conversion(data)
        validate_series_data(s, slow + tclength)
        df = ta.stc(s, tclength=tclength, fast=fast, slow=slow, factor=factor)
        return df.values if hasattr(df, "values") else np.asarray(df)

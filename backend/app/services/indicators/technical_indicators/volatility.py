"""
ボラティリティ系テクニカル指標 (Volatility Indicators)

pandas-ta の volatility カテゴリに対応。
市場の変動性とリスク評価に使用する指標群。

登録してあるテクニカルの一覧:
- ATR (Average True Range)
- NATR (Normalized ATR)
- Bollinger Bands
- Keltner Channels
- Donchian Channels
- Acceleration Bands
- Ulcer Index
- RVI (Relative Volatility Index)
- True Range
- Yang-Zhang Volatility
- Parkinson Volatility
- Garman-Klass Volatility
- Mass Index
- Aberration
- HWC (Holt-Winter Channel)
- PDIST (Price Distance)
- Thermo
"""

import logging
from typing import Tuple, cast

import numpy as np
import pandas as pd
import pandas_ta_classic as ta  # type: ignore
from numba import njit, prange

from ..data_validation import (
    create_nan_series_bundle,
    handle_pandas_ta_errors,
    run_multi_series_indicator,
    run_series_indicator,
)


@njit(parallel=True, cache=True)
def _njit_yang_zhang_loop(open_arr, high_arr, low_arr, close_arr, length):
    """
    Yang-Zhang ボラティリティを計算する Numba 加速ループ。

    Args:
        open_arr: 始値の配列
        high_arr: 高値の配列
        low_arr: 安値の配列
        close_arr: 終値の配列
        length: 計算期間

    Returns:
        Yang-Zhang ボラティリティの配列
    """
    n = len(open_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length + 1:
        return result

    # Standard Yang-Zhang uses rolling variance of log returns
    # We can pre-calculate the log returns
    log_oc = np.zeros(n)
    log_co = np.zeros(n)
    rs_term = np.zeros(n)

    for i in prange(1, n):
        log_oc[i] = np.log(open_arr[i] / close_arr[i - 1])
        log_co[i] = np.log(close_arr[i] / open_arr[i])
        rs_term[i] = (
            np.log(high_arr[i] / close_arr[i]) * np.log(high_arr[i] / open_arr[i])
        ) + (np.log(low_arr[i] / close_arr[i]) * np.log(low_arr[i] / open_arr[i]))

    k = 0.34 / (1.34 + (length + 1) / (length - 1))

    # Rolling calculations
    for i in prange(length, n):
        # Rolling variance (unbiased) of log_oc and log_co
        # Rolling mean of rs_term
        s_oc1, s_oc2 = 0.0, 0.0
        s_co1, s_co2 = 0.0, 0.0
        s_rs = 0.0

        for j in range(i - length + 1, i + 1):
            v_oc = log_oc[j]
            v_co = log_co[j]
            s_oc1 += v_oc
            s_oc2 += v_oc * v_oc
            s_co1 += v_co
            s_co2 += v_co * v_co
            s_rs += rs_term[j]

        v_oc_final = (s_oc2 - (s_oc1 * s_oc1) / length) / (length - 1)
        v_co_final = (s_co2 - (s_co1 * s_co1) / length) / (length - 1)
        m_rs = s_rs / length

        yz_variance = v_oc_final + k * v_co_final + (1.0 - k) * m_rs
        if yz_variance > 0:
            result[i] = np.sqrt(yz_variance)
        else:
            result[i] = 0.0

    return result


logger = logging.getLogger(__name__)


@njit(parallel=True, cache=True)
def _njit_parkinson_loop(high_arr, low_arr, length):
    """
    Parkinson ボラティリティを計算する Numba 加速ループ。

    Args:
        high_arr: 高値の配列
        low_arr: 安値の配列
        length: 計算期間

    Returns:
        Parkinson ボラティリティの配列
    """
    n = len(high_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return result

    const = 1.0 / (4.0 * np.log(2.0))
    log_hl_sq = np.zeros(n)
    for i in prange(n):
        if low_arr[i] > 0:
            log_hl_sq[i] = np.log(high_arr[i] / low_arr[i]) ** 2

    for i in prange(length - 1, n):
        s = 0.0
        for j in range(i - length + 1, i + 1):
            s += log_hl_sq[j]

        result[i] = np.sqrt(const * (s / length))

    return result


@njit(parallel=True, cache=True)
def _njit_garman_klass_loop(open_arr, high_arr, low_arr, close_arr, length):
    """
    Garman-Klass ボラティリティを計算する Numba 加速ループ。

    Args:
        open_arr: 始値の配列
        high_arr: 高値の配列
        low_arr: 安値の配列
        close_arr: 終値の配列
        length: 計算期間

    Returns:
        Garman-Klass ボラティリティの配列
    """
    n = len(open_arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return result

    const = 2.0 * np.log(2.0) - 1.0
    inst_var = np.zeros(n)
    for i in prange(n):
        if low_arr[i] > 0 and open_arr[i] > 0:
            v1 = 0.5 * (np.log(high_arr[i] / low_arr[i]) ** 2)
            v2 = const * (np.log(close_arr[i] / open_arr[i]) ** 2)
            val = v1 - v2
            inst_var[i] = val if val > 0 else 0.0

    for i in prange(length - 1, n):
        s = 0.0
        for j in range(i - length + 1, i + 1):
            s += inst_var[j]
        result[i] = np.sqrt(s / length)

    return result


class VolatilityIndicators:
    """
    ボラティリティ系指標クラス

    ATR, Bollinger Bandsなどのボラティリティ系テクニカル指標を提供。
    市場の変動性とリスク評価に使用します。
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

        def compute():
            """ATR を計算するヘルパー関数"""
            result = ta.atr(high=high, low=low, close=close, length=length)
            if result is None:
                logger.error("ATR: Calculation returned None - returning NaN series")
            return result

        return run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            length,
            compute,
            min_data_length=length,
        )

    @staticmethod
    @handle_pandas_ta_errors
    def natr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """Normalized Average True Range"""
        return run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            length,
            lambda: ta.natr(high=high, low=low, close=close, length=length),
            min_data_length=length,
        )

    @staticmethod
    @handle_pandas_ta_errors
    def bbands(
        data: pd.Series, length: int = 20, std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ボリンジャーバンド"""
        result = run_series_indicator(
            data,
            length,
            lambda: ta.bbands(data, length=length, std=std),
            fallback_factory=lambda: cast(
                Tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(data, 3),
            ),
        )

        if isinstance(result, tuple):
            return cast(Tuple[pd.Series, pd.Series, pd.Series], result)

        if result is None:
            logger.error("BBands: Calculation returned None - returning NaN series")
            return cast(
                Tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(data, 3),
            )

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
        period: int = 20,
        scalar: float = 2.0,
        mamode: str = "sma",
        std_dev: bool = False,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels: returns (upper, middle, lower)"""

        def nan_result() -> Tuple[pd.Series, pd.Series, pd.Series]:
            """計算失敗時に NaN の Series を返すヘルパー関数"""
            return cast(
                Tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(close, 3),
            )

        df = run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            period,
            lambda: ta.kc(
                high=high,
                low=low,
                close=close,
                length=period,
                scalar=scalar,
                mamode=mamode,
            ),
            fallback_factory=nan_result,
        )

        if isinstance(df, tuple):
            return cast(Tuple[pd.Series, pd.Series, pd.Series], df)

        if df.empty:
            return nan_result()

        # カラム名: KC{mamode[0]}_{length}_{scalar}
        m = mamode[0].lower()
        # pandas-ta は整数として埋め込む場合と浮動小数点として埋め込む場合があるため、両方試行
        try:
            # 浮動小数点形式 (例: 2.0)
            return (
                df[f"KCU{m}_{period}_{float(scalar)}"],
                df[f"KCB{m}_{period}_{float(scalar)}"],
                df[f"KCL{m}_{period}_{float(scalar)}"],
            )
        except KeyError:
            try:
                # 整数形式 (例: 2)
                return (
                    df[f"KCU{m}_{period}_{int(scalar)}"],
                    df[f"KCB{m}_{period}_{int(scalar)}"],
                    df[f"KCL{m}_{period}_{int(scalar)}"],
                )
            except (KeyError, Exception):
                return nan_result()

    @staticmethod
    @handle_pandas_ta_errors
    def donchian(
        high: pd.Series,
        low: pd.Series,
        length: int = 20,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels: returns (upper, middle, lower)"""

        def nan_result() -> Tuple[pd.Series, pd.Series, pd.Series]:
            return cast(
                Tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(high, 3),
            )

        df = run_multi_series_indicator(
            {"high": high, "low": low},
            length,
            lambda: ta.donchian(high=high, low=low, length=length),
            fallback_factory=nan_result,
        )

        if isinstance(df, tuple):
            return cast(Tuple[pd.Series, pd.Series, pd.Series], df)

        if df.empty:
            return nan_result()

        # カラム名: DCU_{length}_{length}, DCM_{length}_{length}, DCL_{length}_{length}
        try:
            return (
                df[f"DCU_{length}_{length}"],
                df[f"DCM_{length}_{length}"],
                df[f"DCL_{length}_{length}"],
            )
        except (KeyError, Exception):
            return nan_result()

    @staticmethod
    @handle_pandas_ta_errors
    def accbands(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Acceleration Bands: returns (upper, middle, lower)"""

        def nan_result() -> Tuple[pd.Series, pd.Series, pd.Series]:
            return cast(
                Tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(close, 3),
            )

        result = run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            period,
            lambda: ta.accbands(high=high, low=low, close=close, length=period),
            fallback_factory=nan_result,
        )

        if isinstance(result, tuple):
            return cast(Tuple[pd.Series, pd.Series, pd.Series], result)

        if result.empty:
            return nan_result()

        # カラム名: ACCBU_{length}, ACCBM_{length}, ACCBL_{length}
        try:
            return (
                result[f"ACCBU_{period}"],
                result[f"ACCBM_{period}"],
                result[f"ACCBL_{period}"],
            )
        except (KeyError, Exception):
            return nan_result()

    @staticmethod
    @handle_pandas_ta_errors
    def ui(data: pd.Series, period: int = 14) -> pd.Series:
        """Ulcer Index"""
        return run_series_indicator(data, None, lambda: ta.ui(data, window=period))

    @staticmethod
    @handle_pandas_ta_errors
    def rvi(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        length: int = 14,
        scalar: float = 100.0,
        refined: bool = False,
        thirds: bool = False,
        mamode: str | None = None,
        drift: int | None = None,
        offset: int | None = None,
    ) -> pd.Series:
        """Relative Volatility Index"""
        return run_multi_series_indicator(
            {"close": close, "high": high, "low": low},
            length,
            lambda: ta.rvi(
                close=close,
                high=high,
                low=low,
                length=length,
                scalar=scalar,
                refined=refined,
                thirds=thirds,
                mamode=mamode,
                drift=drift,
                offset=offset,
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def true_range(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        drift: int = 1,
    ) -> pd.Series:
        """True Range"""
        return run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            None,
            lambda: ta.true_range(high=high, low=low, close=close, drift=drift),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def yang_zhang(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 20,
    ) -> pd.Series:
        """
        Yang-Zhang Volatility Estimator - Numba Optimized Version
        """
        result = run_multi_series_indicator(
            {"open_": open_, "high": high, "low": low, "close": close},
            length,
            lambda: pd.Series(
                _njit_yang_zhang_loop(
                    open_.values.astype(np.float64),
                    high.values.astype(np.float64),
                    low.values.astype(np.float64),
                    close.values.astype(np.float64),
                    length,
                ),
                index=close.index,
                name=f"YZVOL_{length}",
            ),
        )
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def parkinson(
        high: pd.Series,
        low: pd.Series,
        length: int = 20,
    ) -> pd.Series:
        """
        Parkinson Volatility Estimator - Numba Optimized Version
        """
        return run_multi_series_indicator(
            {"high": high, "low": low},
            length,
            lambda: pd.Series(
                _njit_parkinson_loop(
                    high.values.astype(np.float64),
                    low.values.astype(np.float64),
                    length,
                ),
                index=high.index,
                name=f"PARKVOL_{length}",
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def garman_klass(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 20,
    ) -> pd.Series:
        """
        Garman-Klass Volatility Estimator - Numba Optimized Version
        """
        return run_multi_series_indicator(
            {"open_": open_, "high": high, "low": low, "close": close},
            length,
            lambda: pd.Series(
                _njit_garman_klass_loop(
                    open_.values.astype(np.float64),
                    high.values.astype(np.float64),
                    low.values.astype(np.float64),
                    close.values.astype(np.float64),
                    length,
                ),
                index=close.index,
                name=f"GKVOL_{length}",
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def massi(
        high: pd.Series,
        low: pd.Series,
        fast: int = 9,
        slow: int = 25,
    ) -> pd.Series:
        """Mass Index

        トレンドの反転を予測するためのボラティリティ指標。
        高値と安値のレンジ拡大パターンを検出。

        Args:
            high: 高値
            low: 安値
            fast: 高速 EMA 期間（デフォルト: 9）
            slow: 低速 EMA 期間（デフォルト: 25）

        Returns:
            Mass Index
        """
        if fast <= 0:
            raise ValueError("fast must be positive")

        return run_multi_series_indicator(
            {"high": high, "low": low},
            slow,
            lambda: ta.massi(high=high, low=low, fast=fast, slow=slow),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def aberration(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 5,
        atr_length: int = 15,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Aberration"""
        result = run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            max(length, atr_length),
            lambda: ta.aberration(
                high=high, low=low, close=close, length=length, atr_length=atr_length
            ),
            fallback_factory=lambda: cast(
                Tuple[pd.Series, pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(close, 4),
            ),
        )

        if isinstance(result, tuple):
            return cast(Tuple[pd.Series, pd.Series, pd.Series, pd.Series], result)

        if result.empty:
            return cast(
                Tuple[pd.Series, pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(close, 4),
            )

        # Returns multiple columns. Usually ZG, SG, XG, ATR
        return (
            result.iloc[:, 0],
            result.iloc[:, 1],
            result.iloc[:, 2],
            result.iloc[:, 3],
        )

    @staticmethod
    @handle_pandas_ta_errors
    def hwc(
        close: pd.Series,
        na: int = 2,
        nb: int = 3,
        nc: int = 4,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Holt-Winter Channel"""
        result = run_series_indicator(
            close,
            max(na, nb, nc),
            lambda: ta.hwc(close=close, na=na, nb=nb, nc=nc),
            fallback_factory=lambda: cast(
                Tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(close, 3),
            ),
        )

        if isinstance(result, tuple):
            return cast(Tuple[pd.Series, pd.Series, pd.Series], result)

        if result.empty:
            return cast(
                Tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(close, 3),
            )

        return result.iloc[:, 0], result.iloc[:, 1], result.iloc[:, 2]

    @staticmethod
    @handle_pandas_ta_errors
    def pdist(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Price Distance"""
        return run_multi_series_indicator(
            {"open_": open_, "high": high, "low": low, "close": close},
            None,
            lambda: ta.pdist(open_=open_, high=high, low=low, close=close),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def thermo(
        high: pd.Series,
        low: pd.Series,
        length: int = 20,
        long_: int = 2,
        short: int = 2,
        mamode: str = "ema",
        drift: int = 1,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Thermo"""
        result = run_multi_series_indicator(
            {"high": high, "low": low},
            length,
            lambda: ta.thermo(
                high=high,
                low=low,
                length=length,
                long=long_,
                short=short,
                mamode=mamode,
                drift=drift,
            ),
            fallback_factory=lambda: cast(
                Tuple[pd.Series, pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(high, 4),
            ),
        )

        if isinstance(result, tuple):
            return cast(Tuple[pd.Series, pd.Series, pd.Series, pd.Series], result)

        if result.empty:
            return cast(
                Tuple[pd.Series, pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(high, 4),
            )

        # Returns Thermo, ThermoMa, ThermoLa, ThermoSa
        return (
            result.iloc[:, 0],
            result.iloc[:, 1],
            result.iloc[:, 2],
            result.iloc[:, 3],
        )

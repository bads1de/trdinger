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
- Relative Volatility Index (RVI)
- True Range
- Yang-Zhang Volatility
- Parkinson Volatility
- Garman-Klass Volatility
- Mass Index (MASSI)
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
    validate_series_params,
)

TA_LIB_AVAILABLE = False
try:
    import talib  # noqa: F401

    TA_LIB_AVAILABLE = True
except ImportError:
    TA_LIB_AVAILABLE = False


logger = logging.getLogger(__name__)


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
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, length, length
        )
        if validation is not None:
            return validation

        result = ta.atr(
            high=high.values, low=low.values, close=close.values, length=length
        )

        if result is None:
            logger.error("ATR: Calculation returned None - returning NaN series")
            return pd.Series(np.full(len(high), np.nan), index=high.index)

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def natr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """Normalized Average True Range"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, length, length
        )
        if validation is not None:
            return validation

        result = ta.natr(high=high, low=low, close=close, length=length)
        if result is None or (hasattr(result, "empty") and result.empty):
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def bbands(
        data: pd.Series, length: int = 20, std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ボリンジャーバンド"""
        validation = validate_series_params(data, length)
        if validation is not None:
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return (nan_series, nan_series, nan_series)

        result = ta.bbands(data, length=length, std=std)

        if result is None:
            logger.error("BBands: Calculation returned None - returning NaN series")
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return (nan_series, nan_series, nan_series)

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
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, period
        )

        def nan_result() -> Tuple[pd.Series, pd.Series, pd.Series]:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series.copy(), nan_series.copy()

        if validation is not None:
            return nan_result()

        # keltner (kc) を計算
        df = ta.kc(
            high=high,
            low=low,
            close=close,
            length=period,
            scalar=scalar,
            mamode=mamode,
        )

        if df is None or df.empty:
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
        validation = validate_multi_series_params({"high": high, "low": low}, length)

        def nan_result() -> Tuple[pd.Series, pd.Series, pd.Series]:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series.copy(), nan_series.copy()

        if validation is not None:
            return nan_result()

        df = ta.donchian(high=high, low=low, length=length)

        if df is None or df.empty:
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
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, period
        )

        def nan_result() -> Tuple[pd.Series, pd.Series, pd.Series]:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series.copy(), nan_series.copy()

        if validation is not None:
            return nan_result()

        result = ta.accbands(high=high, low=low, close=close, length=period)

        if result is None or result.empty:
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
        validation = validate_series_params(data)
        if validation is not None:
            return validation

        length = period
        result = ta.ui(data, window=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

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

        validation = validate_multi_series_params(
            {"close": close, "high": high, "low": low}, length
        )
        if validation is not None:
            return validation

        result = ta.rvi(
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
        )

        if result is None or (hasattr(result, "isna") and result.isna().all()):
            return pd.Series(np.full(len(close), np.nan), index=close.index)

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def true_range(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        drift: int = 1,
    ) -> pd.Series:
        """True Range"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}
        )
        if validation is not None:
            return validation

        result = ta.true_range(high=high, low=low, close=close, drift=drift)

        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)

        return result

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
        Yang-Zhang Volatility Estimator
        """
        validation = validate_multi_series_params(
            {"open_": open_, "high": high, "low": low, "close": close}, length
        )
        if validation is not None:
            return validation

        N = length

        # 1. Overnight Volatility (Close[t-1] to Open[t])
        # log(O_t / C_{t-1})
        log_oc = np.log(open_ / close.shift(1))
        # sum(log_oc - mean(log_oc))^2 / (N - 1)
        # pandas rolling().var() calculates unbiased variance (/(N-1)) by default
        vol_overnight = log_oc.rolling(window=N).var()

        # 2. Open-to-Close Volatility (Open[t] to Close[t])
        # log(C_t / O_t)
        log_co = np.log(close / open_)
        vol_open_to_close = log_co.rolling(window=N).var()

        # 3. Rogers-Satchell Volatility
        # log(H/C) * log(H/O) + log(L/C) * log(L/O)
        rs_term = (np.log(high / close) * np.log(high / open_)) + (
            np.log(low / close) * np.log(low / open_)
        )

        vol_rs = rs_term.rolling(window=N).mean()

        # Yang-Zhang Weights
        k = 0.34 / (1.34 + (N + 1) / (N - 1))

        # Combine
        # YZ^2 = Vol_Overnight + k * Vol_OpenClose + (1-k) * Vol_RS
        yz_variance = vol_overnight + k * vol_open_to_close + (1 - k) * vol_rs

        # Return volatility (std dev), annualized if needed but here just raw
        # Usually multiplied by sqrt(periods_per_year) for annualized,
        # but we keep it as raw volatility per bar
        return np.sqrt(yz_variance)

    @staticmethod
    @handle_pandas_ta_errors
    def parkinson(
        high: pd.Series,
        low: pd.Series,
        length: int = 20,
    ) -> pd.Series:
        """
        Parkinson Volatility Estimator
        """
        validation = validate_multi_series_params({"high": high, "low": low}, length)
        if validation is not None:
            return validation

        # Prevent division by zero or log of zero/negative
        # Assuming High >= Low > 0
        hl_ratio = high / low

        # Replace any potential invalid values (though theoretically shouldn't exist in valid OHLC)
        hl_ratio = hl_ratio.replace([np.inf, -np.inf], np.nan)

        # (ln(H/L))^2
        log_hl_sq = np.log(hl_ratio) ** 2

        # Constant: 1 / (4 * ln(2))
        const = 1.0 / (4.0 * np.log(2.0))

        # Instantaneous variance estimate
        inst_var = const * log_hl_sq

        # Rolling mean of variance
        rolling_var = inst_var.rolling(window=length).mean()

        # Return volatility (std dev)
        return np.sqrt(rolling_var)

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
        Garman-Klass Volatility Estimator
        """
        validation = validate_multi_series_params(
            {"open_": open_, "high": high, "low": low, "close": close}, length
        )
        if validation is not None:
            return validation

        # 1. (ln(H/L))^2
        log_hl_sq = np.log(high / low) ** 2

        # 2. (ln(C/O))^2
        log_co_sq = np.log(close / open_) ** 2

        # Constant: 2*ln(2) - 1
        const = 2.0 * np.log(2.0) - 1.0

        # Instantaneous variance estimate
        inst_var = 0.5 * log_hl_sq - const * log_co_sq

        # Ensure non-negative (theoretical variance should be positive, but numerical issues can occur)
        inst_var = inst_var.clip(lower=0.0)

        # Rolling mean of variance
        rolling_var = inst_var.rolling(window=length).mean()

        # Return volatility (std dev)
        return np.sqrt(rolling_var)

    @staticmethod
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
        validation = validate_multi_series_params({"high": high, "low": low}, slow)
        if validation is not None:
            return validation

        if fast <= 0:
            raise ValueError("fast must be positive")

        result = ta.massi(high=high, low=low, fast=fast, slow=slow)
        if result is None or result.empty:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

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
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, max(length, atr_length)
        )
        if validation is not None:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series, nan_series

        result = ta.aberration(
            high=high, low=low, close=close, length=length, atr_length=atr_length
        )
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series, nan_series

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
        validation = validate_series_params(close, max(na, nb, nc))
        if validation is not None:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series

        result = ta.hwc(close=close, na=na, nb=nb, nc=nc)
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series

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
        validation = validate_multi_series_params(
            {"open_": open_, "high": high, "low": low, "close": close}
        )
        if validation is not None:
            return validation

        result = ta.pdist(open_=open_, high=high, low=low, close=close)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

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
        validation = validate_multi_series_params({"high": high, "low": low}, length)
        if validation is not None:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series, nan_series, nan_series

        result = ta.thermo(
            high=high,
            low=low,
            length=length,
            long=long_,
            short=short,
            mamode=mamode,
            drift=drift,
        )
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series, nan_series, nan_series

        # Returns Thermo, ThermoMa, ThermoLa, ThermoSa
        return (
            result.iloc[:, 0],
            result.iloc[:, 1],
            result.iloc[:, 2],
            result.iloc[:, 3],
        )

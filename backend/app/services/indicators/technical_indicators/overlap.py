"""
オーバーレイ系テクニカル指標 (Overlap Indicators)

pandas-ta の overlap カテゴリに対応。
価格チャートに重ねて表示する指標群。

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
- RMA (Wilde's Moving Average)
- LINREG (Linear Regression)
- LINREGSLOPE (Linear Regression Slope)
- Supertrend
- Ichimoku Cloud (一目均衡表)
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

from ..data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
    validate_series_params,
)

logger = logging.getLogger(__name__)


class OverlapIndicators:
    """
    オーバーレイ系指標クラス

    移動平均線、Supertrend などの価格に重ねて表示するテクニカル指標を提供。
    """

    @staticmethod
    @handle_pandas_ta_errors
    def sma(data: pd.Series, length: int) -> pd.Series:
        """単純移動平均"""
        validation = validate_series_params(data, length)
        if validation is not None:
            return validation

        return ta.sma(data, length=length)

    @staticmethod
    @handle_pandas_ta_errors
    def ema(data: pd.Series, length: int) -> pd.Series:
        """指数移動平均"""
        validation = validate_series_params(data, length)
        if validation is not None:
            return validation
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

        validation = validate_series_params(data, length)
        if validation is not None:
            return validation
        return ta.wma(data, window=length)

    @staticmethod
    @handle_pandas_ta_errors
    def trima(
        data: pd.Series, length: int = 10, talib: bool | None = None
    ) -> pd.Series:
        """三角移動平均"""
        validation = validate_series_params(data, length)
        if validation is not None:
            return validation

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
        validation = validate_series_params(data, length)
        if validation is not None:
            return validation

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
        validation = validate_series_params(data, length)
        if validation is not None:
            return validation
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
        validation = validate_series_params(data, length, min_data_length=length * 2)
        if validation is not None:
            return validation

        result = ta.dema(data, window=length)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def tema(data: pd.Series, length: int) -> pd.Series:
        """三重指数移動平均"""
        validation = validate_series_params(data, length, min_data_length=length * 3)
        if validation is not None:
            return validation

        return ta.tema(data, window=length)

    @staticmethod
    @handle_pandas_ta_errors
    def t3(data: pd.Series, length: int, a: float = 0.7) -> pd.Series:
        """T3移動平均"""
        validation = validate_series_params(data, length, min_data_length=length * 6)
        if validation is not None:
            return validation

        # Use pandas-ta directly
        result = ta.t3(data, window=length, a=a)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def kama(data: pd.Series, length: int = 30) -> pd.Series:
        """カウフマン適応移動平均"""
        validation = validate_series_params(data, length)
        if validation is not None:
            return validation
        return ta.kama(data, window=length)

    @staticmethod
    @handle_pandas_ta_errors
    def hma(data: pd.Series, length: int = 20) -> pd.Series:
        """Hull移動平均"""
        validation = validate_series_params(data, length)
        if validation is not None:
            return validation

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
        validation = validate_multi_series_params(
            {"close": close, "volume": volume}, length
        )
        if validation is not None:
            return validation

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
        """線形回帰 (pandas-ta ベクトル化版)"""
        validation = validate_series_params(data, length, min_data_length=length)
        if validation is not None:
            return validation

        # pandas-ta の linreg を使用 (内部でベクトル化されている)
        # intercept=True の場合は y切片そのものを返す
        # intercept=False の場合はその時点での回帰線の値を返す
        # offset は 0 固定 (デフォルト)
        # tsf (Time Series Forecast) は linreg と同様
        result = ta.linreg(data, length=length, offset=0)

        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # interceptが必要な場合（あまり一般的ではないが元のコードにあるためサポート）
        if intercept:
            # intercept = y - slope * x
            # pandas-taには直接interceptを出すフラグがないため、slopeを取得して逆算
            slope = ta.linreg(data, length=length, slope=True)
            # xの中心を (length-1) と想定
            intercept_val = result - slope * (length - 1)
            return intercept_val * scalar

        return result * scalar

    @staticmethod
    @handle_pandas_ta_errors
    def linregslope(
        data: pd.Series, length: int = 14, scalar: float = 1.0
    ) -> pd.Series:
        """線形回帰スロープ (pandas-ta ベクトル化版)"""
        validation = validate_series_params(data, length, min_data_length=length)
        if validation is not None:
            return validation

        # pandas-ta の linreg に slope=True を指定
        result = ta.linreg(data, length=length, slope=True)

        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        return result * scalar

    @staticmethod
    @handle_pandas_ta_errors
    def rma(data: pd.Series, length: int = 10) -> pd.Series:
        """Wilde's Moving Average"""
        validation = validate_series_params(data, length)
        if validation is not None:
            return validation

        result = ta.rma(data, length=length)
        if result is None or (hasattr(result, "empty") and result.empty):
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def supertrend(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 7,
        multiplier: float = 3.0,
        **kwargs,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Supertrend インジケーター

        Args:
            high: 高値
            low: 安値
            close: 終値
            period: 期間（デフォルト: 7）
            multiplier: ATR乗数（デフォルト: 3.0）
            **kwargs: 'factor' を 'multiplier' のエイリアスとしてサポート

        Returns:
            Tuple[lower, upper, direction]:
                - lower: 下側バンド (SUPERTl)
                - upper: 上側バンド (SUPERTs)
                - direction: 方向 (1=強気, -1=弱気)
        """
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, period
        )

        def nan_result() -> Tuple[pd.Series, pd.Series, pd.Series]:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series.copy(), nan_series.copy()

        if validation is not None:
            return nan_result()

        # 'factor' を 'multiplier' のエイリアスとしてサポート
        if "factor" in kwargs:
            multiplier = kwargs["factor"]

        # Numba Implementation
        try:
            h_arr = high.values.astype(np.float64)
            l_arr = low.values.astype(np.float64)
            c_arr = close.values.astype(np.float64)

            lower, upper, trend = OverlapIndicators._supertrend_loop(
                h_arr, l_arr, c_arr, period, float(multiplier)
            )

            idx = high.index
            # Construct DataFrame or Series as originally returned
            # Original: SUPERTl, SUPERTs, SUPERTd

            # Format depends on multiplier (int or float in name)

            # But wait, original code tries float key first then int key.
            # Let's just return the series directly without worrying about column names for now,
            # Or reconstruct exact keys if necessary. The calling code expects Tuple[Series, Series, Series].

            return (
                pd.Series(lower, index=idx, name=f"SUPERTl_{period}_{multiplier}"),
                pd.Series(upper, index=idx, name=f"SUPERTs_{period}_{multiplier}"),
                pd.Series(trend, index=idx, name=f"SUPERTd_{period}_{multiplier}"),
            )

        except Exception as e:
            logger.warning(
                f"Supertrend Numba optimization failed: {e}. Falling back..."
            )
            pass

        df = ta.supertrend(
            high=high, low=low, close=close, length=period, multiplier=multiplier
        )

        if df is None or df.empty:
            return nan_result()

        # カラム名: SUPERTl_{length}_{multiplier}, SUPERTs_{length}_{multiplier}, SUPERTd_{length}_{multiplier}
        try:
            # 浮動小数点形式 (例: 3.0)
            return (
                df[f"SUPERTl_{period}_{float(multiplier)}"],
                df[f"SUPERTs_{period}_{float(multiplier)}"],
                df[f"SUPERTd_{period}_{float(multiplier)}"],
            )
        except KeyError:
            try:
                # 整数形式 (例: 3)
                return (
                    df[f"SUPERTl_{period}_{int(multiplier)}"],
                    df[f"SUPERTs_{period}_{int(multiplier)}"],
                    df[f"SUPERTd_{period}_{int(multiplier)}"],
                )
            except (KeyError, Exception):
                return nan_result()

    @staticmethod
    def ichimoku(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_span_b_period: int = 52,
    ) -> Dict[str, pd.Series]:
        """Ichimoku Cloud (一目均衡表)

        Args:
            high: 高値
            low: 安値
            close: 終値
            tenkan_period: 転換線期間（デフォルト: 9）
            kijun_period: 基準線期間（デフォルト: 26）
            senkou_span_b_period: 先行スパンB期間（デフォルト: 52）

        Returns:
            Dict with keys: tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
        """
        max_period = max(tenkan_period, kijun_period, senkou_span_b_period)
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, max_period
        )

        def nan_result() -> Dict[str, pd.Series]:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return {
                "tenkan_sen": nan_series.copy(),
                "kijun_sen": nan_series.copy(),
                "senkou_span_a": nan_series.copy(),
                "senkou_span_b": nan_series.copy(),
                "chikou_span": nan_series.copy(),
            }

        if validation is not None:
            return nan_result()

        # pandas-ta はタプル (ichimoku_df, span_df) を返す
        try:
            result = ta.ichimoku(
                high=high,
                low=low,
                close=close,
                tenkan=tenkan_period,
                kijun=kijun_period,
                senkou=senkou_span_b_period,
            )

            # resultがNone、またはタプルで最初の要素がNone/空の場合
            if result is None:
                return nan_result()

            if isinstance(result, tuple):
                if result[0] is None or result[0].empty:
                    return nan_result()
                df = result[0]
            else:
                if result.empty:
                    return nan_result()
                df = result

            # カラム名パターン: ITS_{tenkan}, IKS_{kijun}, ISA_{tenkan}, ISB_{kijun}, ICS_{kijun}
            return {
                "tenkan_sen": df[f"ITS_{tenkan_period}"],
                "kijun_sen": df[f"IKS_{kijun_period}"],
                "senkou_span_a": df[f"ISA_{tenkan_period}"],
                "senkou_span_b": df[f"ISB_{kijun_period}"],
                "chikou_span": df[f"ICS_{kijun_period}"],
            }
        except (KeyError, Exception):
            # pandas-taが想定外の結果を返した場合
            return nan_result()

    @staticmethod
    @handle_pandas_ta_errors
    def hilo(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        high_length: int = 13,
        low_length: int = 13,
        mamode: str = "sma",
        offset: int = 0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Gann HiLo"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, max(high_length, low_length)
        )
        if validation is not None:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series

        result = ta.hilo(
            high=high,
            low=low,
            close=close,
            high_length=high_length,
            low_length=low_length,
            mamode=mamode,
            offset=offset,
        )
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series

        # Returns HILO, HILOl, HILOs
        return result.iloc[:, 0], result.iloc[:, 1], result.iloc[:, 2]

    @staticmethod
    @handle_pandas_ta_errors
    def hl2(high: pd.Series, low: pd.Series) -> pd.Series:
        """High-Low Average"""
        validation = validate_multi_series_params({"high": high, "low": low})
        if validation is not None:
            return validation  # Should return Series

        result = ta.hl2(high=high, low=low)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def hlc3(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """HLC Average"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}
        )
        if validation is not None:
            return validation

        result = ta.hlc3(high=high, low=low, close=close)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def ohlc4(
        open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """OHLC Average"""
        validation = validate_multi_series_params(
            {"open_": open_, "high": high, "low": low, "close": close}
        )
        if validation is not None:
            return validation

        result = ta.ohlc4(open_=open_, high=high, low=low, close=close)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def midpoint(close: pd.Series, length: int = 2) -> pd.Series:
        """Midpoint"""
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        result = ta.midpoint(close=close, length=length)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def midprice(high: pd.Series, low: pd.Series, length: int = 2) -> pd.Series:
        """Midprice"""
        validation = validate_multi_series_params({"high": high, "low": low}, length)
        if validation is not None:
            return validation

        result = ta.midprice(high=high, low=low, length=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def vidya(
        close: pd.Series, length: int = 14, drift: int = 1, offset: int = 0
    ) -> pd.Series:
        """VIDYA"""
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        result = ta.vidya(close=close, length=length, drift=drift, offset=offset)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def wcp(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Weighted Close Price"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}
        )
        if validation is not None:
            return validation

        result = ta.wcp(high=high, low=low, close=close)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def mcgd(close: pd.Series, length: int = 10, offset: int = 0) -> pd.Series:
        """McGinley Dynamic"""
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        result = ta.mcgd(close=close, length=length, offset=offset)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def jma(
        close: pd.Series, length: int = 7, phase: int = 50, offset: int = 0
    ) -> pd.Series:
        """Jurik Moving Average"""
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        result = ta.jma(close=close, length=length, phase=phase, offset=offset)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def fwma(close: pd.Series, length: int = 10, asc: bool = True) -> pd.Series:
        """Fibonacci Weighted Moving Average"""
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        result = ta.fwma(close=close, length=length, asc=asc)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def pwma(close: pd.Series, length: int = 10, asc: bool = True) -> pd.Series:
        """Pascal Weighted Moving Average"""
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        result = ta.pwma(close=close, length=length, asc=asc)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def sinwma(close: pd.Series, length: int = 14) -> pd.Series:
        """Sine Weighted Moving Average"""
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        result = ta.sinwma(close=close, length=length)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def ssf(close: pd.Series, length: int = 10, poles: int = 2) -> pd.Series:
        """Ehlers Super Smoother Filter"""
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        result = ta.ssf(close=close, length=length, poles=poles)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def swma(close: pd.Series, length: int = 10) -> pd.Series:
        """Symmetric Weighted Moving Average"""
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        result = ta.swma(close=close, length=length)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

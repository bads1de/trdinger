"""
トレンド系テクニカル指標 (Trend Indicators)

pandas-ta の trend カテゴリに対応。
トレンドの方向性と強さを分析する指標群。

登録してあるテクニカルの一覧:
- SAR (Parabolic SAR)
- AMAT (Archer Moving Averages Trends)
- DPO (Detrended Price Oscillator)
- VORTEX (Vortex Indicator)
- ADX (Average Directional Index)
- AROON (Aroon Indicator)
- CHOP (Choppiness Index)
- VHF (Vertical Horizontal Filter)
"""

import logging
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
    validate_series_params,
)

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

    Parabolic SAR, ADX などのトレンド方向性を分析するテクニカル指標を提供。
    トレンドの方向性と強さの分析に使用します。
    """

    @staticmethod
    @handle_pandas_ta_errors
    def sar(
        high: pd.Series,
        low: pd.Series,
        af: float = 0.02,
        max_af: float = 0.2,
    ) -> pd.Series:
        """
        パラボリックSAR (Stable Implementation)

        pandas_taのデフォルト実装はデータの全長に依存して初期化されるため、
        未来データのリークを防ぐためにカスタム実装を使用します。
        """
        validation = validate_multi_series_params({"high": high, "low": low})
        if validation is not None:
            return validation

        n = len(high)
        if n < 2:
            return pd.Series(np.nan, index=high.index)

        high_arr = high.values
        low_arr = low.values
        sar = np.zeros(n)

        # 初期状態の設定
        # 最初のトレンドを決定するために2本目の足を使用
        is_long = high_arr[1] > high_arr[0]
        af_val = af

        if is_long:
            sar[1] = low_arr[0]
            ep = high_arr[1]
        else:
            sar[1] = high_arr[0]
            ep = low_arr[1]

        for i in range(2, n):
            # 前回のSARを計算
            prev_sar = sar[i - 1]

            if is_long:
                sar[i] = prev_sar + af_val * (ep - prev_sar)
                # SARは直近2期間の安値を超えてはならない
                sar[i] = min(sar[i], low_arr[i - 1], low_arr[i - 2])

                if low_arr[i] < sar[i]:
                    # トレンド転換 (Long -> Short)
                    is_long = False
                    sar[i] = ep
                    ep = low_arr[i]
                    af_val = af
                else:
                    # トレンド継続
                    if high_arr[i] > ep:
                        ep = high_arr[i]
                        af_val = min(af_val + af, max_af)
            else:
                sar[i] = prev_sar + af_val * (ep - prev_sar)
                # SARは直近2期間の高値を超えてはならない
                sar[i] = max(sar[i], high_arr[i - 1], high_arr[i - 2])

                if high_arr[i] > sar[i]:
                    # トレンド転換 (Short -> Long)
                    is_long = True
                    sar[i] = ep
                    ep = high_arr[i]
                    af_val = af
                else:
                    # トレンド継続
                    if low_arr[i] < ep:
                        ep = low_arr[i]
                        af_val = min(af_val + af, max_af)

        return pd.Series(sar, index=high.index).replace(0, np.nan)

    @staticmethod
    @handle_pandas_ta_errors
    def amat(
        data: pd.Series, fast: int = 3, slow: int = 30, signal: int = 10
    ) -> pd.Series:
        """Archer Moving Averages Trends"""
        # AMAT特有のデータ検証
        min_length = max(fast, slow, signal) + 10
        validation = validate_series_params(data, min_data_length=min_length)
        if validation is not None:
            return validation

        result = ta.amat(data, fast=fast, slow=slow, signal=signal)
        if result is None or (hasattr(result, "empty") and result.empty):
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        # AMAT returns DataFrame, get the main series
        if hasattr(result, "iloc"):
            return result.iloc[:, 0] if len(result.shape) > 1 else result
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def dpo(
        data: pd.Series,
        length: int = 20,
        centered: bool = True,
        offset: int = 0,
    ) -> pd.Series:
        """Detrended Price Oscillator"""
        validation = validate_series_params(data, length)
        if validation is not None:
            return validation

        result = ta.dpo(
            close=data,
            length=length,
            centered=centered,
            offset=offset,
        )
        if result is None or (hasattr(result, "empty") and result.empty):
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def vortex(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
        drift: int = 1,
        offset: int = 0,
    ) -> Tuple[pd.Series, pd.Series]:
        """Vortex Indicator"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, length
        )
        if validation is not None:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series

        if drift <= 0:
            raise ValueError(f"drift must be positive: {drift}")

        result = ta.vortex(
            high=high,
            low=low,
            close=close,
            length=length,
            drift=drift,
            offset=offset,
        )

        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series

        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    @handle_pandas_ta_errors
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
        lensig: int = 14,
        scalar: float = 100.0,
        mamode: str = "rma",
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ADX: returns (adx, dmp, dmn)"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, length
        )

        def nan_result() -> Tuple[pd.Series, pd.Series, pd.Series]:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series.copy(), nan_series.copy()

        if validation is not None:
            return nan_result()

        result = ta.adx(
            high=high,
            low=low,
            close=close,
            length=length,
            lensig=lensig,
            scalar=scalar,
            mamode=mamode,
        )

        if result is None or result.empty:
            return nan_result()

        # カラム名: ADX_{length}, DMP_{length}, DMN_{length}
        try:
            return (
                result[f"ADX_{length}"],
                result[f"DMP_{length}"],
                result[f"DMN_{length}"],
            )
        except (KeyError, Exception):
            return nan_result()

    @staticmethod
    @handle_pandas_ta_errors
    def aroon(
        high: pd.Series,
        low: pd.Series,
        length: int = 14,
        scalar: float = 100.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Aroon: returns (aroon_up, aroon_down, aroon_osc)"""
        validation = validate_multi_series_params({"high": high, "low": low}, length)

        def nan_result() -> Tuple[pd.Series, pd.Series, pd.Series]:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series.copy(), nan_series.copy()

        if validation is not None:
            return nan_result()

        result = ta.aroon(high=high, low=low, length=length, scalar=scalar)

        if result is None or result.empty:
            return nan_result()

        # カラム名: AROONU_{length}, AROOND_{length}, AROONOSC_{length}
        try:
            return (
                result[f"AROONU_{length}"],
                result[f"AROOND_{length}"],
                result[f"AROONOSC_{length}"],
            )
        except (KeyError, Exception):
            return nan_result()

    @staticmethod
    @handle_pandas_ta_errors
    def chop(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
        atr_length: int = 1,
        scalar: float = 100.0,
        drift: int = 1,
    ) -> pd.Series:
        """Choppiness Index"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, length
        )
        if validation is not None:
            return validation

        result = ta.chop(
            high=high,
            low=low,
            close=close,
            length=length,
            atr_length=atr_length,
            scalar=scalar,
            drift=drift,
        )

        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def vhf(
        data: pd.Series,
        length: int = 28,
        scalar: float = 100.0,
        drift: int = 1,
        offset: int = 0,
    ) -> pd.Series:
        """Vertical Horizontal Filter"""
        # VHF requires sufficient data length
        min_length = length * 2
        validation = validate_series_params(data, length, min_data_length=min_length)
        if validation is not None:
            return validation

        result = ta.vhf(
            close=data,
            length=length,
            scalar=scalar,
            drift=drift,
            offset=offset,
        )

        if result is None or (hasattr(result, "isna") and result.isna().all()):
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def cksp(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        p: int = 10,
        x: float = 1.0,
        q: int = 9,
    ) -> Tuple[pd.Series, pd.Series]:
        """Chande Kroll Stop"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, p
        )
        if validation is not None:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series

        result = ta.cksp(high=high, low=low, close=close, p=p, x=x, q=q)
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series

        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    @handle_pandas_ta_errors
    def decay(
        close: pd.Series,
        length: int = 5,
        mode: str = "linear",
    ) -> pd.Series:
        """Decay"""
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        result = ta.decay(close=close, length=length, mode=mode)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result.fillna(0)

    @staticmethod
    @handle_pandas_ta_errors
    def qstick(
        open_: pd.Series,
        close: pd.Series,
        length: int = 8,
    ) -> pd.Series:
        """QStick"""
        validation = validate_multi_series_params(
            {"open_": open_, "close": close}, length
        )
        if validation is not None:
            return validation

        result = ta.qstick(open_=open_, close=close, length=length)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result.fillna(0)

    @staticmethod
    @handle_pandas_ta_errors
    def ttm_trend(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 6,
    ) -> pd.Series:
        """TTM Trend"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, length
        )
        if validation is not None:
            return validation

        result = ta.ttm_trend(high=high, low=low, close=close, length=length)
        if result is None or (hasattr(result, "empty") and result.empty):
            return pd.Series(np.full(len(close), np.nan), index=close.index)

        if isinstance(result, pd.DataFrame):
            return result.iloc[:, 0]
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def decreasing(
        close: pd.Series,
        length: int = 1,
        strict: bool = False,
        as_int: bool = True,
    ) -> pd.Series:
        """Decreasing"""
        # Data validation might be minimal for simple comparisons
        if close is None:
            return pd.Series([], dtype=float)

        result = ta.decreasing(close=close, length=length, strict=strict, asint=as_int)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def increasing(
        close: pd.Series,
        length: int = 1,
        strict: bool = False,
        as_int: bool = True,
    ) -> pd.Series:
        """Increasing"""
        if close is None:
            return pd.Series([], dtype=float)

        result = ta.increasing(close=close, length=length, strict=strict, asint=as_int)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def long_run(
        fast: pd.Series,
        slow: pd.Series,
        length: int = 2,
    ) -> pd.Series:
        """Long Run"""
        # Requires len(fast) >= length
        validation = validate_multi_series_params({"fast": fast, "slow": slow}, length)
        if validation is not None:
            return validation

        result = ta.long_run(fast=fast, slow=slow, length=length)
        if result is None:
            return pd.Series(np.full(len(fast), np.nan), index=fast.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def short_run(
        fast: pd.Series,
        slow: pd.Series,
        length: int = 2,
    ) -> pd.Series:
        """Short Run"""
        validation = validate_multi_series_params({"fast": fast, "slow": slow}, length)
        if validation is not None:
            return validation

        result = ta.short_run(fast=fast, slow=slow, length=length)
        if result is None:
            return pd.Series(np.full(len(fast), np.nan), index=fast.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def tsignals(
        trend: pd.Series,
        signal: pd.Series = None,  # Some use cases imply creating signal internally or passing trend as signal
        trend_reset: int = 0,
        trade_offset: int = 0,
        trend_offset: int = 0,  # Depending on pandas-ta version
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Trend Signals"""
        # This function typically takes a 'trend' indicator output (like 1, -1) and generates signals
        if trend is None:
            nan_series = pd.Series([], dtype=float)
            return nan_series, nan_series, nan_series, nan_series

        # signal argument is often just the trend itself if not separated?
        # Actually ta.tsignals needs `trend` series.
        result = ta.tsignals(
            trend=trend,
            trend_reset=trend_reset,
            trade_offset=trade_offset,
            trend_offset=trend_offset,
        )
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(trend), np.nan), index=trend.index)
            # Returns TS_Trends, TS_Trades, TS_Entries, TS_Exits usually
            return nan_series, nan_series, nan_series, nan_series

        # Ensure we have enough columns
        if result.shape[1] < 4:
            # If not enough columns, pad with NaNs or handle gracefully
            # Creating a list of series from columns available
            columns = [result.iloc[:, i] for i in range(result.shape[1])]
            # Pad with NaNs
            nan_series = pd.Series(np.full(len(trend), np.nan), index=trend.index)
            while len(columns) < 4:
                columns.append(nan_series)
            return tuple(columns[:4])

        return (
            result.iloc[:, 0],
            result.iloc[:, 1],
            result.iloc[:, 2],
            result.iloc[:, 3],
        )

    @staticmethod
    @handle_pandas_ta_errors
    def xsignals(
        signal: pd.Series,
        xa: float = 80,
        xb: float = 20,
        above: bool = True,
        long: bool = True,
        str_tag: str = "XA",
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Cross Signals"""
        if signal is None:
            nan_series = pd.Series([], dtype=float)
            return nan_series, nan_series, nan_series, nan_series

        result = ta.xsignals(
            signal=signal, xa=xa, xb=xb, above=above, long=long, str=str_tag
        )
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(signal), np.nan), index=signal.index)
            # Returns TS_Trends, TS_Trades, TS_Entries, TS_Exits typically or similar struct
            return nan_series, nan_series, nan_series, nan_series

        # Ensure we have enough columns
        if result.shape[1] < 4:
            columns = [result.iloc[:, i] for i in range(result.shape[1])]
            nan_series = pd.Series(np.full(len(signal), np.nan), index=signal.index)
            while len(columns) < 4:
                columns.append(nan_series)
            return tuple(columns[:4])

        return (
            result.iloc[:, 0],
            result.iloc[:, 1],
            result.iloc[:, 2],
            result.iloc[:, 3],
        )

    @staticmethod
    @handle_pandas_ta_errors
    def linregslope(
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """Linear Regression Slope"""
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        # pandas-ta uses 'slope'
        result = ta.slope(close=close, length=length)

        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def ichimoku(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tenkan: int = 9,
        kijun: int = 26,
        senkou: int = 52,
        include_chikou: bool = True,
        offset: int = 26,
    ) -> pd.DataFrame:
        """Ichimoku Kinko Hyo"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, senkou
        )
        if validation is not None:
            return pd.DataFrame()

        try:
            result, span = ta.ichimoku(
                high=high,
                low=low,
                close=close,
                tenkan=tenkan,
                kijun=kijun,
                senkou=senkou,
                include_chikou=include_chikou,
                offset=offset,
            )
        except Exception:
            return pd.DataFrame()

        if result is None or result.empty:
            return pd.DataFrame()

        rename_map = {
            f"ITS_{tenkan}_{kijun}_{senkou}": "tenkan_sen",
            f"IKS_{tenkan}_{kijun}_{senkou}": "kijun_sen",
            f"ISA_{tenkan}_{kijun}_{senkou}": "senkou_span_a",
            f"ISB_{tenkan}_{kijun}_{senkou}": "senkou_span_b",
            f"ICS_{tenkan}_{kijun}_{senkou}": "chikou_span",
        }

        result = result.rename(columns=rename_map)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def sma(
        close: pd.Series,
        length: int = 10,
    ) -> pd.Series:
        """Simple Moving Average"""
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        result = ta.sma(close=close, length=length)

        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

"""
トレンド系テクニカル指標 (Trend Indicators)

pandas-ta の trend カテゴリに対応。
トレンドの方向性と強さを分析する指標群。

登録してあるテクニカルの一覧:
- SAR (Parabolic SAR)
- AMAT (Archer Moving Averages Trends)
- DPO (Detrended Price Oscillator)
- Vortex
- ADX (Average Directional Index)
- Aroon
- CHOP (Choppiness Index)
- VHF (Vertical Horizontal Filter)
- CKSP (Chande Kroll Stop)
- Decay
- QStick
- TTM Trend
- Decreasing
- Increasing
- Long Run
- Short Run
- Linear Regression Slope
- Ichimoku Kinko Hyo
- SMA (Simple Moving Average)
"""

import logging
from typing import Any, Tuple, cast

import numpy as np
import pandas as pd
import pandas_ta_classic as ta  # type: ignore
from numba import njit

from ...data_validation import (
    create_nan_series_bundle,
    create_nan_series_like,
    handle_pandas_ta_errors,
    run_multi_series_indicator,
    run_series_indicator,
)

logger = logging.getLogger(__name__)


class TrendIndicators:
    """
    トレンド系指標クラス

    Parabolic SAR, ADX などのトレンド方向性を分析するテクニカル指標を提供。
    トレンドの方向性と強さの分析に使用します。
    """

    @staticmethod
    @njit(cache=True)
    def _sar_loop(high_arr, low_arr, af, max_af):
        """
        パラボリック SAR を計算する Numba 加速ループ。
        """
        n = len(high_arr)
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
                # sar[i] = prev_sar + af_val * (ep - prev_sar)
                val = prev_sar + af_val * (ep - prev_sar)
                # SARは直近2期間の安値を超えてはならない
                m1 = low_arr[i - 1]
                m2 = low_arr[i - 2]
                if val > m1:
                    val = m1
                if val > m2:
                    val = m2
                sar[i] = val

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
                # sar[i] = prev_sar + af_val * (ep - prev_sar)
                val = prev_sar + af_val * (ep - prev_sar)
                # SARは直近2期間の高値を超えてはならない
                m1 = high_arr[i - 1]
                m2 = high_arr[i - 2]
                if val < m1:
                    val = m1
                if val < m2:
                    val = m2
                sar[i] = val

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

        return sar

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

        def compute() -> pd.Series:
            """
            パラボリック SAR の計算ロジック。
            """
            n = len(high)
            if n < 2:
                return create_nan_series_like(high)

            high_arr = high.values.astype(np.float64)
            low_arr = low.values.astype(np.float64)

            sar_res = TrendIndicators._sar_loop(high_arr, low_arr, af, max_af)

            return pd.Series(sar_res, index=high.index).replace(0, np.nan)

        return cast(
            pd.Series,
            run_multi_series_indicator({"high": high, "low": low}, None, compute),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def amat(
        data: pd.Series, fast: int = 3, slow: int = 30, signal: int = 10
    ) -> pd.Series:
        """Archer Moving Averages Trends"""
        # AMAT特有のデータ検証
        min_length = max(fast, slow, signal) + 10
        result: Any = run_series_indicator(
            data,
            None,
            lambda: ta.amat(data, fast=fast, slow=slow, signal=signal),
            min_data_length=min_length,
        )
        # AMAT returns DataFrame, get the main series
        if hasattr(result, "iloc"):
            return result.iloc[:, 0] if len(result.shape) > 1 else result
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def dpo(
        data: pd.Series,
        length: int = 20,
        centered: bool = False,
        offset: int = 0,
    ) -> pd.Series:
        """Detrended Price Oscillator"""
        return cast(
            pd.Series,
            run_series_indicator(
                data,
                length,
                lambda: ta.dpo(
                    close=data,
                    length=length,
                    centered=centered,
                    offset=offset,
                ),
            ),
        )

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
        if drift <= 0:
            raise ValueError(f"drift must be positive: {drift}")

        result: Any = run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            length,
            lambda: ta.vortex(
                high=high,
                low=low,
                close=close,
                length=length,
                drift=drift,
                offset=offset,
            ),
            fallback_factory=lambda: cast(
                tuple[pd.Series, pd.Series], create_nan_series_bundle(high, 2)
            ),
        )
        if isinstance(result, tuple):
            return cast(tuple[pd.Series, pd.Series], result)

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

        def nan_result() -> Tuple[pd.Series, pd.Series, pd.Series]:
            return cast(
                Tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(high, 3),
            )

        result: Any = run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            length,
            lambda: ta.adx(
                high=high,
                low=low,
                close=close,
                length=length,
                lensig=lensig,
                scalar=scalar,
                mamode=mamode,
            ),
            fallback_factory=nan_result,
        )

        if isinstance(result, tuple):
            return cast(tuple[pd.Series, pd.Series, pd.Series], result)

        if hasattr(result, "empty") and getattr(result, "empty", False):
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

        def nan_result() -> Tuple[pd.Series, pd.Series, pd.Series]:
            return cast(
                Tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(high, 3),
            )

        result: Any = run_multi_series_indicator(
            {"high": high, "low": low},
            length,
            lambda: ta.aroon(high=high, low=low, length=length, scalar=scalar),
            fallback_factory=nan_result,
        )

        if isinstance(result, tuple):
            return cast(tuple[pd.Series, pd.Series, pd.Series], result)

        if hasattr(result, "empty") and getattr(result, "empty", False):
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
        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"high": high, "low": low, "close": close},
                length,
                lambda: ta.chop(
                    high=high,
                    low=low,
                    close=close,
                    length=length,
                    atr_length=atr_length,
                    scalar=scalar,
                    drift=drift,
                ),
            ),
        )

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
        return cast(
            pd.Series,
            run_series_indicator(
                data,
                length,
                lambda: ta.vhf(
                    close=data,
                    length=length,
                    scalar=scalar,
                    drift=drift,
                    offset=offset,
                ),
                min_data_length=min_length,
            ),
        )

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
        result: Any = run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            p,
            lambda: ta.cksp(high=high, low=low, close=close, p=p, x=x, q=q),
            fallback_factory=lambda: cast(
                Tuple[pd.Series, pd.Series], create_nan_series_bundle(close, 2)
            ),
        )

        if isinstance(result, tuple):
            return cast(Tuple[pd.Series, pd.Series], result)

        return cast(Tuple[pd.Series, pd.Series], (result.iloc[:, 0], result.iloc[:, 1]))

    @staticmethod
    @handle_pandas_ta_errors
    def decay(
        close: pd.Series,
        length: int = 5,
        mode: str = "linear",
    ) -> pd.Series:
        """Decay"""
        return cast(
            pd.Series,
            run_series_indicator(
                close, length, lambda: ta.decay(close=close, length=length, mode=mode)
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def qstick(
        open_: pd.Series,
        close: pd.Series,
        length: int = 8,
    ) -> pd.Series:
        """QStick"""
        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"open_": open_, "close": close},
                length,
                lambda: ta.qstick(open_=open_, close=close, length=length),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def ttm_trend(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 6,
    ) -> pd.Series:
        """TTM Trend"""
        result: Any = run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            length,
            lambda: ta.ttm_trend(high=high, low=low, close=close, length=length),
        )

        if hasattr(result, "empty") and getattr(result, "empty", False):
            return create_nan_series_like(close)

        if isinstance(result, pd.DataFrame):
            return result.iloc[:, 0]
        return cast(pd.Series, result)

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
            return create_nan_series_like(close)
        return cast(pd.Series, result)

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
            return create_nan_series_like(close)
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
        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"fast": fast, "slow": slow},
                length,
                lambda: ta.long_run(fast=fast, slow=slow, length=length),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def short_run(
        fast: pd.Series,
        slow: pd.Series,
        length: int = 2,
    ) -> pd.Series:
        """Short Run"""
        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"fast": fast, "slow": slow},
                length,
                lambda: ta.short_run(fast=fast, slow=slow, length=length),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def linregslope(
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """Linear Regression Slope"""
        return cast(
            pd.Series,
            run_series_indicator(
                close, length, lambda: ta.slope(close=close, length=length)
            ),
        )

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

        def nan_result() -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "tenkan_sen": np.full(len(high), np.nan),
                    "kijun_sen": np.full(len(high), np.nan),
                    "senkou_span_a": np.full(len(high), np.nan),
                    "senkou_span_b": np.full(len(high), np.nan),
                    "chikou_span": np.full(len(high), np.nan),
                },
                index=high.index,
            )

        try:
            result: Any = run_multi_series_indicator(
                {"high": high, "low": low, "close": close},
                senkou,
                lambda: ta.ichimoku(
                    high=high,
                    low=low,
                    close=close,
                    tenkan=tenkan,
                    kijun=kijun,
                    senkou=senkou,
                    include_chikou=include_chikou,
                    offset=offset,
                ),
                fallback_factory=nan_result,
            )

            if isinstance(result, tuple):
                result = result[0]

            if isinstance(result, pd.Series):
                return nan_result()

            if result is None or (
                hasattr(result, "empty") and getattr(result, "empty", False)
            ):
                return nan_result()

            rename_map = {
                f"ITS_{tenkan}_{kijun}_{senkou}": "tenkan_sen",
                f"IKS_{tenkan}_{kijun}_{senkou}": "kijun_sen",
                f"ISA_{tenkan}_{kijun}_{senkou}": "senkou_span_a",
                f"ISB_{tenkan}_{kijun}_{senkou}": "senkou_span_b",
                f"ICS_{tenkan}_{kijun}_{senkou}": "chikou_span",
            }

            result = result.rename(columns=rename_map)
            return result
        except Exception:
            return nan_result()

    @staticmethod
    @handle_pandas_ta_errors
    def sma(
        close: pd.Series,
        length: int = 10,
    ) -> pd.Series:
        """Simple Moving Average"""
        return cast(
            pd.Series,
            run_series_indicator(
                close, length, lambda: ta.sma(close=close, length=length)
            ),
        )

"""
オーバーレイ系テクニカル指標 (Overlap Indicators)

pandas-ta の overlap カテゴリに対応。
価格チャートに重ねて表示する指標群。

登録してあるテクニカルの一覧:
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- WMA (Weighted Moving Average)
- TRIMA (Triangular Moving Average)
- ZLMA (Zero Lag Moving Average)
- ALMA (Arnaud Legoux Moving Average)
- DEMA (Double Exponential Moving Average)
- TEMA (Triple Exponential Moving Average)
- T3 (Tillson's T3 Moving Average)
- KAMA (Kaufman's Adaptive Moving Average)
- HMA (Hull Moving Average)
- VWMA (Volume Weighted Moving Average)
- LINREG (Linear Regression)
- LINREGSLOPE (Linear Regression Slope)
- RMA (Wilder's Moving Average)
- Supertrend
- Ichimoku Cloud (一目均衡表)
- HILO (Gann HiLo)
- HL2 (High-Low Average)
- HLC3 (HLC Average)
- OHLC4 (OHLC Average)
- Midpoint
- Midprice
- VIDYA
- WCP (Weighted Close Price)
- MCGD (McGinley Dynamic)
- JMA (Jurik Moving Average)
- FWMA (Fibonacci Weighted Moving Average)
- PWMA (Pascal Weighted Moving Average)
- SinWMA (Sine Weighted Moving Average)
- SSF (Ehlers Super Smoother Filter)
- SWMA (Symmetric Weighted Moving Average)
"""

from typing import Dict, Tuple

import pandas as pd
import pandas_ta_classic as ta

from ..data_validation import (
    create_nan_series_bundle,
    create_nan_series_like,
    create_nan_series_map,
    handle_pandas_ta_errors,
    run_multi_series_indicator,
    run_series_indicator,
    validate_multi_series_params,
)


class OverlapIndicators:
    """
    オーバーレイ系指標クラス

    移動平均線、Supertrend などの価格に重ねて表示するテクニカル指標を提供。
    """

    @staticmethod
    @handle_pandas_ta_errors
    def sma(data: pd.Series, length: int) -> pd.Series:
        """単純移動平均"""
        return run_series_indicator(data, length, lambda: ta.sma(data, length=length))

    @staticmethod
    @handle_pandas_ta_errors
    def ema(data: pd.Series, length: int) -> pd.Series:
        """指数移動平均"""
        return run_series_indicator(
            data,
            length,
            lambda: ta.ema(data, length=length, adjust=False, sma=True),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def wma(
        data: pd.Series | None = None,
        length: int = 14,
        close: pd.Series | None = None,
    ) -> pd.Series:
        """加重移動平均"""
        # dataが提供されない場合はcloseを使用
        if data is None and close is not None:
            data = close
        elif data is None:
            raise ValueError("Either 'data' or 'close' must be provided")

        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        return run_series_indicator(data, length, lambda: ta.wma(data, length=length))

    @staticmethod
    @handle_pandas_ta_errors
    def trima(
        data: pd.Series, length: int = 10, talib: bool | None = None
    ) -> pd.Series:
        """三角移動平均"""
        return run_series_indicator(
            data,
            length,
            lambda: ta.trima(data, length=length, talib=talib),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def zlma(
        data: pd.Series,
        length: int = 10,
        mamode: str = "ema",
        offset: int = 0,
    ) -> pd.Series:
        """Zero Lag移動平均"""
        return run_series_indicator(
            data,
            length,
            lambda: ta.zlma(data, length=length, mamode=mamode, offset=offset),
        )

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
        if sigma <= 0:
            raise ValueError(f"sigma must be positive: {sigma}")
        if not 0.0 <= distribution_offset <= 1.0:
            raise ValueError(
                f"distribution_offset must be between 0.0 and 1.0: {distribution_offset}"
            )

        return run_series_indicator(
            data,
            length,
            lambda: ta.alma(
                data,
                length=length,
                sigma=sigma,
                distribution_offset=distribution_offset,
                offset=offset,
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def dema(data: pd.Series, length: int) -> pd.Series:
        """二重指数移動平均"""
        return run_series_indicator(
            data,
            length,
            lambda: ta.dema(data, length=length),
            min_data_length=length * 2,
        )

    @staticmethod
    @handle_pandas_ta_errors
    def tema(data: pd.Series, length: int) -> pd.Series:
        """三重指数移動平均"""
        return run_series_indicator(
            data,
            length,
            lambda: ta.tema(data, window=length),
            min_data_length=length * 3,
        )

    @staticmethod
    @handle_pandas_ta_errors
    def t3(data: pd.Series, length: int, a: float = 0.7) -> pd.Series:
        """T3移動平均"""
        return run_series_indicator(
            data,
            length,
            lambda: ta.t3(data, window=length, a=a),
            min_data_length=length * 6,
        )

    @staticmethod
    @handle_pandas_ta_errors
    def kama(data: pd.Series, length: int = 30) -> pd.Series:
        """カウフマン適応移動平均"""
        return run_series_indicator(data, length, lambda: ta.kama(data, window=length))

    @staticmethod
    @handle_pandas_ta_errors
    def hma(data: pd.Series, length: int = 20) -> pd.Series:
        """Hull移動平均"""
        return run_series_indicator(data, length, lambda: ta.hma(data, length=length))

    @staticmethod
    @handle_pandas_ta_errors
    def vwma(
        close: pd.Series,
        volume: pd.Series,
        length: int = 20,
    ) -> pd.Series:
        """出来高加重移動平均"""
        return run_multi_series_indicator(
            {"close": close, "volume": volume},
            length,
            lambda: ta.vwma(close=close, volume=volume, length=length),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def linreg(
        data: pd.Series,
        length: int = 14,
        scalar: float = 1.0,
        intercept: bool = False,
    ) -> pd.Series:
        """線形回帰 (pandas-ta ベクトル化版)"""
        result = run_series_indicator(
            data,
            length,
            lambda: ta.linreg(data, length=length, offset=0),
            min_data_length=length,
        )
        # interceptが必要な場合（あまり一般的ではないが元のコードにあるためサポート）
        if intercept:
            if isinstance(result, pd.Series) and result.isna().all():
                return result * scalar

            slope = run_series_indicator(
                data,
                length,
                lambda: ta.linreg(data, length=length, slope=True),
                min_data_length=length,
            )
            if isinstance(slope, pd.Series) and slope.isna().all():
                return create_nan_series_like(data)

            intercept_val = result - slope * (length - 1)
            return intercept_val * scalar

        return result * scalar

    @staticmethod
    @handle_pandas_ta_errors
    def linregslope(
        data: pd.Series, length: int = 14, scalar: float = 1.0
    ) -> pd.Series:
        """線形回帰スロープ (pandas-ta ベクトル化版)"""
        result = run_series_indicator(
            data,
            length,
            lambda: ta.linreg(data, length=length, slope=True),
            min_data_length=length,
        )
        return result * scalar

    @staticmethod
    @handle_pandas_ta_errors
    def rma(data: pd.Series, length: int = 10) -> pd.Series:
        """Wilde's Moving Average"""
        return run_series_indicator(data, length, lambda: ta.rma(data, length=length))

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
        # 'factor' を 'multiplier' のエイリアスとしてサポート
        if "factor" in kwargs:
            multiplier = kwargs["factor"]

        result = run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            period,
            lambda: ta.supertrend(
                high=high, low=low, close=close, length=period, multiplier=multiplier
            ),
            fallback_factory=lambda: create_nan_series_bundle(high, 3),
        )

        if isinstance(result, tuple):
            return result

        # カラム名: SUPERTl_{length}_{multiplier}, SUPERTs_{length}_{multiplier}, SUPERTd_{length}_{multiplier}
        try:
            # 浮動小数点形式 (例: 3.0)
            return (
                result[f"SUPERTl_{period}_{float(multiplier)}"],
                result[f"SUPERTs_{period}_{float(multiplier)}"],
                result[f"SUPERTd_{period}_{float(multiplier)}"],
            )
        except KeyError:
            try:
                # 整数形式 (例: 3)
                return (
                    result[f"SUPERTl_{period}_{int(multiplier)}"],
                    result[f"SUPERTs_{period}_{int(multiplier)}"],
                    result[f"SUPERTd_{period}_{int(multiplier)}"],
                )
            except (KeyError, Exception):
                return create_nan_series_bundle(high, 3)  # type: ignore[return-value]

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

        # 長さ不一致は既存仕様どおり ValueError にする
        validate_multi_series_params(
            {"high": high, "low": low, "close": close},
            max_period,
        )

        def nan_result() -> Dict[str, pd.Series]:
            """
            一目均衡表の計算に失敗した場合、またはデータ不足の場合に NaN の結果を返します。
            """
            return create_nan_series_map(
                high,
                [
                    "tenkan_sen",
                    "kijun_sen",
                    "senkou_span_a",
                    "senkou_span_b",
                    "chikou_span",
                ],
            )

        try:
            result = run_multi_series_indicator(
                {"high": high, "low": low, "close": close},
                max_period,
                lambda: ta.ichimoku(
                    high=high,
                    low=low,
                    close=close,
                    tenkan=tenkan_period,
                    kijun=kijun_period,
                    senkou=senkou_span_b_period,
                ),
                fallback_factory=nan_result,
            )

            if isinstance(result, dict):
                return result

            if isinstance(result, tuple):
                result = result[0]

            if result is None or (hasattr(result, "empty") and result.empty):
                return nan_result()

            # カラム名パターン: ITS_{tenkan}, IKS_{kijun}, ISA_{tenkan}, ISB_{kijun}, ICS_{kijun}
            return {
                "tenkan_sen": result[f"ITS_{tenkan_period}"],
                "kijun_sen": result[f"IKS_{kijun_period}"],
                "senkou_span_a": result[f"ISA_{tenkan_period}"],
                "senkou_span_b": result[f"ISB_{kijun_period}"],
                "chikou_span": result[f"ICS_{kijun_period}"],
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
        result = run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            max(high_length, low_length),
            lambda: ta.hilo(
                high=high,
                low=low,
                close=close,
                high_length=high_length,
                low_length=low_length,
                mamode=mamode,
                offset=offset,
            ),
            fallback_factory=lambda: create_nan_series_bundle(close, 3),
        )

        if isinstance(result, tuple):
            return result

        # Returns HILO, HILOl, HILOs
        return result.iloc[:, 0], result.iloc[:, 1], result.iloc[:, 2]

    @staticmethod
    @handle_pandas_ta_errors
    def hl2(high: pd.Series, low: pd.Series) -> pd.Series:
        """High-Low Average"""
        return run_multi_series_indicator(
            {"high": high, "low": low}, None, lambda: ta.hl2(high=high, low=low)
        )

    @staticmethod
    @handle_pandas_ta_errors
    def hlc3(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """HLC Average"""
        return run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            None,
            lambda: ta.hlc3(high=high, low=low, close=close),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def ohlc4(
        open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """OHLC Average"""
        return run_multi_series_indicator(
            {"open_": open_, "high": high, "low": low, "close": close},
            None,
            lambda: ta.ohlc4(open_=open_, high=high, low=low, close=close),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def midpoint(close: pd.Series, length: int = 2) -> pd.Series:
        """Midpoint"""
        return run_series_indicator(
            close, length, lambda: ta.midpoint(close=close, length=length)
        )

    @staticmethod
    @handle_pandas_ta_errors
    def midprice(high: pd.Series, low: pd.Series, length: int = 2) -> pd.Series:
        """Midprice"""
        return run_multi_series_indicator(
            {"high": high, "low": low},
            length,
            lambda: ta.midprice(high=high, low=low, length=length),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def vidya(
        close: pd.Series, length: int = 14, drift: int = 1, offset: int = 0
    ) -> pd.Series:
        """VIDYA"""
        return run_series_indicator(
            close,
            length,
            lambda: ta.vidya(close=close, length=length, drift=drift, offset=offset),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def wcp(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Weighted Close Price"""
        return run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            None,
            lambda: ta.wcp(high=high, low=low, close=close),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def mcgd(close: pd.Series, length: int = 10, offset: int = 0) -> pd.Series:
        """McGinley Dynamic"""
        return run_series_indicator(
            close, length, lambda: ta.mcgd(close=close, length=length, offset=offset)
        )

    @staticmethod
    @handle_pandas_ta_errors
    def jma(
        close: pd.Series, length: int = 7, phase: int = 50, offset: int = 0
    ) -> pd.Series:
        """Jurik Moving Average"""
        return run_series_indicator(
            close,
            length,
            lambda: ta.jma(close=close, length=length, phase=phase, offset=offset),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def fwma(close: pd.Series, length: int = 10, asc: bool = True) -> pd.Series:
        """Fibonacci Weighted Moving Average"""
        return run_series_indicator(
            close, length, lambda: ta.fwma(close=close, length=length, asc=asc)
        )

    @staticmethod
    @handle_pandas_ta_errors
    def pwma(close: pd.Series, length: int = 10, asc: bool = True) -> pd.Series:
        """Pascal Weighted Moving Average"""
        return run_series_indicator(
            close, length, lambda: ta.pwma(close=close, length=length, asc=asc)
        )

    @staticmethod
    @handle_pandas_ta_errors
    def sinwma(close: pd.Series, length: int = 14) -> pd.Series:
        """Sine Weighted Moving Average"""
        return run_series_indicator(
            close, length, lambda: ta.sinwma(close=close, length=length)
        )

    @staticmethod
    @handle_pandas_ta_errors
    def ssf(close: pd.Series, length: int = 10, poles: int = 2) -> pd.Series:
        """Ehlers Super Smoother Filter"""
        return run_series_indicator(
            close, length, lambda: ta.ssf(close=close, length=length, poles=poles)
        )

    @staticmethod
    @handle_pandas_ta_errors
    def swma(close: pd.Series, length: int = 10) -> pd.Series:
        """Symmetric Weighted Moving Average"""
        return run_series_indicator(
            close, length, lambda: ta.swma(close=close, length=length)
        )

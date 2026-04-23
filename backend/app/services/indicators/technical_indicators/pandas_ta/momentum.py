"""
モメンタム系テクニカル指標 (Momentum Indicators)

pandas-ta の momentum カテゴリに対応。
価格の勢いと転換点の検出に使用する指標群。

登録してあるテクニカルの一覧:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- PPO (Percentage Price Oscillator)
- TRIX (Triple Exponential Average)
- DM (Directional Movement)
- ER (Efficiency Ratio)
- LRSI (Laguerre RSI)
- PO (Projection Oscillator)
- TRIXH (TRIX Histogram)
- VWMACD (Volume Weighted MACD)
- Stochastic Oscillator
- Stochastic RSI
- Williams %R
- CCI (Commodity Channel Index)
- CMO (Chande Momentum Oscillator)
- STC (Schaff Trend Cycle)
- Fisher Transform
- KST (Know Sure Thing)
- ROC (Rate of Change)
- Momentum
- QQE (Qualitative Quantitative Estimation)
- Squeeze Pro
- CTI (Correlation Trend Indicator)
- APO (Absolute Price Oscillator)
- TSI (True Strength Index)
- PGO (Pretty Good Oscillator)
- PSL (Psychological Line)
- Squeeze
- UO (Ultimate Oscillator)
- AO (Awesome Oscillator)
- BOP (Balance of Power)
- CG (Center of Gravity)
- Coppock Curve
- Bias
- Efficiency Ratio (Kaufman)
- BRAR (Brayer)
- CFO (Chande Forecast Oscillator)
- ERI (Elder Ray Index)
- Inertia
- KDJ
- RSX
- RVGI (Relative Vigor Index)
- Slope
- SMI Ergodic
- TD Sequential
"""

from typing import Any, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

from ...data_validation import (
    create_nan_series_bundle,
    create_nan_series_like,
    handle_pandas_ta_errors,
    normalize_non_finite,
    run_multi_series_indicator,
    run_series_indicator,
)


def _create_nan_array_bundle(
    length: int, count: int
) -> tuple[np.ndarray, ...]:
    """同じ長さの NaN 配列を複数作る。"""
    base = np.full(length, np.nan)
    return tuple(base.copy() for _ in range(count))


class MomentumIndicators:
    """
    モメンタム系指標クラス

    RSI, MACD, ストキャスティクスなどのモメンタム系テクニカル指標を提供。
    価格の勢いと転換点の検出に使用します。
    """

    @staticmethod
    @handle_pandas_ta_errors
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """相対力指数"""

        def compute_rsi() -> pd.Series:
            result: Any = ta.rsi(data, window=period)
            if result is None:
                return create_nan_series_like(data)

            result = cast(pd.Series, result)

            # 一定価格など、変動がない場合のRSIは50とする
            # pandas_ta_classicはRS=0/0のケースで0を返すことがある
            valid_values = result.dropna()
            if not valid_values.empty and bool(valid_values.eq(0.0).all()):
                if float(data.std()) == 0.0:
                    return pd.Series(50.0, index=result.index)

            return result

        return cast(
            pd.Series,
            run_series_indicator(data, period, compute_rsi),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def macd(
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)

        MACDライン = EMA(fast) - EMA(slow)
        シグナルライン = EMA(MACDライン, signal)
        ヒストグラム = MACDライン - シグナルライン

        pandas_taのmacd関数は内部で独自のEMA初期化を行うため、
         standaloneのEMA計算と結果がずれることがある。
        一貫性のため、明示的にEMAを使用して計算する。
        """
        min_length = slow + signal  # 十分なウォームアップ期間

        def compute_macd() -> Tuple[pd.Series, pd.Series, pd.Series]:
            from .overlap import OverlapIndicators

            ema_fast = OverlapIndicators.ema(data, length=fast)
            ema_slow = OverlapIndicators.ema(data, length=slow)
            macd_line = ema_fast - ema_slow
            signal_line = OverlapIndicators.ema(macd_line, length=signal)
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram

        # Use run_series_indicator with fallback for the full computation
        result: Any = run_series_indicator(
            data,
            min_length,
            compute_macd,
            fallback_factory=lambda: cast(
                tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(data, 3),
            ),
        )

        if isinstance(result, tuple):
            return cast(tuple[pd.Series, pd.Series, pd.Series], result)

        # Fallback: pandas_taの直接呼び出し（互換性のため）
        raw_result: Any = run_series_indicator(
            data,
            None,
            lambda: ta.macd(data, fast=fast, slow=slow, signal=signal),
            fallback_factory=lambda: cast(
                tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(data, 3),
            ),
        )

        if isinstance(raw_result, tuple):
            return cast(tuple[pd.Series, pd.Series, pd.Series], raw_result)

        return (
            raw_result.iloc[:, 0],
            raw_result.iloc[:, 1],
            raw_result.iloc[:, 2],
        )

    @staticmethod
    @handle_pandas_ta_errors
    def ppo(
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Percentage Price Oscillator"""
        result: Any = run_series_indicator(
            data,
            None,
            lambda: ta.ppo(data, fast=fast, slow=slow, signal=signal),
            fallback_factory=lambda: cast(
                tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(data, 3),
            ),
        )

        if isinstance(result, tuple):
            return cast(tuple[pd.Series, pd.Series, pd.Series], result)

        return (
            result.iloc[:, 0],
            result.iloc[:, 1],
            result.iloc[:, 2],
        )

    @staticmethod
    @handle_pandas_ta_errors
    def trix(
        data: pd.Series,
        length: int = 15,
        signal: int = 9,
        scalar: Optional[float] = None,
        drift: int = 1,
        offset: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """TRIX (Triple Exponential Average)"""
        result: Any = run_series_indicator(
            data,
            length,
            lambda: ta.trix(
                close=data,
                length=length,
                signal=signal,
                scalar=scalar,
                drift=drift,
                offset=offset,
            ),
            fallback_factory=lambda: _create_nan_array_bundle(len(data), 3),
        )

        if isinstance(result, tuple):
            return result

        trix_line = result.iloc[:, 0].to_numpy()
        signal_line = result.iloc[:, 1].to_numpy()
        histogram = trix_line - signal_line
        return trix_line, signal_line, histogram

    @staticmethod
    @handle_pandas_ta_errors
    def dm(
        high: pd.Series,
        low: pd.Series,
        length: int = 14,
        mamode: str = "rma",
        talib: bool | None = None,
        drift: int = 1,
        offset: int = 0,
    ) -> Tuple[pd.Series, pd.Series]:
        """Directional Movement"""
        result: Any = run_multi_series_indicator(
            {"high": high, "low": low},
            length,
            lambda: ta.dm(
                high=high,
                low=low,
                length=length,
                mamode=mamode,
                talib=talib,
                drift=drift,
                offset=offset,
            ),
            min_data_length=length,
            fallback_factory=lambda: cast(
                tuple[pd.Series, pd.Series], create_nan_series_bundle(high, 2)
            ),
        )

        if isinstance(result, tuple):
            return cast(tuple[pd.Series, pd.Series], result)

        if result is None or (
            hasattr(result, "empty") and getattr(result, "empty", False)
        ):
            return cast(
                tuple[pd.Series, pd.Series], create_nan_series_bundle(high, 2)
            )

        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    @handle_pandas_ta_errors
    def er(
        close: pd.Series,
        length: int = 10,
        drift: int = 1,
        offset: int = 0,
    ) -> pd.Series:
        """Efficiency Ratio"""
        return cast(
            pd.Series,
            run_series_indicator(
                close,
                length,
                lambda: ta.er(
                    close=close, length=length, drift=drift, offset=offset
                ),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def lrsi(
        close: pd.Series,
        length: int = 14,
        gamma: float = 0.5,
        offset: int = 0,
    ) -> pd.Series:
        """Laguerre RSI"""
        return cast(
            pd.Series,
            run_series_indicator(
                close,
                length,
                lambda: ta.lrsi(
                    close=close, length=length, gamma=gamma, offset=offset
                ),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def po(
        close: pd.Series,
        length: int = 14,
        offset: int = 0,
    ) -> pd.Series:
        """Projection Oscillator"""
        return cast(
            pd.Series,
            run_series_indicator(
                close,
                length,
                lambda: ta.po(close=close, length=length, offset=offset),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def trixh(
        close: pd.Series,
        length: int = 18,
        signal: int = 9,
        scalar: float = 100.0,
        drift: int = 1,
        offset: int = 0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """TRIX Histogram"""
        result: Any = run_series_indicator(
            close,
            length,
            lambda: ta.trixh(
                close=close,
                length=length,
                signal=signal,
                scalar=scalar,
                drift=drift,
                offset=offset,
            ),
            fallback_factory=lambda: cast(
                tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(close, 3),
            ),
        )

        if isinstance(result, tuple):
            return result

        return result.iloc[:, 0], result.iloc[:, 1], result.iloc[:, 2]

    @staticmethod
    @handle_pandas_ta_errors
    def vwmacd(
        close: pd.Series,
        volume: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        offset: int = 0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Volume Weighted MACD"""
        result: Any = run_multi_series_indicator(
            {"close": close, "volume": volume},
            max(fast, slow, signal),
            lambda: ta.vwmacd(
                close=close,
                volume=volume,
                fast=fast,
                slow=slow,
                signal=signal,
                offset=offset,
            ),
            fallback_factory=lambda: cast(
                tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(close, 3),
            ),
        )

        if isinstance(result, tuple):
            return result

        return result.iloc[:, 0], result.iloc[:, 1], result.iloc[:, 2]

    @staticmethod
    @handle_pandas_ta_errors
    def stoch(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k: int = 14,
        d: int = 3,
        smooth_k: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        ストキャスティクス（Stochastic Oscillator）

        Args:
            high: 高値
            low: 安値
            close: 終値
            k: %K期間（デフォルト: 14）
            d: %D平滑化期間（デフォルト: 3）
            smooth_k: %K平滑化期間（デフォルト: 3）

        Returns:
            Tuple[%K, %D]
        """
        result: Any = run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            k,
            lambda: ta.stoch(
                high=high,
                low=low,
                close=close,
                length=k,
                smoothd=d,
                smoothk=smooth_k,
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
    def stochrsi(
        data: pd.Series,
        rsi_length: int = 14,
        stoch_length: int = 14,
        k: int = 3,
        d: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic RSI

        RSIにストキャスティクス計算を適用したオシレーター。
        RSIの買われすぎ・売られすぎをより敏感に検出します。

        Args:
            data: 価格データ (通常はclose)
            rsi_length: RSI計算期間 (デフォルト: 14)
            stoch_length: Stochastic計算期間 (デフォルト: 14)
            k: %K平滑化期間 (デフォルト: 3)
            d: %D平滑化期間 (デフォルト: 3)

        Returns:
            Tuple[pd.Series, pd.Series]: (%K, %D)
        """
        # 最小必要データ長の確認
        min_required_length = rsi_length + stoch_length
        if stoch_length <= 0 or k <= 0 or d <= 0:
            raise ValueError("stoch_length, k, and d must be positive")

        result: Any = run_series_indicator(
            data,
            rsi_length,
            lambda: ta.stochrsi(
                close=data,
                rsi_length=rsi_length,
                stoch_length=stoch_length,
                k=k,
                d=d,
            ),
            min_data_length=min_required_length,
            fallback_factory=lambda: cast(
                tuple[pd.Series, pd.Series], create_nan_series_bundle(data, 2)
            ),
        )

        if isinstance(result, tuple):
            return cast(tuple[pd.Series, pd.Series], result)

        # pandas-taの返り値は通常 STOCHRSIk_*, STOCHRSId_* の2列
        return (result.iloc[:, 0], result.iloc[:, 1])

    @staticmethod
    @handle_pandas_ta_errors
    def willr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
        **kwargs,
    ) -> pd.Series:
        """
        ウィリアムズ%R（Williams %R）

        Args:
            high: 高値
            low: 安値
            close: 終値
            length: 期間（デフォルト: 14）

        Returns:
            Williams %R の値（-100 から 0 の範囲）
        """
        return cast(
            pd.Series,
            run_series_indicator(
                close,
                length,
                lambda: ta.willr(
                    high=high, low=low, close=close, length=length, **kwargs
                ),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 20,
    ) -> pd.Series:
        """
        Commodity Channel Index（商品チャネル指数）

        Args:
            high: 高値
            low: 安値
            close: 終値
            length: 期間（デフォルト: 20）

        Returns:
            CCI の値
        """
        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"high": high, "low": low, "close": close},
                length,
                lambda: ta.cci(high=high, low=low, close=close, length=length),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def cmo(
        data: pd.Series,
        length: int = 14,
        talib: bool | None = None,
    ) -> pd.Series:
        """チャンデ・モメンタム・オシレーター"""
        return cast(
            pd.Series,
            run_series_indicator(
                data, length, lambda: ta.cmo(data, length=length, talib=talib)
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def stc(
        data: pd.Series,
        fast: int = 23,
        slow: int = 50,
        cycle: int = 10,
        d1: int = 3,
        d2: int = 3,
    ) -> pd.Series:
        """Schaff Trend Cycle"""
        if fast <= 0 or slow <= 0:
            raise ValueError("fast and slow must be positive")

        factor_base = max(d1, d2)
        factor = max(0.1, min(factor_base / max(cycle, 1), 0.9))

        result: Any = run_series_indicator(
            data,
            cycle,
            lambda: ta.stc(
                data,
                fast=fast,
                slow=slow,
                tclength=cycle,
                factor=factor,
            ),
        )

        if isinstance(result, pd.DataFrame):
            series = result.iloc[:, 0]
        else:
            series = result

        return series

    @staticmethod
    @handle_pandas_ta_errors
    def fisher(
        high: pd.Series,
        low: pd.Series,
        length: int = 9,
        signal: int = 1,
    ) -> Tuple[pd.Series, pd.Series]:
        """フィッシャー変換"""
        if signal <= 0:
            raise ValueError(f"signal must be positive: {signal}")

        result: Any = run_multi_series_indicator(
            {"high": high, "low": low},
            length,
            lambda: ta.fisher(
                high=high, low=low, length=length, signal=signal
            ),
            fallback_factory=lambda: cast(
                tuple[pd.Series, pd.Series], create_nan_series_bundle(high, 2)
            ),
        )

        if isinstance(result, tuple):
            return result

        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    @handle_pandas_ta_errors
    def kst(
        data: pd.Series,
        roc1: int = 10,
        roc2: int = 15,
        roc3: int = 20,
        roc4: int = 30,
        sma1: int = 10,
        sma2: int = 10,
        sma3: int = 10,
        sma4: int = 15,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series]:
        """Know Sure Thing"""
        max_period = max(
            roc1, roc2, roc3, roc4, sma1, sma2, sma3, sma4, signal
        )
        result: Any = run_series_indicator(
            data,
            max_period,
            lambda: ta.kst(
                data,
                roc1=roc1,
                roc2=roc2,
                roc3=roc3,
                roc4=roc4,
                sma1=sma1,
                sma2=sma2,
                sma3=sma3,
                sma4=sma4,
                signal=signal,
            ),
            fallback_factory=lambda: cast(
                tuple[pd.Series, pd.Series], create_nan_series_bundle(data, 2)
            ),
        )

        if isinstance(result, tuple):
            return cast(tuple[pd.Series, pd.Series], result)

        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    @handle_pandas_ta_errors
    def roc(
        data: pd.Series | None = None,
        period: int = 10,
        close: pd.Series | None = None,
    ) -> pd.Series:
        """変化率"""
        length = period

        if data is None and close is not None:
            data = close
        elif data is None:
            raise ValueError("Either 'data' or 'close' must be provided")

        return cast(
            pd.Series,
            run_series_indicator(
                data, length, lambda: ta.roc(data, window=length)
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def mom(data: pd.Series, length: int = 10) -> pd.Series:
        """モメンタム"""
        return cast(
            pd.Series,
            run_series_indicator(
                data, length, lambda: ta.mom(data, length=length)
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def qqe(data: pd.Series, length: int = 14) -> pd.Series:
        """Qualitative Quantitative Estimation"""

        def compute() -> pd.Series:
            """
            QQE の計算ロジック。
            """
            result: Any = ta.qqe(data, length=length)

            if isinstance(result, pd.DataFrame):
                # QQEの主要な列を返す（通常はRSIMA列）
                return (
                    result.iloc[:, 1]
                    if result.shape[1] > 1
                    else result.iloc[:, 0]
                )

            if result is not None:
                return result

            # フォールバック: RSIを返す
            rsi_result = ta.rsi(data, length=length)
            if isinstance(rsi_result, pd.DataFrame):
                return rsi_result.iloc[:, 0]
            return (
                rsi_result
                if rsi_result is not None
                else create_nan_series_like(data)
            )

        return cast(pd.Series, run_series_indicator(data, length, compute))

    @staticmethod
    @handle_pandas_ta_errors
    def cti(
        close: pd.Series,
        length: int = 12,
        offset: int = 0,
    ) -> pd.Series:
        """Correlation Trend Indicator"""
        return cast(
            pd.Series,
            run_series_indicator(
                close,
                length,
                lambda: ta.cti(close=close, length=length, offset=offset),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def squeeze_pro(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        bb_length: int = 20,
        bb_std: float = 2.0,
        kc_length: int = 20,
        kc_scalar_wide: float = 2.0,
        kc_scalar_normal: float = 1.5,
        kc_scalar_narrow: float = 1.0,
        mom_length: int = 12,
        mom_smooth: int = 6,
        use_tr: bool = True,
    ) -> pd.Series | pd.DataFrame:
        """Squeeze Pro"""
        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"high": high, "low": low, "close": close},
                max(bb_length, kc_length),
                lambda: ta.squeeze_pro(
                    high=high,
                    low=low,
                    close=close,
                    bb_length=bb_length,
                    bb_std=bb_std,
                    kc_length=kc_length,
                    kc_scalar_wide=kc_scalar_wide,
                    kc_scalar_normal=kc_scalar_normal,
                    kc_scalar_narrow=kc_scalar_narrow,
                    mom_length=mom_length,
                    mom_smooth=mom_smooth,
                    use_tr=use_tr,
                ),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def apo(
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        ma_mode: str = "ema",
    ) -> pd.Series:
        """Absolute Price Oscillator"""
        if fast <= 0:
            raise ValueError("fast must be positive")
        if fast >= slow:
            raise ValueError("fast period must be less than slow period")

        return cast(
            pd.Series,
            run_series_indicator(
                data,
                slow,
                lambda: ta.apo(data, fast=fast, slow=slow, ma_mode=ma_mode),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def tsi(
        data: pd.Series,
        fast: int = 13,
        slow: int = 25,
        signal: int = 13,
        scalar: float = 100.0,
        mamode: Optional[str] = "ema",
        drift: int = 1,
    ) -> Tuple[pd.Series, pd.Series]:
        """True Strength Index"""
        max_period = max(fast, slow, signal)
        if drift <= 0:
            raise ValueError("drift must be positive")

        result: Any = run_series_indicator(
            data,
            max_period,
            lambda: ta.tsi(
                data,
                fast=fast,
                slow=slow,
                signal=signal,
                scalar=scalar,
                mamode=mamode,
                drift=drift,
            ),
            fallback_factory=lambda: cast(
                tuple[pd.Series, pd.Series], create_nan_series_bundle(data, 2)
            ),
        )

        if isinstance(result, tuple):
            return cast(tuple[pd.Series, pd.Series], result)

        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    @handle_pandas_ta_errors
    def pgo(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """Pretty Good Oscillator"""
        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"high": high, "low": low, "close": close},
                length,
                lambda: ta.pgo(high=high, low=low, close=close, length=length),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def psl(
        close: pd.Series,
        length: int = 12,
        scalar: float = 100.0,
        drift: int = 1,
        open_: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Psychological Line（心理線）

        Args:
            close: 終値
            length: 期間（デフォルト: 12）
            scalar: スケーリング係数（デフォルト: 100.0）
            drift: ドリフト（デフォルト: 1）
            open_: 始値（オプション）

        Returns:
            PSL の値
        """
        return cast(
            pd.Series,
            run_series_indicator(
                close,
                length,
                lambda: ta.psl(
                    close=close,
                    open_=open_,
                    length=length,
                    scalar=scalar,
                    drift=drift,
                ),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def squeeze(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        bb_length: int = 20,
        bb_std: float = 2.0,
        kc_length: int = 20,
        kc_scalar: float = 1.5,
        mom_length: int = 12,
        mom_smooth: int = 6,
        use_tr: bool = True,
    ) -> pd.Series:
        """Squeeze"""
        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"high": high, "low": low, "close": close},
                max(bb_length, kc_length),
                lambda: ta.squeeze(
                    high=high,
                    low=low,
                    close=close,
                    bb_window=bb_length,
                    bb_std=bb_std,
                    kc_window=kc_length,
                    kc_scalar=kc_scalar,
                    mom_window=mom_length,
                    mom_smooth=mom_smooth,
                    use_tr=use_tr,
                ),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def uo(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        fast: int = 7,
        medium: int = 14,
        slow: int = 28,
    ) -> pd.Series:
        """Ultimate Oscillator"""
        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"high": high, "low": low, "close": close},
                slow,
                lambda: ta.uo(
                    high=high,
                    low=low,
                    close=close,
                    fast=fast,
                    medium=medium,
                    slow=slow,
                ),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def ao(
        high: pd.Series, low: pd.Series, fast: int = 5, slow: int = 34
    ) -> pd.Series:
        """Awesome Oscillator"""
        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"high": high, "low": low},
                slow,
                lambda: ta.ao(high=high, low=low, fast=fast, slow=slow),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def bop(
        open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """Balance of Power"""
        return cast(
            pd.Series,
            run_multi_series_indicator(
                {"open_": open_, "high": high, "low": low, "close": close},
                None,
                lambda: ta.bop(open_=open_, high=high, low=low, close=close),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def cg(data: pd.Series, length: int = 10) -> pd.Series:
        """Center of Gravity"""
        return cast(
            pd.Series,
            run_series_indicator(
                data,
                length,
                lambda: ta.cg(data, length=length),
                min_data_length=length + 2,
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def coppock(
        close: pd.Series, length: int = 11, fast: int = 14, slow: int = 10
    ) -> pd.Series:
        """Coppock Curve"""
        min_length = slow + fast + 5
        return cast(
            pd.Series,
            run_series_indicator(
                close,
                length,
                lambda: ta.coppock(
                    close=close, length=length, fast=fast, slow=slow
                ),
                min_data_length=min_length,
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def bias(
        data: pd.Series,
        length: int = 26,
        ma_type: str = "sma",
        offset: int = 0,
    ) -> pd.Series:
        """Bias Indicator - 移動平均からの乖離率"""
        min_length = length * 2

        def compute() -> pd.Series:
            """
            Bias インジケーターの計算ロジック。
            """
            result: Any = None
            try:
                result = ta.bias(
                    close=data,
                    length=length,
                    ma_type=ma_type,
                    offset=offset,
                )
            except Exception:
                ma_func = {
                    "sma": ta.sma,
                    "ema": ta.ema,
                    "wma": ta.wma,
                    "hma": ta.hma,
                    "zlma": ta.zlma,
                }.get(ma_type, ta.sma)

                ma_result = ma_func(data, length=length)
                if ma_result is None or ma_result.isna().all():
                    return create_nan_series_like(data)

                result = (
                    (data - ma_result) / ma_result.replace(0, np.nan)
                ) * 100

            return (
                result if result is not None else create_nan_series_like(data)
            )

        return cast(
            pd.Series,
            run_series_indicator(
                data,
                length,
                compute,
                min_data_length=min_length,
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def efficiency_ratio(data: pd.Series, length: int = 10) -> pd.Series:
        """
        Kaufman Efficiency Ratio (ER)

        ER = Change / Volatility
           = |Price[t] - Price[t-N]| / Sum(|Price[i] - Price[i-1]| for i in t..t-N)
        """
        result: Any = run_series_indicator(
            data,
            length,
            lambda: normalize_non_finite(
                data.diff(length).abs()
                / data.diff(1).abs().rolling(window=length).sum(),
                fill_value=0.0,
            ),
        )
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def brar(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 26,
    ) -> Tuple[pd.Series, pd.Series]:
        """BRAR (Brayer)"""
        result: Any = run_multi_series_indicator(
            {"open_": open_, "high": high, "low": low, "close": close},
            length,
            lambda: ta.brar(
                open_=open_, high=high, low=low, close=close, length=length
            ),
            fallback_factory=lambda: cast(
                tuple[pd.Series, pd.Series], create_nan_series_bundle(close, 2)
            ),
        )

        if isinstance(result, tuple):
            return cast(tuple[pd.Series, pd.Series], result)

        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    @handle_pandas_ta_errors
    def cfo(close: pd.Series, length: int = 9) -> pd.Series:
        """Chande Forecast Oscillator"""
        return cast(
            pd.Series,
            run_series_indicator(
                close, length, lambda: ta.cfo(close=close, length=length)
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def eri(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 13,
    ) -> Tuple[pd.Series, pd.Series]:
        """Elder Ray Index"""
        result: Any = run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            length,
            lambda: ta.eri(high=high, low=low, close=close, length=length),
            fallback_factory=lambda: cast(
                tuple[pd.Series, pd.Series], create_nan_series_bundle(close, 2)
            ),
        )

        if isinstance(result, tuple):
            return cast(tuple[pd.Series, pd.Series], result)

        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    @handle_pandas_ta_errors
    def inertia(
        close: pd.Series,
        high: pd.Series | None = None,
        low: pd.Series | None = None,
        length: int = 20,
        rvi_length: int = 14,
        scalar: float = 100.0,
        refined: bool = False,
        thirds: bool = False,
        mamode: str = "cma",
    ) -> pd.Series:
        """Inertia"""

        return cast(
            pd.Series,
            run_series_indicator(
                close,
                length,
                lambda: ta.inertia(
                    close=close,
                    high=high,
                    low=low,
                    length=length,
                    rvi_length=rvi_length,
                    scalar=scalar,
                    refined=refined,
                    thirds=thirds,
                    mamode=mamode,
                ),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def kdj(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 9,
        signal: int = 3,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """KDJ"""
        result: Any = run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            length,
            lambda: ta.kdj(
                high=high, low=low, close=close, length=length, signal=signal
            ),
            fallback_factory=lambda: cast(
                tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(close, 3),
            ),
        )

        if isinstance(result, tuple):
            return result

        return result.iloc[:, 0], result.iloc[:, 1], result.iloc[:, 2]

    @staticmethod
    @handle_pandas_ta_errors
    def rsx(close: pd.Series, length: int = 14, drift: int = 1) -> pd.Series:
        """RSX"""
        return cast(
            pd.Series,
            run_series_indicator(
                close,
                length,
                lambda: ta.rsx(close=close, length=length, drift=drift),
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def rvgi(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
        swma_length: int = 4,
    ) -> Tuple[pd.Series, pd.Series]:
        """Relative Vigor Index"""
        result: Any = run_multi_series_indicator(
            {"open_": open_, "high": high, "low": low, "close": close},
            length,
            lambda: ta.rvgi(
                open_=open_,
                high=high,
                low=low,
                close=close,
                length=length,
                swma_length=swma_length,
            ),
            fallback_factory=lambda: cast(
                tuple[pd.Series, pd.Series], create_nan_series_bundle(close, 2)
            ),
        )

        if isinstance(result, tuple):
            return cast(tuple[pd.Series, pd.Series], result)

        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    @handle_pandas_ta_errors
    def slope(close: pd.Series, length: int = 1) -> pd.Series:
        """Slope"""
        return cast(
            pd.Series,
            run_series_indicator(
                close, length, lambda: ta.slope(close=close, length=length)
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def smi(
        close: pd.Series,
        fast: int = 5,
        slow: int = 20,
        signal: int = 5,
        scalar: float = 1.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """SMI Ergodic"""
        result: Any = run_series_indicator(
            close,
            slow,
            lambda: ta.smi(
                close=close, fast=fast, slow=slow, signal=signal, scalar=scalar
            ),
            fallback_factory=lambda: cast(
                tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(close, 3),
            ),
        )

        if isinstance(result, tuple):
            return result

        return result.iloc[:, 0], result.iloc[:, 1], result.iloc[:, 2]

    @staticmethod
    @handle_pandas_ta_errors
    def td_seq(
        close: pd.Series, as_bool: bool = False, show_all: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """TD Sequential"""
        return cast(
            pd.Series,
            run_series_indicator(
                close,
                13,
                lambda: ta.td_seq(
                    close=close, asbool=as_bool, show_all=show_all
                ),
            ),
        )

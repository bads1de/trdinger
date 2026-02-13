"""
モメンタム系テクニカル指標 (Momentum Indicators)

pandas-ta の momentum カテゴリに対応。
価格の勢いと転換点の検出に使用する指標群。

登録してあるテクニカルの一覧:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)
- ROC (Rate of Change)
- Momentum
- QQE (Qualitative Quantitative Estimation)
- SQUEEZE (Squeeze)
- STC (Schaff Trend Cycle)
- CMO (Chande Momentum Oscillator)
- FISHER (Fisher Transform)
- KST (Know Sure Thing)
- CTI (Correlation Trend Indicator)
- TSI (True Strength Index)
- PGO (Pretty Good Oscillator)
- PSL (Psychological Line)
- AO (Awesome Oscillator)
- BOP (Balance of Power)
- CG (Center of Gravity)
- COPPOCK (Coppock Curve)
- STOCHRSI (Stochastic RSI)
- BIAS (Bias Indicator)
- ER (Efficiency Ratio)
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

from ..data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
    validate_series_params,
)

logger = logging.getLogger(__name__)


class MomentumIndicators:
    """
    モメンタム系指標クラス

    RSI, MACD, ストキャスティクスなどのモメンタム系テクニカル指標を提供。
    価格の勢いと転換点の検出に使用します。
    """

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """相対力指数"""
        validation = validate_series_params(data, period)
        if validation is not None:
            return validation

        result = ta.rsi(data, window=period)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def macd(
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series]:
        """MACD"""
        validation = validate_series_params(data)
        if validation is not None:
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return (nan_series, nan_series, nan_series)

        result = ta.macd(data, fast=fast, slow=slow, signal=signal)

        if result is None or result.empty:
            # フォールバック: NaN配列を返す
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return (nan_series, nan_series, nan_series)

        return (
            result.iloc[:, 0],  # MACD
            result.iloc[:, 1],  # Signal
            result.iloc[:, 2],  # Histogram
        )

    @staticmethod
    def ppo(
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Percentage Price Oscillator"""
        validation = validate_series_params(data)
        if validation is not None:
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return nan_series, nan_series, nan_series

        result = ta.ppo(data, fast=fast, slow=slow, signal=signal)

        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return nan_series, nan_series, nan_series

        return (
            result.iloc[:, 0].to_numpy(),
            result.iloc[:, 1].to_numpy(),
            result.iloc[:, 2].to_numpy(),
        )

    @staticmethod
    def trix(
        data: pd.Series,
        length: int = 15,
        signal: int = 9,
        scalar: Optional[float] = None,
        drift: int = 1,
        offset: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """TRIX (Triple Exponential Average)"""
        validation = validate_series_params(data, length)
        if validation is not None:
            nan_array = np.full(len(data), np.nan)
            return nan_array, nan_array, nan_array

        result = ta.trix(
            close=data,
            length=length,
            signal=signal,
            scalar=scalar,
            drift=drift,
            offset=offset,
        )

        if result is None or result.empty:
            nan_array = np.full(len(data), np.nan)
            return nan_array, nan_array, nan_array

        trix_line = result.iloc[:, 0].to_numpy()
        signal_line = result.iloc[:, 1].to_numpy()
        histogram = trix_line - signal_line
        return trix_line, signal_line, histogram

    @staticmethod
    @handle_pandas_ta_errors
    def stoch(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k: int = 14,
        d: int = 3,
        smooth_k: int = 3,
        d_length: int = None,
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
            d_length: d のエイリアス（後方互換性）

        Returns:
            Tuple[%K, %D]
        """
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, k
        )
        if validation is not None:
            return validation, validation

        # d_lengthパラメータが指定された場合の処理（後方互換性）
        if d_length is not None and d == 3:
            d = d_length

        result = ta.stoch(
            high=high, low=low, close=close, length=k, smoothd=d, smoothk=smooth_k
        )

        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series

        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
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
        validation = validate_series_params(
            data, rsi_length, min_data_length=min_required_length
        )
        if validation is not None:
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return nan_series, nan_series

        if stoch_length <= 0 or k <= 0 or d <= 0:
            raise ValueError("stoch_length, k, and d must be positive")

        result = ta.stochrsi(
            close=data,
            rsi_length=rsi_length,
            stoch_length=stoch_length,
            k=k,
            d=d,
        )

        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return nan_series, nan_series

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
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        result = ta.willr(high=high, low=low, close=close, length=length, **kwargs)

        if result is None or (hasattr(result, "isna") and result.isna().all()):
            return pd.Series(np.full(len(close), np.nan), index=close.index)

        return result

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
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, length
        )
        if validation is not None:
            return validation

        result = ta.cci(high=high, low=low, close=close, length=length)

        if result is None or (hasattr(result, "empty") and result.empty):
            return pd.Series(np.full(len(high), np.nan), index=high.index)

        return result

    @staticmethod
    def cmo(
        data: pd.Series,
        length: int = 14,
        talib: bool | None = None,
    ) -> pd.Series:
        """チャンデ・モメンタム・オシレーター"""
        validation = validate_series_params(data, length)
        if validation is not None:
            return validation

        result = ta.cmo(data, length=length, talib=talib)
        if result is None or result.empty:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

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
        validation = validate_series_params(data, cycle)
        if validation is not None:
            return validation

        if fast <= 0 or slow <= 0:
            raise ValueError("fast and slow must be positive")

        factor_base = max(d1, d2)
        factor = max(0.1, min(factor_base / max(cycle, 1), 0.9))

        result = ta.stc(
            data,
            fast=fast,
            slow=slow,
            tclength=cycle,
            factor=factor,
        )
        if result is None or result.empty:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        if isinstance(result, pd.DataFrame):
            series = result.iloc[:, 0]
        else:
            series = result

        return series

    @staticmethod
    def fisher(
        high: pd.Series,
        low: pd.Series,
        length: int = 9,
        signal: int = 1,
    ) -> Tuple[pd.Series, pd.Series]:
        """フィッシャー変換"""
        validation = validate_multi_series_params({"high": high, "low": low}, length)
        if validation is not None:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series

        if signal <= 0:
            raise ValueError(f"signal must be positive: {signal}")

        result = ta.fisher(high=high, low=low, length=length, signal=signal)
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series

        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
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
        max_period = max(roc1, roc2, roc3, roc4, sma1, sma2, sma3, sma4, signal)
        validation = validate_series_params(data, max_period)
        if validation is not None:
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return nan_series, nan_series

        result = ta.kst(
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
        )

        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return nan_series, nan_series

        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    def roc(
        data: pd.Series,
        period: int = 10,
        close: pd.Series = None,
    ) -> pd.Series:
        """変化率"""
        length = period
        # dataが提供されない場合はcloseを使用
        if data is None and close is not None:
            data = close
        elif data is None:
            raise ValueError("Either 'data' or 'close' must be provided")

        validation = validate_series_params(data, length)
        if validation is not None:
            return validation

        result = ta.roc(data, window=length)
        if result is None or result.empty:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def mom(data: pd.Series, length: int = 10) -> pd.Series:
        """モメンタム"""
        validation = validate_series_params(data, length)
        if validation is not None:
            return validation

        result = ta.mom(data, length=length)
        if result is None or result.empty:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def qqe(data: pd.Series, length: int = 14) -> pd.Series:
        """Qualitative Quantitative Estimation"""
        validation = validate_series_params(data, length)
        if validation is not None:
            return validation

        result = ta.qqe(data, length=length)

        if result is None or result.empty:
            # フォールバック: RSIを返す
            rsi_result = ta.rsi(data, length=length)
            return (
                rsi_result
                if rsi_result is not None
                else pd.Series(np.full(len(data), np.nan), index=data.index)
            )

        # QQEの主要な列を返す（通常はRSIMA列）
        return result.iloc[:, 1] if result.shape[1] > 1 else result.iloc[:, 0]

    @staticmethod
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
    ) -> pd.Series:
        """Squeeze Pro"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, max(bb_length, kc_length)
        )
        if validation is not None:
            return validation

        # Note: pandas-ta squeeze_pro might return multiple columns or DataFrame with details
        # For uniformity, we often want the main squeeze value or a specific signal
        # Use pandas-ta default returns for now
        result = ta.squeeze_pro(
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
        )

        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)

        # Squeeze Pro typically returns SQZ_PRO_ON, SQZ_PRO_OFF, SQZ_PRO_NO, SQZ_PRO_W, SQZ_PRO_N, SQZ_PRO_S
        # Returning the DataFrame as is, or specific column depending on requirement.
        # Strategy usually looks for compression states.
        # Let's return the simplified main series if identifiable, or the whole DF if useful.

        if isinstance(result, pd.DataFrame):
            # If the user wants just one series, maybe the 'on/off' state or momentum?
            # Standard squeeze returns momentum value. Squeeze Pro returns states mostly?
            # Actually standard squeeze returns momentum bar.
            # Let's check columns. Typical: SQZPRO_20_2.0_20_2_1.5_1
            # We will return the full dataframe and let caller handle, or just the first column?
            # To be safe and compliant with 'Series' return type if possible, or 'DataFrame'
            # But the signature says pd.Series. Let's change return type to Union or just return DF masquerading.
            # However, looking at standard Squeeze, it returns momentum.
            # Squeeze Pro in pandas-ta likely returns a DataFrame with multiple boolean columns state.
            pass

        # For this specific indicator which is complex, returning the raw result (likely DF)
        # but type hinting suggests Series. We should update type hint or wrapper if we want specific column.
        # Let's trust pandas-ta return and just handle None/Empty.
        return result

    @staticmethod
    def cti(data: pd.Series, length: int = 12) -> pd.Series:
        """Correlation Trend Indicator"""
        validation = validate_series_params(data, length)
        if validation is not None:
            return validation

        result = ta.cti(data, length=length)

        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def apo(
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        ma_mode: str = "ema",
    ) -> pd.Series:
        """Absolute Price Oscillator"""
        validation = validate_series_params(data, slow)
        if validation is not None:
            return validation

        if fast <= 0:
            raise ValueError("fast must be positive")
        if fast >= slow:
            raise ValueError("fast period must be less than slow period")

        result = ta.apo(data, fast=fast, slow=slow, ma_mode=ma_mode)

        if result is None or result.empty:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
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
        validation = validate_series_params(data, max_period)
        if validation is not None:
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return nan_series, nan_series

        if drift <= 0:
            raise ValueError("drift must be positive")

        result = ta.tsi(
            data,
            fast=fast,
            slow=slow,
            signal=signal,
            scalar=scalar,
            mamode=mamode,
            drift=drift,
        )

        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return nan_series, nan_series

        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    def pgo(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """Pretty Good Oscillator"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, length
        )
        if validation is not None:
            return validation

        result = ta.pgo(high=high, low=low, close=close, length=length)

        if result is None or result.empty:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

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
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        result = ta.psl(
            close=close, open_=open_, length=length, scalar=scalar, drift=drift
        )

        if result is None or (hasattr(result, "isna") and result.isna().all()):
            return pd.Series(np.full(len(close), np.nan), index=close.index)

        return result

    @staticmethod
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
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, max(bb_length, kc_length)
        )
        if validation is not None:
            return validation

        result = ta.squeeze(
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
        )

        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def uo(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        fast: int = 7,
        medium: int = 14,
        slow: int = 28,
    ) -> pd.Series:
        """Ultimate Oscillator"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, slow
        )
        if validation is not None:
            return validation

        result = ta.uo(
            high=high, low=low, close=close, fast=fast, medium=medium, slow=slow
        )

        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def ao(high: pd.Series, low: pd.Series, fast: int = 5, slow: int = 34) -> pd.Series:
        """Awesome Oscillator"""
        validation = validate_multi_series_params({"high": high, "low": low}, slow)
        if validation is not None:
            return validation

        result = ta.ao(high=high, low=low, fast=fast, slow=slow)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def bop(
        open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """Balance of Power"""
        validation = validate_multi_series_params(
            {"open_": open_, "high": high, "low": low, "close": close}
        )
        if validation is not None:
            return validation

        result = ta.bop(open_=open_, high=high, low=low, close=close)
        if result is None:
            return pd.Series(np.full(len(open_), np.nan), index=open_.index)
        return result

    @staticmethod
    def cg(data: pd.Series, length: int = 10) -> pd.Series:
        """Center of Gravity"""
        # CG特有のデータ検証 (min_length = length + 2)
        validation = validate_series_params(data, length, min_data_length=length + 2)
        if validation is not None:
            return validation

        result = ta.cg(data, length=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def coppock(
        close: pd.Series, length: int = 11, fast: int = 14, slow: int = 10
    ) -> pd.Series:
        """Coppock Curve"""
        min_length = slow + fast + 5
        validation = validate_series_params(close, length, min_data_length=min_length)
        if validation is not None:
            return validation

        result = ta.coppock(close=close, length=length, fast=fast, slow=slow)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def bias(
        data: pd.Series,
        length: int = 26,
        ma_type: str = "sma",
        offset: int = 0,
    ) -> pd.Series:
        """Bias Indicator - 移動平均からの乖離率"""
        # BIAS requires sufficient data length
        min_length = length * 2
        validation = validate_series_params(data, length, min_data_length=min_length)
        if validation is not None:
            return validation

        # pandas-taのbias関数を直接使用
        try:
            result = ta.bias(
                close=data,
                length=length,
                ma_type=ma_type,
                offset=offset,
            )
        except Exception:
            # pandas-taのbiasが利用できない場合のフォールバック実装
            ma_func = {
                "sma": ta.sma,
                "ema": ta.ema,
                "wma": ta.wma,
                "hma": ta.hma,
                "zlma": ta.zlma,
            }.get(ma_type, ta.sma)

            ma_result = ma_func(data, length=length)
            if ma_result is None or ma_result.isna().all():
                return pd.Series(np.full(len(data), np.nan), index=data.index)

            # BIAS = (close - ma) / ma * 100
            result = ((data - ma_result) / ma_result) * 100

        if result is None or (hasattr(result, "isna") and result.isna().all()):
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def efficiency_ratio(data: pd.Series, length: int = 10) -> pd.Series:
        """
        Kaufman Efficiency Ratio (ER)

        ER = Change / Volatility
           = |Price[t] - Price[t-N]| / Sum(|Price[i] - Price[i-1]| for i in t..t-N)
        """
        validation = validate_series_params(data, length)
        if validation is not None:
            return validation

        change = data.diff(length).abs()
        volatility = data.diff(1).abs().rolling(window=length).sum()

        er = change / volatility
        return er.replace([np.inf, -np.inf], 0.0).fillna(0.0)

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
        validation = validate_multi_series_params(
            {"open_": open_, "high": high, "low": low, "close": close}, length
        )
        if validation is not None:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series

        result = ta.brar(open_=open_, high=high, low=low, close=close, length=length)
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series

        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    @handle_pandas_ta_errors
    def cfo(close: pd.Series, length: int = 9) -> pd.Series:
        """Chande Forecast Oscillator"""
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        result = ta.cfo(close=close, length=length)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def eri(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 13,
    ) -> Tuple[pd.Series, pd.Series]:
        """Elder Ray Index"""
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, length
        )
        if validation is not None:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series

        result = ta.eri(high=high, low=low, close=close, length=length)
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series

        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    @handle_pandas_ta_errors
    def inertia(
        close: pd.Series,
        high: pd.Series = None,
        low: pd.Series = None,
        length: int = 20,
        rvi_length: int = 14,
        scalar: float = 100.0,
        refined: bool = False,
        thirds: bool = False,
        mamode: str = "cma",
    ) -> pd.Series:
        """Inertia"""
        # Inertia requires RVI params if high/low provided, otherwise just close/length check
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        result = ta.inertia(
            close=close,
            high=high,
            low=low,
            length=length,
            rvi_length=rvi_length,
            scalar=scalar,
            refined=refined,
            thirds=thirds,
            mamode=mamode,
        )
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

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
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, length
        )
        if validation is not None:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series

        result = ta.kdj(high=high, low=low, close=close, length=length, signal=signal)
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series

        return result.iloc[:, 0], result.iloc[:, 1], result.iloc[:, 2]

    @staticmethod
    @handle_pandas_ta_errors
    def rsx(close: pd.Series, length: int = 14, drift: int = 1) -> pd.Series:
        """RSX"""
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        result = ta.rsx(close=close, length=length, drift=drift)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

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
        validation = validate_multi_series_params(
            {"open_": open_, "high": high, "low": low, "close": close}, length
        )
        if validation is not None:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series

        result = ta.rvgi(
            open_=open_,
            high=high,
            low=low,
            close=close,
            length=length,
            swma_length=swma_length,
        )
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series

        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    @handle_pandas_ta_errors
    def slope(close: pd.Series, length: int = 1) -> pd.Series:
        """Slope"""
        validation = validate_series_params(close, length)
        if validation is not None:
            return validation

        result = ta.slope(close=close, length=length)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

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
        validation = validate_series_params(close, slow)
        if validation is not None:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series

        result = ta.smi(close=close, fast=fast, slow=slow, signal=signal, scalar=scalar)
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series

        return result.iloc[:, 0], result.iloc[:, 1], result.iloc[:, 2]

    @staticmethod
    @handle_pandas_ta_errors
    def td_seq(
        close: pd.Series, as_bool: bool = False, show_all: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """TD Sequential"""
        # Minimum requirement for TD Seq (usually 9 candles)
        validation = validate_series_params(close, 13)
        if validation is not None:
            if show_all:
                # Return DataFrame-like structure if expected
                nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
                return nan_series  # Simplified fallback
            return validation

        result = ta.td_seq(close=close, asbool=as_bool, show_all=show_all)
        if result is None or (hasattr(result, "empty") and result.empty):
            return pd.Series(np.full(len(close), np.nan), index=close.index)

        # If returning DataFrame, caller handles it. If Series (usually TD Seq Number), return it.
        # td_seq usually returns DataFrame with multiple columns. We return as is.
        return result

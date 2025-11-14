"""
モメンタム系テクニカル指標

登録してあるテクニカルの一覧:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)
- ROC (Rate of Change)
- Momentum
- ADX (Average Directional Index)
- QQE (Qualitative Quantitative Estimation)
- SQUEEZE (Squeeze)
- STC (Schaff Trend Cycle)
- CMO (Chande Momentum Oscillator)
- FISHER (Fisher Transform)
- KST (Know Sure Thing)
- CTI (Correlation Trend Indicator)
- TSI (True Strength Index)
- PGO (Pretty Good Oscillator)
- MASSI (Mass Index)
- PSL (Psychological Line)
- AO (Awesome Oscillator)
- AROON (Aroon Indicator)
- CHOP (Choppiness Index)
- BOP (Balance of Power)
- CG (Center of Gravity)
- COPPOCK (Coppock Curve)
- STOCHRSI (Stochastic RSI)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import handle_pandas_ta_errors

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
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        length = period
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        result = ta.rsi(data, window=length)
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
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

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
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        result = ta.ppo(data, fast=fast, slow=slow, signal=signal)

        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return nan_series, nan_series, nan_series

        result = result.bfill().fillna(0)

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
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")
        if signal <= 0:
            raise ValueError(f"signal must be positive: {signal}")
        if drift <= 0:
            raise ValueError(f"drift must be positive: {drift}")

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

        result = result.bfill().fillna(0)
        trix_line = result.iloc[:, 0].to_numpy()
        signal_line = result.iloc[:, 1].to_numpy()
        histogram = trix_line - signal_line
        return trix_line, signal_line, histogram

    @staticmethod
    def stoch(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k: int = 14,
        d: int = 3,
        smooth_k: int = 3,
        d_length: int = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """ストキャスティクス"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if k <= 0:
            raise ValueError(f"k must be positive: {k}")
        if d <= 0:
            raise ValueError(f"d must be positive: {d}")
        if smooth_k <= 0:
            raise ValueError(f"smooth_k must be positive: {smooth_k}")

        # d_lengthパラメータが指定された場合の処理（後方互換性）
        if d_length is not None and d == 3:  # dがデフォルトの場合のみ
            d = d_length

        result = ta.stoch(
            high=high,
            low=low,
            close=close,
            length=k,
            smoothd=d,
            smoothk=smooth_k,
        )

        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series
        return (result.iloc[:, 0], result.iloc[:, 1])

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
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if rsi_length <= 0:
            raise ValueError(f"rsi_length must be positive: {rsi_length}")
        if stoch_length <= 0:
            raise ValueError(f"stoch_length must be positive: {stoch_length}")
        if k <= 0:
            raise ValueError(f"k must be positive: {k}")
        if d <= 0:
            raise ValueError(f"d must be positive: {d}")

        # 空データの処理
        if len(data) == 0:
            empty_series = pd.Series([], dtype=float, index=data.index)
            return empty_series, empty_series

        # 最小必要データ長の確認
        min_required_length = rsi_length + stoch_length
        if len(data) < min_required_length:
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return nan_series, nan_series

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
    def willr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """ウィリアムズ%R"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.willr(high=high, low=low, close=close, length=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 20,
    ) -> pd.Series:
        """商品チャネル指数"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.cci(high=high, low=low, close=close, length=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def cmo(
        data: pd.Series,
        length: int = 14,
        talib: bool | None = None,
    ) -> pd.Series:
        """チャンデ・モメンタム・オシレーター"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        result = ta.cmo(data, length=length, talib=talib)
        if result is None or result.empty:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result.bfill().fillna(0)

    @staticmethod
    def stc(
        data: pd.Series,
        fast: int = 23,
        slow: int = 50,
        cycle: int = 10,
        d1: int = 3,
        d2: int = 3,
    ) -> pd.Series:
        """Schaff Trend Cycle"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if fast <= 0 or slow <= 0:
            raise ValueError("fast and slow must be positive")
        if cycle <= 0:
            raise ValueError("cycle must be positive")

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

        return series.bfill().fillna(0)

    @staticmethod
    def fisher(
        high: pd.Series,
        low: pd.Series,
        length: int = 9,
        signal: int = 1,
    ) -> Tuple[pd.Series, pd.Series]:
        """フィッシャー変換"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")
        if signal <= 0:
            raise ValueError(f"signal must be positive: {signal}")

        result = ta.fisher(high=high, low=low, length=length, signal=signal)
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series

        result = result.bfill().fillna(0)
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
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        for name, value in {
            "roc1": roc1,
            "roc2": roc2,
            "roc3": roc3,
            "roc4": roc4,
            "sma1": sma1,
            "sma2": sma2,
            "sma3": sma3,
            "sma4": sma4,
            "signal": signal,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive: {value}")

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

        result = result.bfill().fillna(0)
        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    def roc(
        data: pd.Series,
        period: int = 10,
        close: pd.Series = None,
    ) -> pd.Series:
        """変化率"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        length = period
        # dataが提供されない場合はcloseを使用
        if data is None and close is not None:
            data = close
        elif data is None:
            raise ValueError("Either 'data' or 'close' must be provided")

        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        result = ta.roc(data, window=length)
        if result is None or result.empty:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def mom(data: pd.Series, length: int = 10) -> pd.Series:
        """モメンタム"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        result = ta.mom(data, length=length)
        if result is None or result.empty:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        length: int = None,
        **kwargs,
    ) -> pd.Series:
        """平均方向性指数"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # backward compatibility: lengthパラメータをperiodにマッピング
        if length is not None:
            period = length

        result = ta.adx(high=high, low=low, close=close, length=period)
        if result is None or result.empty:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result.iloc[:, 0]  # ADX列

    @staticmethod
    def qqe(data: pd.Series, length: int = 14) -> pd.Series:
        """Qualitative Quantitative Estimation"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
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
    def cti(data: pd.Series, length: int = 12) -> pd.Series:
        """Correlation Trend Indicator"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        result = ta.cti(data, length=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result.bfill().fillna(0)

    @staticmethod
    def apo(
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        ma_mode: str = "ema",
    ) -> pd.Series:
        """Absolute Price Oscillator"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if fast <= 0 or slow <= 0:
            raise ValueError("fast and slow must be positive")
        if fast >= slow:
            raise ValueError("fast period must be less than slow period")

        result = ta.apo(data, fast=fast, slow=slow, ma_mode=ma_mode)
        if result is None or result.empty:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result.bfill().fillna(0)

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
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if fast <= 0 or slow <= 0 or signal <= 0:
            raise ValueError("fast, slow, signal must be positive")
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

        result = result.bfill().fillna(0)
        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    def pgo(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """Pretty Good Oscillator"""
        for series, name in ((high, "high"), (low, "low"), (close, "close")):
            if not isinstance(series, pd.Series):
                raise TypeError(f"{name} must be pandas Series")

        result = ta.pgo(high=high, low=low, close=close, length=length)
        if result is None or result.empty:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result.bfill().fillna(0)

    @staticmethod
    def massi(
        high: pd.Series,
        low: pd.Series,
        fast: int = 9,
        slow: int = 25,
    ) -> pd.Series:
        """Mass Index"""
        for series, name in ((high, "high"), (low, "low")):
            if not isinstance(series, pd.Series):
                raise TypeError(f"{name} must be pandas Series")
        if fast <= 0 or slow <= 0:
            raise ValueError("fast and slow must be positive")

        result = ta.massi(high=high, low=low, fast=fast, slow=slow)
        if result is None or result.empty:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result.bfill().fillna(0)

    @staticmethod
    def psl(
        close: pd.Series,
        length: int = 12,
        scalar: float = 100.0,
        drift: int = 1,
        open_: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Psychological Line"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if open_ is not None and not isinstance(open_, pd.Series):
            raise TypeError("open_ must be pandas Series")
        if length <= 0 or drift <= 0:
            raise ValueError("length and drift must be positive")

        result = ta.psl(
            close=close,
            open_=open_,
            length=length,
            scalar=scalar,
            drift=drift,
        )
        if result is None or result.empty:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result.bfill().fillna(0)

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
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

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
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.uo(
            high=high, low=low, close=close, fast=fast, medium=medium, slow=slow
        )

        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result.bfill().fillna(0)

    @staticmethod
    def ao(high: pd.Series, low: pd.Series, fast: int = 5, slow: int = 34) -> pd.Series:
        """Awesome Oscillator"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        # AO特有のデータ検証
        if len(high) != len(low):
            raise ValueError("AO requires high and low series to have the same length")

        # 最小データ長チェック
        min_length = slow + 5
        if len(high) < min_length:
            raise ValueError(
                f"Insufficient data for AO calculation. Need at least {min_length} points, got {len(high)}"
            )

        result = ta.ao(high=high, low=low, fast=fast, slow=slow)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result.bfill().fillna(0)

    @staticmethod
    def aroon(
        high: pd.Series, low: pd.Series, length: int = 14
    ) -> Tuple[pd.Series, pd.Series]:
        """Aroon Indicator"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        # AROON特有のデータ検証
        if len(high) != len(low):
            raise ValueError(
                "AROON requires high and low series to have the same length"
            )

        # 最小データ長チェック
        min_length = length + 5
        if len(high) < min_length:
            raise ValueError(
                f"Insufficient data for AROON calculation. Need at least {min_length} points, got {len(high)}"
            )

        result = ta.aroon(high=high, low=low, length=length)
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series

        # 結果がDataFrameの場合、Aroon UpとAroon Downを分離
        if isinstance(result, pd.DataFrame) and len(result.columns) >= 2:
            return result.iloc[:, 0], result.iloc[:, 1]
        else:
            # 単一シリーズの場合はコピーを作成
            up = (
                result.copy()
                if hasattr(result, "copy")
                else pd.Series(result, index=high.index)
            )
            down = pd.Series(np.full(len(result), np.nan), index=high.index)
            return up, down

    @staticmethod
    def chop(
        high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
    ) -> pd.Series:
        """Choppiness Index"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # CHOP特有のデータ検証
        series_lengths = [len(high), len(low), len(close)]
        if not all(length == series_lengths[0] for length in series_lengths):
            raise ValueError("CHOP requires all series to have the same length")

        # 最小データ長チェック
        min_length = length + 5
        if series_lengths[0] < min_length:
            raise ValueError(
                f"Insufficient data for CHOP calculation. Need at least {min_length} points, got {series_lengths[0]}"
            )

        result = ta.chop(high=high, low=low, close=close, length=length)
        if result is None:
            return pd.Series(np.full(series_lengths[0], np.nan), index=high.index)
        return result.bfill().fillna(0)

    @staticmethod
    def bop(
        open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """Balance of Power"""
        if not isinstance(open_, pd.Series):
            raise TypeError("open_ must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # BOP特有のデータ検証
        series_lengths = [len(open_), len(high), len(low), len(close)]
        if not all(length == series_lengths[0] for length in series_lengths):
            raise ValueError("BOP requires all series to have the same length")

        result = ta.bop(open_=open_, high=high, low=low, close=close)
        if result is None:
            return pd.Series(np.full(series_lengths[0], np.nan), index=open_.index)
        return result.bfill().fillna(0)

    @staticmethod
    def cg(data: pd.Series, length: int = 10) -> pd.Series:
        """Center of Gravity"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        # CG特有のデータ検証
        min_length = length + 2
        if len(data) < min_length:
            raise ValueError(
                f"Insufficient data for CG calculation. Need at least {min_length} points, got {len(data)}"
            )

        result = ta.cg(data, length=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result.bfill().fillna(0)

    @staticmethod
    def coppock(
        close: pd.Series, length: int = 11, fast: int = 14, slow: int = 10
    ) -> pd.Series:
        """Coppock Curve"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # Coppock特有のデータ検証
        min_length = slow + fast + 5
        if len(close) < min_length:
            raise ValueError(
                f"Insufficient data for Coppock calculation. Need at least {min_length} points, got {len(close)}"
            )

        result = ta.coppock(close=close, length=length, fast=fast, slow=slow)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result.bfill().fillna(0)

    @staticmethod
    def ichimoku(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_span_b_period: int = 52,
    ) -> dict:
        """Ichimoku Cloud (一目均衡表)

        トレンド、サポート/レジスタンス、モメンタムを同時に分析する包括的なインジケーター。
        5つのコンポーネントで構成される:
        - Tenkan-sen (転換線): 短期トレンド
        - Kijun-sen (基準線): 中期トレンド
        - Senkou Span A (先行スパンA): 未来のサポート/レジスタンス
        - Senkou Span B (先行スパンB): より長期のサポート/レジスタンス
        - Chikou Span (遅行スパン): 遅行スパン

        Args:
            high: 高値の系列
            low: 安値の系列
            close: 終値の系列
            tenkan_period: 転換線の期間 (default: 9)
            kijun_period: 基準線の期間 (default: 26)
            senkou_span_b_period: 先行スパンBの期間 (default: 52)

        Returns:
            dict: 各コンポーネントを含む辞書
        """
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # データ長の検証
        series_lengths = [len(high), len(low), len(close)]
        if not all(length == series_lengths[0] for length in series_lengths):
            raise ValueError("Ichimoku requires all series to have the same length")

        if len(high) == 0:
            # 空データの場合
            empty_series = pd.Series(np.full(0, np.nan), index=high.index)
            return {
                "tenkan_sen": empty_series,
                "kijun_sen": empty_series,
                "senkou_span_a": empty_series,
                "senkou_span_b": empty_series,
                "chikou_span": empty_series,
            }

        # pandas-taを使ってIchimoku Cloudを計算
        try:
            result = ta.ichimoku(
                high=high,
                low=low,
                close=close,
                tenkan=tenkan_period,
                kijun=kijun_period,
                senkou=senkou_span_b_period,
            )

            if result is None or result.empty:
                # pandas-taが失敗した場合のフォールバック
                nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
                return {
                    "tenkan_sen": nan_series,
                    "kijun_sen": nan_series,
                    "senkou_span_a": nan_series,
                    "senkou_span_b": nan_series,
                    "chikou_span": nan_series,
                }

            # 結果を処理
            ichimoku_dict = {}

            # tenkan_sen (転換線) - 最短期間の高値と安値の平均
            if "TENKAN" in result.columns:
                ichimoku_dict["tenkan_sen"] = result["TENKAN"].fillna(np.nan)
            else:
                # フォールバック計算
                tenkan_high = high.rolling(window=tenkan_period).max()
                tenkan_low = low.rolling(window=tenkan_period).min()
                ichimoku_dict["tenkan_sen"] = ((tenkan_high + tenkan_low) / 2).fillna(
                    np.nan
                )

            # kijun_sen (基準線) - 中期間の高値と安値の平均
            if "KIJUN" in result.columns:
                ichimoku_dict["kijun_sen"] = result["KIJUN"].fillna(np.nan)
            else:
                kijun_high = high.rolling(window=kijun_period).max()
                kijun_low = low.rolling(window=kijun_period).min()
                ichimoku_dict["kijun_sen"] = ((kijun_high + kijun_low) / 2).fillna(
                    np.nan
                )

            # senkou_span_a (先行スパンA) - tenkanとkijunの平均を前方にずらす
            if "SENKOU" in result.columns:
                ichimoku_dict["senkou_span_a"] = result["SENKOU"].fillna(np.nan)
            else:
                # フォールバック計算
                senkou_a = (
                    ichimoku_dict["tenkan_sen"] + ichimoku_dict["kijun_sen"]
                ) / 2
                # 前方にずらす (pandas-taは自動で処理してくれるが、フォールバックではNaNで埋める)
                ichimoku_dict["senkou_span_a"] = senkou_a.shift(kijun_period).fillna(
                    np.nan
                )

            # senkou_span_b (先行スパンB) - 長期間の高値と安値の平均を前方にずらす
            if "SANSEN" in result.columns:
                ichimoku_dict["senkou_span_b"] = result["SANSEN"].fillna(np.nan)
            else:
                senkou_b_high = high.rolling(window=senkou_span_b_period).max()
                senkou_b_low = low.rolling(window=senkou_span_b_period).min()
                senkou_b = (senkou_b_high + senkou_b_low) / 2
                ichimoku_dict["senkou_span_b"] = senkou_b.shift(kijun_period).fillna(
                    np.nan
                )

            # chikou_span (遅行スパン) - 終値を後方にずらす
            if "CHIKOU" in result.columns:
                ichimoku_dict["chikou_span"] = result["CHIKOU"].fillna(np.nan)
            else:
                ichimoku_dict["chikou_span"] = close.shift(-kijun_period).fillna(np.nan)

            return ichimoku_dict

        except Exception as e:
            logger.warning(
                f"Ichimoku calculation failed with pandas-ta: {e}. Using fallback calculation."
            )
            # pandas-taが失敗した場合のフォールバック計算
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)

            # 単純なフォールバック計算
            tenkan_high = high.rolling(window=tenkan_period).max()
            tenkan_low = low.rolling(window=tenkan_period).min()
            tenkan_sen = ((tenkan_high + tenkan_low) / 2).fillna(np.nan)

            kijun_high = high.rolling(window=kijun_period).max()
            kijun_low = low.rolling(window=kijun_period).min()
            kijun_sen = ((kijun_high + kijun_low) / 2).fillna(np.nan)

            senkou_span_a = (
                ((tenkan_sen + kijun_sen) / 2).shift(kijun_period).fillna(np.nan)
            )

            senkou_b_high = high.rolling(window=senkou_span_b_period).max()
            senkou_b_low = low.rolling(window=senkou_span_b_period).min()
            senkou_span_b = (
                ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period).fillna(np.nan)
            )

            chikou_span = close.shift(-kijun_period).fillna(np.nan)

            return {
                "tenkan_sen": tenkan_sen,
                "kijun_sen": kijun_sen,
                "senkou_span_a": senkou_span_a,
                "senkou_span_b": senkou_span_b,
                "chikou_span": chikou_span,
            }

    @staticmethod
    @handle_pandas_ta_errors
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
        **kwargs,
    ) -> pd.Series:
        """ウィリアムズ%R"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # Williams %R requires sufficient data length
        min_length = max(length * 2, 14)
        if len(high) < min_length:
            return pd.Series(np.full(len(high), np.nan), index=high.index)

        # pandas-taのwpr関数を使用
        try:
            result = ta.wpr(
                high=high,
                low=low,
                close=close,
                length=length,
                **kwargs,
            )
        except Exception:
            # pandas-taが利用できない場合のフォールバック実装
            # Williams %R = [(highest high - current close) / (highest high - lowest low)] * -100

            # 過去length期間の最高値と最低値を計算
            highest_high = high.rolling(window=length, min_periods=1).max()
            lowest_low = low.rolling(window=length, min_periods=1).min()

            # Williams %R計算
            result = ((highest_high - close) / (highest_high - lowest_low)) * -100

        if result is None or (hasattr(result, "isna") and result.isna().all()):
            return pd.Series(np.full(len(close), np.nan), index=close.index)

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def psychological_line(
        close: pd.Series,
        length: int = 12,
        offset: int = 0,
    ) -> pd.Series:
        """Psychological Line (PSY) - 投資家の心理状態を測定するオシレーター"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # PSY requires sufficient data length
        min_length = max(length * 2, 12)
        if len(close) < min_length:
            return pd.Series(np.full(len(close), np.nan), index=close.index)

        # pandas-taのpsl関数を使用 (PSL = Psychological Line)
        try:
            result = ta.psl(
                close=close,
                length=length,
                offset=offset,
            )
        except Exception:
            # pandas-taが利用できない場合のフォールバック実装
            # PSY = (上昇日数 / 総日数) * 100

            # 価格の変化を計算
            price_change = close.diff()

            # 上昇日数をカウント
            up_days = (price_change > 0).rolling(window=length, min_periods=1).sum()

            # PSY計算
            result = (up_days / length) * 100

        if result is None or (hasattr(result, "isna") and result.isna().all()):
            return pd.Series(np.full(len(close), np.nan), index=close.index)

        return result

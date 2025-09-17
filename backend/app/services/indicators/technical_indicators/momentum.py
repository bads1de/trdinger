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
"""

from typing import Tuple
import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


class MomentumIndicators:
    """
    モメンタム系指標クラス
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

        # d_lengthパラメータが指定された場合の処理
        # ユーザーが明示的にsmooth_kとd_lengthの両方を指定した場合、d_lengthをdとして使用
        if d_length is not None:
            # smooth_kがデフォルト値(1)の場合のみd_lengthで上書き
            if smooth_k == 1:
                smooth_k = d_length

        result = ta.stoch(
            high=high,
            low=low,
            close=close,
            k=k,
            d=d,
            smooth_k=smooth_k,
        )

        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series
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
        **kwargs
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
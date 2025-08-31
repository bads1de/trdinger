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
- Aroon
- MFI (Money Flow Index)
- APO (Absolute Price Oscillator)
- AO (Awesome Oscillator)
- BIAS (Bias)
- BRAR (BRAR Index)
- CG (Center of Gravity)
- COPPOCK (Coppock Curve)
- ER (Efficiency Ratio)
- ERI (Elder Ray Index)
- FISHER (Fisher Transform)
- INERTIA (Inertia)
- PGO (Pretty Good Oscillator)
- PSL (Psychological Line)
- RSX (RSX)
- SQUEEZE (Squeeze)
- SQUEEZE_PRO (Squeeze Pro)
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
    def rsi(data: pd.Series, length: int = 14) -> pd.Series:
        """相対力指数"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        result = ta.rsi(data, length=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def macd(
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
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
    ) -> Tuple[pd.Series, pd.Series]:
        """ストキャスティクス"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

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
        length: int = 10,
        close: pd.Series = None,
    ) -> pd.Series:
        """変化率"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        # dataが提供されない場合はcloseを使用
        if data is None and close is not None:
            data = close
        elif data is None:
            raise ValueError("Either 'data' or 'close' must be provided")

        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        result = ta.roc(data, length=length)
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
        length: int = 14,
    ) -> pd.Series:
        """平均方向性指数"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.adx(high=high, low=low, close=close, length=length)
        if result is None or result.empty:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result.iloc[:, 0]  # ADX列

    @staticmethod
    def aroon(
        high: pd.Series,
        low: pd.Series,
        length: int = 14,
    ) -> Tuple[pd.Series, pd.Series]:
        """アルーン"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        result = ta.aroon(high=high, low=low, length=length)
        if result is None or result.empty:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series
        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """マネーフローインデックス"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        result = ta.mfi(
            high=high,
            low=low,
            close=close,
            volume=volume,
            length=length,
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
        """アルティメットオシレーター"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.uo(
            high=high,
            low=low,
            close=close,
            fast=fast,
            medium=medium,
            slow=slow,
        )
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def apo(data: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Absolute Price Oscillator"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        result = ta.apo(data, fast=fast, slow=slow)
        if result is None or result.empty:
            # フォールバック: 簡易実装 (EMAの差分)
            ema_fast = ta.ema(data, length=fast)
            ema_slow = ta.ema(data, length=slow)
            if ema_fast is not None and ema_slow is not None:
                return ema_fast - ema_slow
            else:
                return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def ao(high: pd.Series, low: pd.Series) -> pd.Series:
        """Awesome Oscillator"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        result = ta.ao(high=high, low=low)
        if result is None or result.empty:
            # フォールバック: 簡易実装 (SMAの差分)
            sma5 = ta.sma((high + low) / 2, length=5)
            sma34 = ta.sma((high + low) / 2, length=34)
            if sma5 is not None and sma34 is not None:
                return sma5 - sma34
            else:
                return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    # 後方互換性のためのエイリアス
    @staticmethod
    def macdext(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD拡張版（標準MACDで代替）"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        return MomentumIndicators.macd(data, fast=fast, slow=slow, signal=signal)

    @staticmethod
    def macdfix(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD固定版（標準MACDで代替）"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        return MomentumIndicators.macd(data, fast=fast, slow=slow, signal=signal)

    @staticmethod
    def stochf(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k: int = 14,
        d: int = 3,
        smooth_k: int = 3,
    ):
        """高速ストキャスティクス（標準ストキャスティクスで代替）"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        return MomentumIndicators.stoch(high, low, close, k=k, d=d, smooth_k=smooth_k)

    @staticmethod
    def cmo(data: pd.Series, length: int = 14) -> pd.Series:
        """チェンジモメンタムオシレーター"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        try:
            result = ta.cmo(data, length=length)
            if result is None or (hasattr(result, "empty") and result.empty):
                return pd.Series(np.full(len(data), np.nan), index=data.index)
            return result
        except (AttributeError, TypeError):
            return pd.Series(np.full(len(data), np.nan), index=data.index)

    @staticmethod
    def trix(data: pd.Series, length: int = 30) -> pd.Series:
        """TRIX"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        result = ta.trix(data, length=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result.iloc[:, 0] if len(result.columns) > 1 else result

    @staticmethod
    def kdj(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k: int = 14,
        d: int = 3,
        j_scalar: float = 3.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """KDJ指標（ストキャスティクスベース）"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        # ストキャスティクスを計算してKDJを導出
        stoch_k, stoch_d = MomentumIndicators.stoch(high, low, close, k=k, d=d)
        j_vals = j_scalar * stoch_k - 2 * stoch_d
        return stoch_k, stoch_d, j_vals

    @staticmethod
    def stochrsi(
        data: pd.Series,
        length: int = 14,
        k_period: int = 5,
        d_period: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """ストキャスティクスRSI"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        result = ta.stochrsi(data, length=length, k_period=k_period, d_period=d_period)
        if result is None:
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return nan_series, nan_series
        return result.iloc[:, 0], result.iloc[:, 1]

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
        if result is None:
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return nan_series, nan_series, nan_series
        return (
            result.iloc[:, 0],  # PPO
            result.iloc[:, 1],  # Histogram
            result.iloc[:, 2],  # Signal
        )

    @staticmethod
    def rvgi(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> Tuple[pd.Series, pd.Series]:
        """Relative Vigor Index"""
        if not isinstance(open_, pd.Series):
            raise TypeError("open_ must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.rvgi(
            open_=open_,
            high=high,
            low=low,
            close=close,
            length=length,
        )
        if result is None:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series
        return result.iloc[:, 0], result.iloc[:, 1]

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
    def smi(
        data: pd.Series,
        fast: int = 13,
        slow: int = 25,
        signal: int = 2,
    ) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Momentum Index"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        try:
            result = ta.smi(data, fast=fast, slow=slow, signal=signal)
            if result is None:
                nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
                return nan_series, nan_series
            return result.iloc[:, 0], result.iloc[:, 1]
        except Exception:
            # フォールバック: RSIとそのEMAシグナル
            rsi = ta.rsi(data, length=max(5, min(slow, len(data) - 1)))
            if rsi is None:
                nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
                return nan_series, nan_series
            signal_ema = rsi.ewm(span=signal).mean()
            return rsi, signal_ema

    @staticmethod
    def kst(
        data: pd.Series,
        r1: int = 10,
        r2: int = 15,
        r3: int = 20,
        r4: int = 30,
        n1: int = 10,
        n2: int = 10,
        n3: int = 10,
        n4: int = 15,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series]:
        """Know Sure Thing"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        result = ta.kst(
            data,
            r1=r1,
            r2=r2,
            r3=r3,
            r4=r4,
            n1=n1,
            n2=n2,
            n3=n3,
            n4=n4,
            signal=signal,
        )
        if result is None:
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return nan_series, nan_series
        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    def stc(
        data: pd.Series,
        tclength: int = 10,
        fast: int = 23,
        slow: int = 50,
        factor: float = 0.5,
        **kwargs,
    ) -> pd.Series:
        """Schaff Trend Cycle"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        try:
            result = ta.stc(
                data, tclength=tclength, fast=fast, slow=slow, factor=factor
            )
            if result is not None:
                return result
        except Exception:
            pass

        # フォールバック: EMAベースの簡易実装
        if len(data) < max(tclength, fast, slow):
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # EMAの組み合わせで近似
        ema_fast = ta.ema(data, length=fast)
        ema_slow = ta.ema(data, length=slow)

        if ema_fast is None or ema_slow is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # MACDのようなシグナルを生成
        stc_values = np.full(len(data), np.nan)
        macd = ema_fast - ema_slow
        signal = ta.ema(macd, length=tclength)

        if signal is not None:
            # スケーリング（0-100の範囲に収める）
            macd_signal_ratio = (macd - signal) / data.std()
            stc_values = 50 + 50 * np.tanh(macd_signal_ratio)

        return pd.Series(stc_values, index=data.index)

    @staticmethod
    def aroonosc(
        high: pd.Series,
        low: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """アルーンオシレーター"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        result = ta.aroon(high=high, low=low, length=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        # アルーンオシレーター = アルーンアップ - アルーンダウン
        return result.iloc[:, 1] - result.iloc[:, 0]

    @staticmethod
    def dx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """Directional Movement Index (DX)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.adx(high=high, low=low, close=close, length=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        # DX列を探す
        dx_col = next((col for col in result.columns if "DX" in col), None)
        if dx_col:
            return result[dx_col]
        else:
            # フォールバック: DMP - DMNの差分
            dmp_col = next((col for col in result.columns if "DMP" in col), None)
            dmn_col = next((col for col in result.columns if "DMN" in col), None)
            if dmp_col and dmn_col:
                return result[dmp_col] - result[dmn_col]
            else:
                return pd.Series(np.full(len(high), np.nan), index=high.index)

    @staticmethod
    def plus_di(high, low, close, length: int = 14) -> pd.Series:
        """Plus Directional Indicator (DI)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.adx(high=high, low=low, close=close, length=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        dmp_col = next(
            (col for col in result.columns if "DMP" in col), result.columns[1]
        )
        return result[dmp_col]

    @staticmethod
    def minus_di(high, low, close, length: int = 14) -> pd.Series:
        """Minus Directional Indicator (DI)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.adx(high=high, low=low, close=close, length=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        dmn_col = next(
            (col for col in result.columns if "DMN" in col), result.columns[2]
        )
        return result[dmn_col]

    @staticmethod
    def plus_dm(high, low, length: int = 14) -> pd.Series:
        """Plus Directional Movement (DM)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        result = ta.dm(high=high, low=low, length=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        dmp_col = next(
            (col for col in result.columns if "DMP" in col), result.columns[0]
        )
        return result[dmp_col]

    @staticmethod
    def minus_dm(high, low, length: int = 14) -> pd.Series:
        """Minus Directional Movement (DM)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        result = ta.dm(high=high, low=low, length=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        dmn_col = next(
            (col for col in result.columns if "DMN" in col), result.columns[1]
        )
        return result[dmn_col]

    @staticmethod
    def ultosc(
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
            high=high,
            low=low,
            close=close,
            fast=fast,
            medium=medium,
            slow=slow,
        )
        if result is None or result.empty:
            # フォールバック: 簡易実装 (weighted average of different periods)
            n = len(high)
            if n < slow:
                return pd.Series(np.full(n, np.nan), index=high.index)

            # 単純な加重平均で近似
            weights = np.array([1, 2, 4])  # fast, medium, slowの重み
            weights = weights / weights.sum()

            fast_ma = ta.sma(close, length=fast)
            medium_ma = ta.sma(close, length=medium)
            slow_ma = ta.sma(close, length=slow)

            if fast_ma is not None and medium_ma is not None and slow_ma is not None:
                # 単純な平均で代替
                return (fast_ma + medium_ma + slow_ma) / 3
            else:
                return pd.Series(np.full(n, np.nan), index=high.index)
        return result

    @staticmethod
    def bop(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
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

        try:
            result = ta.bop(
                open_=open_,
                high=high,
                low=low,
                close=close,
                scalar=1,
            )
        except TypeError:
            result = ta.bop(
                open_=open_,
                high=high,
                low=low,
                close=close,
            )
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def adxr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """ADX評価"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        try:
            result = ta.adxr(high=high, low=low, close=close, length=length)
            if result is None:
                return pd.Series(np.full(len(high), np.nan), index=high.index)
            return result
        except Exception:
            # フォールバック: ADXを返す
            result = ta.adx(high=high, low=low, close=close, length=length)
            if result is None:
                return pd.Series(np.full(len(high), np.nan), index=high.index)
            adx_col = next(
                (col for col in result.columns if "ADX" in col), result.columns[0]
            )
            return result[adx_col]

    # 残りの必要なメソッド（簡素化版）
    @staticmethod
    def rocp(data: pd.Series, length: int = 10) -> pd.Series:
        """変化率（%）"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        try:
            return ta.rocp(data, length=length)
        except AttributeError:
            return ta.roc(data, length=length)

    @staticmethod
    def rocr(data: pd.Series, length: int = 10) -> pd.Series:
        """変化率（比率）"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        try:
            return ta.rocr(data, length=length)
        except AttributeError:
            shifted = data.shift(length)
            return data / shifted

    @staticmethod
    def rocr100(data: pd.Series, length: int = 10) -> pd.Series:
        """変化率（比率100スケール）"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        try:
            return ta.rocr(data, length=length, scalar=100)
        except AttributeError:
            shifted = data.shift(length)
            return (data / shifted) * 100

    @staticmethod
    def rsi_ema_cross(
        data: pd.Series, rsi_length: int = 14, ema_length: int = 9, **kwargs
    ) -> Tuple[pd.Series, pd.Series]:
        """RSI EMAクロス"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        try:
            rsi = ta.rsi(data, length=rsi_length)
            if rsi is not None:
                ema = ta.ema(rsi, length=ema_length)
                if ema is not None:
                    return rsi, ema
        except Exception:
            pass

        # フォールバック: 簡易実装
        n = len(data)
        if n < max(rsi_length, ema_length):
            return pd.Series(np.full(n, np.nan), index=data.index), pd.Series(
                np.full(n, np.nan), index=data.index
            )

        # RSIの簡易計算
        rsi_values = np.full(n, np.nan)
        for i in range(rsi_length - 1, n):
            window = data.iloc[i - rsi_length + 1 : i + 1]
            gains = window.diff()[1:]
            avg_gain = gains[gains > 0].mean() if len(gains[gains > 0]) > 0 else 0
            avg_loss = -gains[gains < 0].mean() if len(gains[gains < 0]) > 0 else 0
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100 - (100 / (1 + rs))
            else:
                rsi_values[i] = 100

        # EMAの簡易計算
        ema_values = np.full(n, np.nan)
        ema_values[ema_length - 1] = rsi_values[
            rsi_length - 1 : rsi_length + ema_length - 1
        ].mean()
        alpha = 2.0 / (ema_length + 1)

        for i in range(ema_length, n):
            if not np.isnan(rsi_values[i]):
                ema_values[i] = alpha * rsi_values[i] + (1 - alpha) * ema_values[i - 1]

        return pd.Series(rsi_values, index=data.index), pd.Series(
            ema_values, index=data.index
        )

    @staticmethod
    def tsi(data: pd.Series, fast: int = 13, slow: int = 25) -> pd.Series:
        """True Strength Index"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        result = ta.tsi(data, fast=fast, slow=slow)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result.iloc[:, 0]

    @staticmethod
    def rvi(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """Relative Volatility Index"""
        if not isinstance(open_, pd.Series):
            raise TypeError("open_ must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        # RVIの簡易実装（RSIベース）
        return ta.rsi(close, length=length)

    @staticmethod
    def pvo(
        volume: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series]:
        """Price Volume Oscillator"""
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        # PVOの簡易実装（ボリュームのMACD）
        result = ta.macd(volume, fast=fast, slow=slow, signal=signal)
        if result is None:
            nan_series = pd.Series(np.full(len(volume), np.nan), index=volume.index)
            return nan_series, nan_series
        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    def cfo(data: pd.Series, length: int = 9) -> pd.Series:
        """Chande Forecast Oscillator"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        try:
            result = ta.cfo(data, length=length)
            if result is not None and not result.isna().all():
                return result if pd.Series(result) else pd.Series(result)
        except Exception:
            pass

        # フォールバック実装: CFOの簡易計算 (トレンド方向の変化率)
        if len(data) < length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # CFOの近似: 価格変化をSMAで平滑化した指標
        price_change = data.pct_change()
        ema_pc = ta.ema(price_change, length=length // 2)  # 価格変化のEMA
        cfo_values = np.full(len(data), np.nan)

        if ema_pc is not None:
            # CFOの概念: トレンド方向の積分
            cfo_values[length:] = (ema_pc * data.shift(length // 2))[
                : len(data) - length
            ]
            # 100スケールに正規化 (CFOの標準範囲)
            cfo_values = (
                (cfo_values - np.nanmean(cfo_values)) / np.nanstd(cfo_values) * 100
            )

        return pd.Series(cfo_values, index=data.index)

    @staticmethod
    def cti(data: pd.Series, length: int = 20) -> pd.Series:
        """Chande Trend Index"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        try:
            result = ta.cti(data, length=length)
            if result is not None and not result.isna().all():
                return result if pd.Series(result) else pd.Series(result)
        except Exception:
            pass

        # フォールバック実装: RSIベースのCTI近似 (トレンド方向の相関係数)
        if len(data) < length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # CTIの概念: 過去データとの相関係数ベースのアプローチ
        cti_values = np.full(len(data), np.nan)

        for i in range(length, len(data)):
            # 現在値と過去値の相関係数を計算
            recent_values = data.iloc[i - length : i].values
            reference_values = data.iloc[0:length].values

            if len(recent_values) == len(reference_values):
                correlation = np.corrcoef(recent_values, reference_values)[0, 1]
                if not np.isnan(correlation):
                    # 相関係数をCTIスコアに変換 (-100 to 100)
                    cti_values[i] = correlation * 100

        return pd.Series(cti_values, index=data.index)

    @staticmethod
    def rmi(
        data: pd.Series = None,
        length: int = 20,
        mom: int = 20,
        close: pd.Series = None,
    ) -> pd.Series:
        """Relative Momentum Index"""
        if not isinstance(data, pd.Series) and close is not None:
            data = close
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        # RMIの簡易実装（RSIベース）
        return ta.rsi(data, length=length)

    @staticmethod
    def dpo(
        data: pd.Series = None,
        length: int = 20,
        close: pd.Series = None,
    ) -> pd.Series:
        """Detrended Price Oscillator"""
        if not isinstance(data, pd.Series) and close is not None:
            data = close
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        result = ta.dpo(data, length=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def chop(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """Choppiness Index"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.chop(high=high, low=low, close=close, length=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def vortex(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> Tuple[pd.Series, pd.Series]:
        """Vortex Indicator"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.vortex(high=high, low=low, close=close, length=length)
        if result is None:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series
        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    def bias(data: pd.Series, length: int = 26) -> pd.Series:
        """Bias"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        result = ta.bias(data, length=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def brar(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 26,
    ) -> pd.Series:
        """BRAR Index"""
        if not isinstance(open_, pd.Series):
            raise TypeError("open_ must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.brar(
            open_=open_,
            high=high,
            low=low,
            close=close,
            length=length,
        )
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def cg(data: pd.Series, length: int = 10) -> pd.Series:
        """Center of Gravity"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        result = ta.cg(data, length=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def coppock(
        data: pd.Series, length: int = 10, fast: int = 11, slow: int = 14
    ) -> pd.Series:
        """Coppock Curve"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        result = ta.coppock(data, length=length, fast=fast, slow=slow)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def er(data: pd.Series, length: int = 10) -> pd.Series:
        """Efficiency Ratio"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        result = ta.er(data, length=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    def eri(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 13,
    ) -> pd.Series:
        """Elder Ray Index"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.eri(high=high, low=low, close=close, length=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def fisher(
        high: pd.Series,
        low: pd.Series,
        length: int = 9,
        signal: int = 1,
    ) -> Tuple[pd.Series, pd.Series]:
        """Fisher Transform"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        result = ta.fisher(high=high, low=low, length=length, signal=signal)
        if result is None:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series
        return result.iloc[:, 0], result.iloc[:, 1]

    @staticmethod
    def inertia(
        close: pd.Series,
        high: pd.Series = None,
        low: pd.Series = None,
        length: int = 20,
        rvi_length: int = 14,
    ) -> pd.Series:
        """Inertia"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.inertia(
            close=close,
            high=high,
            low=low,
            length=length,
            rvi_length=rvi_length,
        )
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    def pgo(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """Pretty Good Oscillator"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.pgo(high=high, low=low, close=close, length=length)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

    @staticmethod
    def psl(
        close: pd.Series,
        open_: pd.Series = None,
        length: int = 12,
    ) -> pd.Series:
        """Psychological Line"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.psl(close=close, open_=open_, length=length)
        if result is None or result.empty:
            # フォールバック: 簡易実装 (close > openの割合)
            if open_ is not None:
                return (close > open_).rolling(length).mean() * 100
            else:
                return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    def rsx(data: pd.Series, length: int = 14) -> pd.Series:
        """RSX"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        result = ta.rsx(data, length=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
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
            bb_length=bb_length,
            bb_std=bb_std,
            kc_length=kc_length,
            kc_scalar=kc_scalar,
            mom_length=mom_length,
            mom_smooth=mom_smooth,
            use_tr=use_tr,
        )
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        return result

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
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

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
        return result

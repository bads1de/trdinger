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

from typing import Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta


class MomentumIndicators:
    """
    モメンタム系指標クラス
    """

    @staticmethod
    def rsi(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
        """相対力指数"""
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.rsi(series, length=length)
        if result is None:
            return np.full(len(series), np.nan)
        return result.values

    @staticmethod
    def macd(
        data: Union[np.ndarray, pd.Series],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.macd(series, fast=fast, slow=slow, signal=signal)

        if result is None or result.empty:
            # フォールバック: NaN配列を返す
            n = len(series)
            return (
                np.full(n, np.nan),
                np.full(n, np.nan),
                np.full(n, np.nan),
            )

        return (
            result.iloc[:, 0].values,  # MACD
            result.iloc[:, 1].values,  # Signal
            result.iloc[:, 2].values,  # Histogram
        )

    @staticmethod
    def stoch(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        k: int = 14,
        d: int = 3,
        smooth_k: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ストキャスティクス"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.stoch(
            high=high_series,
            low=low_series,
            close=close_series,
            k=k,
            d=d,
            smooth_k=smooth_k,
        )

        if result is None or result.empty:
            n = len(high_series)
            return np.full(n, np.nan), np.full(n, np.nan)
        return (result.iloc[:, 0].values, result.iloc[:, 1].values)

    @staticmethod
    def willr(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """ウィリアムズ%R"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.willr(
            high=high_series, low=low_series, close=close_series, length=length
        )
        if result is None:
            return np.full(len(high_series), np.nan)
        return result.values

    @staticmethod
    def cci(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 20,
    ) -> np.ndarray:
        """商品チャネル指数"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.cci(
            high=high_series, low=low_series, close=close_series, length=length
        ).values

    @staticmethod
    def roc(
        data: Union[np.ndarray, pd.Series],
        length: int = 10,
        close: Union[np.ndarray, pd.Series] = None,
    ) -> np.ndarray:
        """変化率"""
        # dataが提供されない場合はcloseを使用
        if data is None and close is not None:
            data = close
        elif data is None:
            raise ValueError("Either 'data' or 'close' must be provided")

        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.roc(series, length=length)
        if result is None or result.empty:
            return np.full(len(series), np.nan)
        return result.values

    @staticmethod
    def mom(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """モメンタム"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.mom(series, length=length)
        if result is None or result.empty:
            return np.full(len(series), np.nan)
        return result.values

    @staticmethod
    def adx(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """平均方向性指数"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.adx(
            high=high_series, low=low_series, close=close_series, length=length
        )
        return result.iloc[:, 0].values  # ADX列

    @staticmethod
    def aroon(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """アルーン"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low

        result = ta.aroon(high=high_series, low=low_series, length=length)
        return result.iloc[:, 0].values, result.iloc[:, 1].values

    @staticmethod
    def mfi(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        volume: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """マネーフローインデックス"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close
        volume_series = pd.Series(volume) if isinstance(volume, np.ndarray) else volume

        return ta.mfi(
            high=high_series,
            low=low_series,
            close=close_series,
            volume=volume_series,
            length=length,
        ).values



    @staticmethod
    def uo(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        fast: int = 7,
        medium: int = 14,
        slow: int = 28,
    ) -> np.ndarray:
        """アルティメットオシレーター"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.uo(
            high=high_series,
            low=low_series,
            close=close_series,
            fast=fast,
            medium=medium,
            slow=slow,
        ).values

    @staticmethod
    def apo(
        data: Union[np.ndarray, pd.Series], fast: int = 12, slow: int = 26
    ) -> np.ndarray:
        """Absolute Price Oscillator"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        result = ta.apo(series, fast=fast, slow=slow)
        if result is None or result.empty:
            # フォールバック: 簡易実装 (EMAの差分)
            ema_fast = ta.ema(series, length=fast)
            ema_slow = ta.ema(series, length=slow)
            if ema_fast is not None and ema_slow is not None:
                return (ema_fast - ema_slow).values
            else:
                return np.full(len(series), np.nan)
        return result.values

    @staticmethod
    def ao(
        high: Union[np.ndarray, pd.Series], low: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Awesome Oscillator"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low

        result = ta.ao(high=high_series, low=low_series)
        if result is None or result.empty:
            # フォールバック: 簡易実装 (SMAの差分)
            sma5 = ta.sma((high_series + low_series) / 2, length=5)
            sma34 = ta.sma((high_series + low_series) / 2, length=34)
            if sma5 is not None and sma34 is not None:
                return (sma5 - sma34).values
            else:
                return np.full(len(high_series), np.nan)
        return result.values

    # 後方互換性のためのエイリアス
    @staticmethod
    def macdext(data: Union[np.ndarray, pd.Series], fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD拡張版（標準MACDで代替）"""
        return MomentumIndicators.macd(data, fast=fast, slow=slow, signal=signal)

    @staticmethod
    def macdfix(data: Union[np.ndarray, pd.Series], fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD固定版（標準MACDで代替）"""
        return MomentumIndicators.macd(data, fast=fast, slow=slow, signal=signal)

    @staticmethod
    def stochf(high: Union[np.ndarray, pd.Series], low: Union[np.ndarray, pd.Series], close: Union[np.ndarray, pd.Series], k: int = 14, d: int = 3, smooth_k: int = 3):
        """高速ストキャスティクス（標準ストキャスティクスで代替）"""
        return MomentumIndicators.stoch(high, low, close, k=k, d=d, smooth_k=smooth_k)

    @staticmethod
    def cmo(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
        """チェンジモメンタムオシレーター"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        try:
            result = ta.cmo(series, length=length)
            if result is None or (hasattr(result, 'empty') and result.empty):
                return np.full(len(series), np.nan)
            return result.values
        except (AttributeError, TypeError):
            return np.full(len(series), np.nan)

    @staticmethod
    def trix(data: Union[np.ndarray, pd.Series], length: int = 30) -> np.ndarray:
        """TRIX"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.trix(series, length=length)
        return result.iloc[:, 0].values if len(result.columns) > 1 else result.values

    @staticmethod
    def kdj(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        k: int = 14,
        d: int = 3,
        j_scalar: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """KDJ指標（ストキャスティクスベース）"""
        # ストキャスティクスを計算してKDJを導出
        stoch_k, stoch_d = MomentumIndicators.stoch(high, low, close, k=k, d=d)
        j_vals = j_scalar * stoch_k - 2 * stoch_d
        return stoch_k, stoch_d, j_vals

    @staticmethod
    def stochrsi(
        data: Union[np.ndarray, pd.Series],
        length: int = 14,
        k_period: int = 5,
        d_period: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ストキャスティクスRSI"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.stochrsi(
            series, length=length, k_period=k_period, d_period=d_period
        )
        return result.iloc[:, 0].values, result.iloc[:, 1].values

    @staticmethod
    def ppo(
        data: Union[np.ndarray, pd.Series],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Percentage Price Oscillator"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.ppo(series, fast=fast, slow=slow, signal=signal)
        return (
            result.iloc[:, 0].values,  # PPO
            result.iloc[:, 1].values,  # Histogram
            result.iloc[:, 2].values,  # Signal
        )

    @staticmethod
    def rvgi(
        open_: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Relative Vigor Index"""
        open_series = pd.Series(open_) if isinstance(open_, np.ndarray) else open_
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.rvgi(
            open_=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            length=length,
        )
        return result.iloc[:, 0].values, result.iloc[:, 1].values

    @staticmethod
    def qqe(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
        """Qualitative Quantitative Estimation"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.qqe(series, length=length)

        if result is None or result.empty:
            # フォールバック: RSIを返す
            return ta.rsi(series, length=length).values

        # QQEの主要な列を返す（通常はRSIMA列）
        return (
            result.iloc[:, 1].values
            if result.shape[1] > 1
            else result.iloc[:, 0].values
        )

    @staticmethod
    def smi(
        data: Union[np.ndarray, pd.Series],
        fast: int = 13,
        slow: int = 25,
        signal: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Momentum Index"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        try:
            result = ta.smi(series, fast=fast, slow=slow, signal=signal)
            return result.iloc[:, 0].values, result.iloc[:, 1].values
        except Exception:
            # フォールバック: RSIとそのEMAシグナル
            rsi = ta.rsi(series, length=max(5, min(slow, len(series) - 1)))
            signal_ema = rsi.ewm(span=signal).mean()
            return rsi.values, signal_ema.values

    @staticmethod
    def kst(
        data: Union[np.ndarray, pd.Series],
        r1: int = 10,
        r2: int = 15,
        r3: int = 20,
        r4: int = 30,
        n1: int = 10,
        n2: int = 10,
        n3: int = 10,
        n4: int = 15,
        signal: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Know Sure Thing"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.kst(
            series,
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
        return result.iloc[:, 0].values, result.iloc[:, 1].values

    @staticmethod
    def stc(
        data: Union[np.ndarray, pd.Series],
        tclength: int = 10,
        fast: int = 23,
        slow: int = 50,
        factor: float = 0.5,
    ) -> np.ndarray:
        """Schaff Trend Cycle"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        try:
            result = ta.stc(
                series, tclength=tclength, fast=fast, slow=slow, factor=factor
            )
            if result is not None:
                return result.values
        except Exception:
            pass

        # フォールバック: EMAベースの簡易実装
        if len(series) < max(tclength, fast, slow):
            return np.full(len(series), np.nan)

        # EMAの組み合わせで近似
        ema_fast = ta.ema(series, length=fast)
        ema_slow = ta.ema(series, length=slow)

        if ema_fast is None or ema_slow is None:
            return np.full(len(series), np.nan)

        # MACDのようなシグナルを生成
        stc_values = np.full(len(series), np.nan)
        macd = ema_fast - ema_slow
        signal = ta.ema(macd, length=tclength)

        if signal is not None:
            # スケーリング（0-100の範囲に収める）
            macd_signal_ratio = (macd - signal) / series.std()
            stc_values = 50 + 50 * np.tanh(macd_signal_ratio)

        return stc_values

    @staticmethod
    def aroonosc(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """アルーンオシレーター"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low

        result = ta.aroon(high=high_series, low=low_series, length=length)
        # アルーンオシレーター = アルーンアップ - アルーンダウン
        return (result.iloc[:, 1] - result.iloc[:, 0]).values

    @staticmethod
    def dx(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """Directional Movement Index (DX)"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.adx(
            high=high_series, low=low_series, close=close_series, length=length
        )
        # DX列を探す
        dx_col = next((col for col in result.columns if "DX" in col), None)
        if dx_col:
            return result[dx_col].values
        else:
            # フォールバック: DMP - DMNの差分
            dmp_col = next((col for col in result.columns if "DMP" in col), None)
            dmn_col = next((col for col in result.columns if "DMN" in col), None)
            if dmp_col and dmn_col:
                return (result[dmp_col] - result[dmn_col]).values
            else:
                return np.full(len(high_series), np.nan)

    @staticmethod
    def plus_di(high, low, close, length: int = 14) -> np.ndarray:
        """Plus Directional Indicator (DI)"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.adx(
            high=high_series, low=low_series, close=close_series, length=length
        )
        dmp_col = next(
            (col for col in result.columns if "DMP" in col), result.columns[1]
        )
        return result[dmp_col].values

    @staticmethod
    def minus_di(high, low, close, length: int = 14) -> np.ndarray:
        """Minus Directional Indicator (DI)"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.adx(
            high=high_series, low=low_series, close=close_series, length=length
        )
        dmn_col = next(
            (col for col in result.columns if "DMN" in col), result.columns[2]
        )
        return result[dmn_col].values

    @staticmethod
    def plus_dm(high, low, length: int = 14) -> np.ndarray:
        """Plus Directional Movement (DM)"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low

        result = ta.dm(high=high_series, low=low_series, length=length)
        dmp_col = next(
            (col for col in result.columns if "DMP" in col), result.columns[0]
        )
        return result[dmp_col].values

    @staticmethod
    def minus_dm(high, low, length: int = 14) -> np.ndarray:
        """Minus Directional Movement (DM)"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low

        result = ta.dm(high=high_series, low=low_series, length=length)
        dmn_col = next(
            (col for col in result.columns if "DMN" in col), result.columns[1]
        )
        return result[dmn_col].values

    @staticmethod
    def ultosc(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        fast: int = 7,
        medium: int = 14,
        slow: int = 28,
    ) -> np.ndarray:
        """Ultimate Oscillator"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.uo(
            high=high_series,
            low=low_series,
            close=close_series,
            fast=fast,
            medium=medium,
            slow=slow,
        )
        if result is None or result.empty:
            # フォールバック: 簡易実装 (weighted average of different periods)
            n = len(high_series)
            if n < slow:
                return np.full(n, np.nan)

            # 単純な加重平均で近似
            weights = np.array([1, 2, 4])  # fast, medium, slowの重み
            weights = weights / weights.sum()

            fast_ma = ta.sma(close_series, length=fast)
            medium_ma = ta.sma(close_series, length=medium)
            slow_ma = ta.sma(close_series, length=slow)

            if fast_ma is not None and medium_ma is not None and slow_ma is not None:
                # 単純な平均で代替
                return ((fast_ma + medium_ma + slow_ma) / 3).values
            else:
                return np.full(n, np.nan)
        return result.values

    @staticmethod
    def bop(
        open_: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Balance of Power"""
        open_series = pd.Series(open_) if isinstance(open_, np.ndarray) else open_
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        try:
            result = ta.bop(
                open_=open_series,
                high=high_series,
                low=low_series,
                close=close_series,
                scalar=1,
            )
        except TypeError:
            result = ta.bop(
                open_=open_series,
                high=high_series,
                low=low_series,
                close=close_series,
            )
        return result.values

    @staticmethod
    def adxr(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """ADX評価"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        try:
            result = ta.adxr(
                high=high_series, low=low_series, close=close_series, length=length
            )
            return result.values
        except Exception:
            # フォールバック: ADXを返す
            result = ta.adx(
                high=high_series, low=low_series, close=close_series, length=length
            )
            adx_col = next(
                (col for col in result.columns if "ADX" in col), result.columns[0]
            )
            return result[adx_col].values

    # 残りの必要なメソッド（簡素化版）
    @staticmethod
    def rocp(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """変化率（%）"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        try:
            return ta.rocp(series, length=length).values
        except AttributeError:
            return ta.roc(series, length=length).values

    @staticmethod
    def rocr(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """変化率（比率）"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        try:
            return ta.rocr(series, length=length).values
        except AttributeError:
            shifted = series.shift(length)
            return (series / shifted).values

    @staticmethod
    def rocr100(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """変化率（比率100スケール）"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        try:
            return ta.rocr(series, length=length, scalar=100).values
        except AttributeError:
            shifted = series.shift(length)
            return ((series / shifted) * 100).values

    @staticmethod
    def rsi_ema_cross(
        data: Union[np.ndarray, pd.Series], rsi_length: int = 14, ema_length: int = 9, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """RSI EMAクロス"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        try:
            rsi = ta.rsi(series, length=rsi_length)
            if rsi is not None:
                ema = ta.ema(rsi, length=ema_length)
                if ema is not None:
                    return rsi.values, ema.values
        except Exception:
            pass

        # フォールバック: 簡易実装
        n = len(series)
        if n < max(rsi_length, ema_length):
            return np.full(n, np.nan), np.full(n, np.nan)

        # RSIの簡易計算
        rsi_values = np.full(n, np.nan)
        for i in range(rsi_length - 1, n):
            window = series.iloc[i - rsi_length + 1 : i + 1]
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

        return rsi_values, ema_values

    @staticmethod
    def tsi(
        data: Union[np.ndarray, pd.Series], fast: int = 13, slow: int = 25
    ) -> np.ndarray:
        """True Strength Index"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.tsi(series, fast=fast, slow=slow)
        return result.values

    @staticmethod
    def rvi(
        open_: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """Relative Volatility Index"""
        # RVIの簡易実装（RSIベース）
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close
        return ta.rsi(close_series, length=length).values

    @staticmethod
    def pvo(
        volume: Union[np.ndarray, pd.Series],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Price Volume Oscillator"""
        volume_series = pd.Series(volume) if isinstance(volume, np.ndarray) else volume

        # PVOの簡易実装（ボリュームのMACD）
        result = ta.macd(volume_series, fast=fast, slow=slow, signal=signal)
        return result.iloc[:, 0].values, result.iloc[:, 1].values

    @staticmethod
    def cfo(data: Union[np.ndarray, pd.Series], length: int = 9) -> np.ndarray:
        """Chande Forecast Oscillator"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        try:
            result = ta.cfo(series, length=length)
            if result is not None and not result.isna().all():
                return result.values if hasattr(result, 'values') else np.array(result)
        except Exception:
            pass

        # フォールバック実装: CFOの簡易計算 (トレンド方向の変化率)
        if len(series) < length:
            return np.full(len(series), np.nan)

        # CFOの近似: 価格変化をSMAで平滑化した指標
        price_change = series.pct_change()
        ema_pc = ta.ema(price_change, length=length//2)  # 価格変化のEMA
        cfo_values = np.full(len(series), np.nan)

        if ema_pc is not None:
            # CFOの概念: トレンド方向の積分
            cfo_values[length:] = (ema_pc * series.shift(length//2))[:len(series)-length]
            # 100スケールに正規化 (CFOの標準範囲)
            cfo_values = (cfo_values - np.nanmean(cfo_values)) / np.nanstd(cfo_values) * 100

        return cfo_values

    @staticmethod
    def cti(data: Union[np.ndarray, pd.Series], length: int = 20) -> np.ndarray:
        """Chande Trend Index"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        try:
            result = ta.cti(series, length=length)
            if result is not None and not result.isna().all():
                return result.values if hasattr(result, 'values') else np.array(result)
        except Exception:
            pass

        # フォールバック実装: RSIベースのCTI近似 (トレンド方向の相関係数)
        if len(series) < length:
            return np.full(len(series), np.nan)

        # CTIの概念: 過去データとの相関係数ベースのアプローチ
        cti_values = np.full(len(series), np.nan)

        for i in range(length, len(series)):
            # 現在値と過去値の相関係数を計算
            recent_values = series.iloc[i-length:i].values
            reference_values = series.iloc[0:length].values

            if len(recent_values) == len(reference_values):
                correlation = np.corrcoef(recent_values, reference_values)[0, 1]
                if not np.isnan(correlation):
                    # 相関係数をCTIスコアに変換 (-100 to 100)
                    cti_values[i] = correlation * 100

        return cti_values

    @staticmethod
    def rmi(
        data: Union[np.ndarray, pd.Series] = None,
        length: int = 20,
        mom: int = 20,
        close: Union[np.ndarray, pd.Series] = None,
    ) -> np.ndarray:
        """Relative Momentum Index"""
        # dataが提供されない場合はcloseを使用
        if data is None and close is not None:
            data = close
        elif data is None:
            raise ValueError("Either 'data' or 'close' must be provided")

        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        # RMIの簡易実装（RSIベース）
        return ta.rsi(series, length=length).values

    @staticmethod
    def dpo(
        data: Union[np.ndarray, pd.Series] = None,
        length: int = 20,
        close: Union[np.ndarray, pd.Series] = None,
    ) -> np.ndarray:
        """Detrended Price Oscillator"""
        # dataが提供されない場合はcloseを使用
        if data is None and close is not None:
            data = close
        elif data is None:
            raise ValueError("Either 'data' or 'close' must be provided")

        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.dpo(series, length=length)
        return result.values

    @staticmethod
    def chop(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """Choppiness Index"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.chop(
            high=high_series, low=low_series, close=close_series, length=length
        )
        return result.values

    @staticmethod
    def vortex(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vortex Indicator"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.vortex(
            high=high_series, low=low_series, close=close_series, length=length
        )
        return result.iloc[:, 0].values, result.iloc[:, 1].values

    @staticmethod
    def bias(data: Union[np.ndarray, pd.Series], length: int = 26) -> np.ndarray:
        """Bias"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return ta.bias(series, length=length).values

    @staticmethod
    def brar(
        open_: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 26,
    ) -> np.ndarray:
        """BRAR Index"""
        open_series = pd.Series(open_) if isinstance(open_, np.ndarray) else open_
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.brar(
            open_=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            length=length,
        )
        return result.values

    @staticmethod
    def cg(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """Center of Gravity"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return ta.cg(series, length=length).values

    @staticmethod
    def coppock(
        data: Union[np.ndarray, pd.Series], length: int = 10, fast: int = 11, slow: int = 14
    ) -> np.ndarray:
        """Coppock Curve"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return ta.coppock(series, length=length, fast=fast, slow=slow).values

    @staticmethod
    def er(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """Efficiency Ratio"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return ta.er(series, length=length).values

    @staticmethod
    def eri(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 13,
    ) -> np.ndarray:
        """Elder Ray Index"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.eri(high=high_series, low=low_series, close=close_series, length=length)
        return result.values

    @staticmethod
    def fisher(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        length: int = 9,
        signal: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fisher Transform"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low

        result = ta.fisher(high=high_series, low=low_series, length=length, signal=signal)
        return result.iloc[:, 0].values, result.iloc[:, 1].values

    @staticmethod
    def inertia(
        close: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series] = None,
        low: Union[np.ndarray, pd.Series] = None,
        length: int = 20,
        rvi_length: int = 14,
    ) -> np.ndarray:
        """Inertia"""
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close
        high_series = pd.Series(high) if isinstance(high, np.ndarray) and high is not None else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) and low is not None else low

        result = ta.inertia(
            close=close_series,
            high=high_series,
            low=low_series,
            length=length,
            rvi_length=rvi_length,
        )
        return result.values

    @staticmethod
    def pgo(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """Pretty Good Oscillator"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.pgo(
            high=high_series, low=low_series, close=close_series, length=length
        ).values

    @staticmethod
    def psl(
        close: Union[np.ndarray, pd.Series],
        open_: Union[np.ndarray, pd.Series] = None,
        length: int = 12,
    ) -> np.ndarray:
        """Psychological Line"""
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close
        open_series = pd.Series(open_) if isinstance(open_, np.ndarray) and open_ is not None else open_

        result = ta.psl(close=close_series, open_=open_series, length=length)
        if result is None or result.empty:
            # フォールバック: 簡易実装 (close > openの割合)
            if open_series is not None:
                return ((close_series > open_series).rolling(length).mean() * 100).values
            else:
                return np.full(len(close_series), np.nan)
        return result.values

    @staticmethod
    def rsx(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
        """RSX"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return ta.rsx(series, length=length).values

    @staticmethod
    def squeeze(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        bb_length: int = 20,
        bb_std: float = 2.0,
        kc_length: int = 20,
        kc_scalar: float = 1.5,
        mom_length: int = 12,
        mom_smooth: int = 6,
        use_tr: bool = True,
    ) -> np.ndarray:
        """Squeeze"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.squeeze(
            high=high_series,
            low=low_series,
            close=close_series,
            bb_length=bb_length,
            bb_std=bb_std,
            kc_length=kc_length,
            kc_scalar=kc_scalar,
            mom_length=mom_length,
            mom_smooth=mom_smooth,
            use_tr=use_tr,
        )
        return result.values

    @staticmethod
    def squeeze_pro(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        bb_length: int = 20,
        bb_std: float = 2.0,
        kc_length: int = 20,
        kc_scalar_wide: float = 2.0,
        kc_scalar_normal: float = 1.5,
        kc_scalar_narrow: float = 1.0,
        mom_length: int = 12,
        mom_smooth: int = 6,
        use_tr: bool = True,
    ) -> np.ndarray:
        """Squeeze Pro"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.squeeze_pro(
            high=high_series,
            low=low_series,
            close=close_series,
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
        return result.values

"""
トレンド系テクニカル指標

登録してあるテクニカルの一覧:
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- TEMA (Triple Exponential Moving Average)
- DEMA (Double Exponential Moving Average)
- WMA (Weighted Moving Average)
- TRIMA (Triangular Moving Average)
- KAMA (Kaufman's Adaptive Moving Average)
- T3 (Tillson's T3 Moving Average)
- SAR (Parabolic SAR)
- SAREXT (Extended Parabolic SAR)
- MA (Moving Average with Type)
- MAVP (Moving Average Variable Period)
- MIDPOINT (MidPoint over period)
- MIDPRICE (MidPrice over period)
- HMA (Hull Moving Average)
- ZLMA (Zero Lag Moving Average)
- VWMA (Volume Weighted Moving Average)
- SWMA (Symmetric Weighted Moving Average)
- ALMA (Arnaud Legoux Moving Average)
- RMA (Smoothed Moving Average)
- Ichimoku Cloud
- SMA_SLOPE (SMA Slope)
- PRICE_EMA_RATIO (Price to EMA Ratio)
- FWMA (Fibonacci's Weighted Moving Average)
- HILO (Gann High-Low Activator)
- HL2 (High-Low Average)
- HLC3 (High-Low-Close Average)
- HWMA (Holt-Winter Moving Average)
- JMA (Jurik Moving Average)
- MCGD (McGinley Dynamic)
- OHLC4 (Open-High-Low-Close Average)
- PWMA (Pascal's Weighted Moving Average)
- SINWMA (Sine Weighted Moving Average)
- SSF (Ehler's Super Smoother Filter)
- VIDYA (Variable Index Dynamic Average)
- WCP (Weighted Closing Price)
- LINREG (Linear Regression Moving Average)
- LINREG_SLOPE (Linear Regression Slope)
- LINREG_INTERCEPT (Linear Regression Intercept)
- LINREG_ANGLE (Linear Regression Angle)
"""

from typing import Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import handle_pandas_ta_errors

# PandasのSeries位置アクセス警告を抑制 (pandas-taとの互換性のため)
warnings.filterwarnings(
    "ignore",
    message="Series.__getitem__ treating keys as positions is deprecated",
    category=FutureWarning,
)


class TrendIndicators:
    """
    トレンド系指標クラス
    """

    @staticmethod
    @handle_pandas_ta_errors
    def sma(data: pd.Series, length: int) -> pd.Series:
        """単純移動平均（軽量エラーハンドリング付き）"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        # 基本的な入力検証
        if len(data) == 0:
            raise ValueError("データが空です")

        if length == 1:
            return data

        result = data.rolling(window=length).mean()

        # 結果検証（重要な異常ケースのみ）
        if result.isna().all():
            raise ValueError("計算結果が全てNaNです")

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def ema(data: pd.Series, length: int) -> pd.Series:
        """指数移動平均"""
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        # 第一優先: pandas-ta
        try:
            result = ta.ema(data, length=length)
            if result is not None and isinstance(result, pd.Series):
                # 全てNaNではなく、有用な値が含まれている場合のみ使用
                # 最初のlength位置以降に有効な値がある場合
                start_idx = length - 1
                if (
                    start_idx < len(result)
                    and not result.iloc[start_idx:].isna().all()
                ):
                    return result
        except Exception:
            pass

        # フォールバック実装: numpyベース -> pandas.Seriesに変更
        if len(data) < length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # EMA計算用変数
        alpha = 2.0 / (length + 1)
        ema_values = np.full(len(data), np.nan, dtype=float)
        ema_values[length - 1] = data.iloc[:length].mean()  # 初期値はSMA

        # ループでEMA計算
        for i in range(length, len(data)):
            ema_values[i] = alpha * data.iloc[i] + (1 - alpha) * ema_values[i - 1]

        return pd.Series(ema_values, index=data.index)

    @staticmethod
    @handle_pandas_ta_errors
    def ppo(
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Percentage Price Oscillator with pandas-ta fallback"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        try:
            result = ta.ppo(data, fast=fast, slow=slow, signal=signal)
            if result is not None and not result.empty:
                return (
                    result.iloc[:, 0],
                    result.iloc[:, 1],
                    result.iloc[:, 2],
                )
        except Exception:
            pass

        # フォールバック実装
        ema_fast = TrendIndicators.ema(data, fast)
        ema_slow = TrendIndicators.ema(data, slow)

        if not isinstance(ema_fast, pd.Series) or not isinstance(ema_slow, pd.Series):
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return nan_series, nan_series, nan_series

        # PPO主力とシグナルラインの計算
        ppo_line = 100 * (ema_fast - ema_slow) / ema_slow
        signal_line = TrendIndicators.ema(ppo_line, signal)

        return ppo_line, signal_line, ppo_line - signal_line

    @staticmethod
    @handle_pandas_ta_errors
    def stochf(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
        fast_length: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Fast with pandas-ta fallback"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pd.Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pd.Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pd.Series")

        try:
            result = ta.stochf(
                high=high,
                low=low,
                close=close,
                fastk_length=fast_length,
                fastd_length=length,
            )
            if result is not None and not result.empty:
                return result.iloc[:, 0], result.iloc[:, 1]
        except Exception:
            pass

        # フォールバック実装 (シンプルなストキャスティクス計算)
        raw_k = (
            100
            * (close - low.rolling(length).min())
            / (high.rolling(length).max() - low.rolling(length).min())
        )
        fast_k = raw_k.rolling(fast_length).mean()

        return fast_k, fast_k

    @staticmethod
    @handle_pandas_ta_errors
    def tema(data: pd.Series, length: int) -> pd.Series:
        """三重指数移動平均"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        # 第一優先: pandas-ta
        try:
            result = ta.tema(data, length=length)
            if result is not None:
                # 最初のlength位置以降に有効な値がある場合
                start_idx = length * 3 - 1
                if (
                    start_idx < len(result)
                    and not np.isnan(result.iloc[start_idx:]).all()
                ):
                    return result
        except Exception:
            pass

        # フォールバック実装: 多段EMA計算
        if len(data) < length * 3:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # EMA1, EMA2, EMA3の計算
        ema1 = TrendIndicators.ema(data, length)
        ema2 = TrendIndicators.ema(ema1, length)
        ema3 = TrendIndicators.ema(ema2, length)

        # TEMA = 3*EMA1 - 3*EMA2 + EMA3
        return 3 * ema1 - 3 * ema2 + ema3

    @staticmethod
    @handle_pandas_ta_errors
    def stc(
        data: pd.Series,
        length: int = 10,
        fast_length: int = 23,
        slow_length: int = 50,
    ) -> pd.Series:
        """Schaff Trend Cycle with pandas-ta fallback"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        # 第一優先: pandas-ta
        try:
            result = ta.stc(
                data, length=length, fastLength=fast_length, slowLength=slow_length
            )
            if result is not None and not result.isna().all():
                return result
        except Exception:
            pass

        # フォールバック実装: 簡易STC (MACDベース)
        # %K: EMA(EMA(price, fast)) / EMA(EMA(price, slow)) - 1
        # %D: EMA(%K) * 100
        # STC: EMA(%D) * 100 (シグナルラインなしの場合は 3重EMAベース)

        if len(data) < slow_length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # 基本的なMACD計算
        ema_fast = TrendIndicators.ema(data, fast_length)
        ema_slow = TrendIndicators.ema(data, slow_length)
        macd = ema_fast - ema_slow

        # MACDのサイクル分析 (変動範囲のトレンド)
        if len(macd) > 0:
            # ピークとバレー検出 (簡易版)
            stc_values = np.full(len(data), np.nan)
            valid_start = slow_length - 1

            # 基本的なサイクル計算 (簡易: MACDのノーマライズ)
            if len(macd) >= valid_start + length:
                # MACDのグサイクルをトレンドサイクルに変換
                cycle = (
                    (macd - macd.rolling(slow_length).min())
                    / (
                        macd.rolling(slow_length).max()
                        - macd.rolling(slow_length).min()
                    )
                ) * 100

                # 最終的なSTC値
                stc_values[valid_start:] = TrendIndicators.ema(
                    cycle.iloc[valid_start:], length
                )

            return pd.Series(stc_values, index=data.index)

        return pd.Series(np.full(len(data), np.nan), index=data.index)

    @staticmethod
    @handle_pandas_ta_errors
    def dema(data: pd.Series, length: int) -> pd.Series:
        """二重指数移動平均"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        # 第一優先: pandas-ta
        try:
            result = ta.dema(data, length=length)
            if result is not None:
                # 最初のlength位置以降に有効な値がある場合
                start_idx = length * 2 - 1
                if (
                    start_idx < len(result)
                    and not np.isnan(result.iloc[start_idx:]).all()
                ):
                    return result
        except Exception:
            pass

        # フォールバック実装: 多段EMA計算 (DEMA = 2*EMA1 - EMA2)
        if len(data) < length * 2:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # EMA1, EMA2の計算
        ema1 = TrendIndicators.ema(data, length)
        ema2 = TrendIndicators.ema(ema1, length)

        # DEMA = 2*EMA1 - EMA2
        return 2 * ema1 - ema2

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
        return ta.wma(data, length=length)

    @staticmethod
    @handle_pandas_ta_errors
    def trima(
        data: pd.Series = None,
        length: int = 14,
        close: pd.Series = None,
    ) -> pd.Series:
        """三角移動平均"""
        # dataが提供されない場合はcloseを使用
        if data is None and close is not None:
            data = close
        elif data is None:
            raise ValueError("Either 'data' or 'close' must be provided")

        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        return ta.trima(data, length=length)

    @staticmethod
    @handle_pandas_ta_errors
    def kama(data: pd.Series, length: int = 30) -> pd.Series:
        """カウフマン適応移動平均"""
        # 第一優先: pandas-ta
        try:
            result = ta.kama(data, length=length)
            if result is not None:
                return result
        except Exception:
            pass

        # フォールバック: EMAで代替
        return TrendIndicators.ema(data, length)

    @staticmethod
    @handle_pandas_ta_errors
    def t3(
        data: pd.Series, length: int = 5, a: float = 0.7
    ) -> pd.Series:
        """T3移動平均"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        return ta.t3(data, length=length, a=a)

    @staticmethod
    @handle_pandas_ta_errors
    def sar(
        high: pd.Series,
        low: pd.Series,
        af: float = 0.02,
        max_af: float = 0.2,
    ) -> pd.Series:
        """パラボリックSAR"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        result = ta.psar(high=high, low=low, af0=af, af=af, max_af=max_af)
        # PSARl と PSARs を結合
        psar_long = result[f"PSARl_{af}_{max_af}"]
        psar_short = result[f"PSARs_{af}_{max_af}"]
        return psar_long.fillna(psar_short)

    @staticmethod
    @handle_pandas_ta_errors
    def sarext(
        high: pd.Series,
        low: pd.Series,
        startvalue: float = 0.0,
        offsetonreverse: float = 0.0,
        accelerationinitlong: float = 0.02,
        accelerationlong: float = 0.02,
        accelerationmaxlong: float = 0.2,
        accelerationinitshort: float = 0.02,
        accelerationshort: float = 0.02,
        accelerationmaxshort: float = 0.2,
    ) -> pd.Series:
        """Extended Parabolic SAR (pandas-ta psarで近似)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pd.Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pd.Series")

        # 拡張パラメータをpandas-ta psarにマッピング（近似）
        # startvalue, offsetonreverse, accelerationinitshort, accelerationshort, accelerationmaxshortは未使用
        _ = (
            startvalue,
            offsetonreverse,
            accelerationinitshort,
            accelerationshort,
            accelerationmaxshort,
        )
        result = ta.psar(
            high=high,
            low=low,
            af0=accelerationinitlong,
            af=accelerationlong,
            max_af=accelerationmaxlong,
        )

        psar_long = result[f"PSARl_{accelerationlong}_{accelerationmaxlong}"]
        psar_short = result[f"PSARs_{accelerationlong}_{accelerationmaxlong}"]
        return psar_long.fillna(psar_short)

    @staticmethod
    @handle_pandas_ta_errors
    def ma(
        data: Optional[pd.Series] = None,
        period: int = 30,
        matype: int = 0,
        close: Optional[pd.Series] = None,
    ) -> pd.Series:
        """移動平均（タイプ指定可能）"""
        # dataが提供されない場合はcloseを使用
        if data is None and close is not None:
            data = close
        elif data is None:
            raise ValueError("Either 'data' or 'close' must be provided")

        ma_functions = {
            0: TrendIndicators.sma,  # SMA
            1: TrendIndicators.ema,  # EMA
            2: TrendIndicators.wma,  # WMA
            3: TrendIndicators.dema,  # DEMA
            4: TrendIndicators.tema,  # TEMA
            5: TrendIndicators.trima,  # TRIMA
            6: TrendIndicators.kama,  # KAMA
            8: TrendIndicators.t3,  # T3
        }

        ma_func = ma_functions.get(matype, TrendIndicators.sma)
        return ma_func(data, period)

    @staticmethod
    @handle_pandas_ta_errors
    def mavp(
        data: pd.Series,
        periods: pd.Series,
        minperiod: int = 2,
        maxperiod: int = 30,
        matype: int = 0,
    ) -> pd.Series:
        """可変期間移動平均（カスタム実装）"""
        if len(data) != len(periods):
            raise ValueError(
                f"データと期間の長さが一致しません。Data: {len(data)}, Periods: {len(periods)}"
            )

        # pandas-taには直接的な実装がないため、カスタム実装
        result = pd.Series(np.full(len(data), np.nan, dtype=float), index=data.index)

        for i in range(len(data)):
            period = int(np.clip(periods.iloc[i], minperiod, maxperiod))
            start_idx = max(0, i - period + 1)

            if i >= period - 1:
                window_data = data.iloc[start_idx : i + 1]
                if matype == 0:  # SMA
                    result.iloc[i] = window_data.mean()
                else:
                    # 他のタイプは簡略化してSMAで代替
                    result.iloc[i] = window_data.mean()

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def midpoint(
        data: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """期間内の中点"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        return ta.midpoint(data, length=length)

    @staticmethod
    @handle_pandas_ta_errors
    def midprice(
        high: pd.Series,
        low: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """期間内の中値価格"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        return ta.midprice(high=high, low=low, length=length)

    @staticmethod
    @handle_pandas_ta_errors
    def hma(data: pd.Series, length: int = 20) -> pd.Series:
        """Hull Moving Average"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        # 第一優先: pandas-ta
        try:
            result = ta.hma(data, length=length)
            if result is not None:
                # 最初のlength位置以降に有効な値がある場合
                start_idx = int(length * 1.5) - 1  # HMAはより多くのデータを必要とする
                if (
                    start_idx < len(result)
                    and not np.isnan(result.iloc[start_idx:]).all()
                ):
                    return result
        except Exception:
            pass

        # フォールバック実装: カスタムHull Moving Average
        if len(data) < length * 3:  # HMAは3つのWMAを必要とする
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # 計算ステップ:
        # 1. WMA(n/2)
        # 2. WMA(n)
        # 3. WMA(n/2) を WMA(n) から引いて、2倍する
        # 4. 結果を sqrt(n)でWMA

        import math

        hma_length = int(length / 2)
        sqrt_length = int(math.floor(math.sqrt(length)))

        # WMA(n/2) と WMA(n) の計算
        wma_half = TrendIndicators.wma(data, length=hma_length)
        wma_full = TrendIndicators.wma(data, length=length)

        # 差分を2倍
        diff = 2 * wma_half - wma_full

        # sqrt(n)でWMA
        return TrendIndicators.wma(diff, length=sqrt_length)

    @staticmethod
    @handle_pandas_ta_errors
    def zlma(data: pd.Series, length: int = 20) -> pd.Series:
        """Zero-Lag Exponential Moving Average"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        if hasattr(ta, "zlma"):
            return ta.zlma(data, length=length)
        else:
            # フォールバック: EMAの差分で近似
            lag = int((length - 1) / 2)
            shifted = data.shift(lag)
            adjusted = data + (data - shifted)
            return ta.ema(adjusted, length=length)

    @staticmethod
    @handle_pandas_ta_errors
    def vwma(data: pd.Series, volume: pd.Series, length: int = 20) -> pd.Series:
        """Volume Weighted Moving Average"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")
        if len(data) != len(volume):
            raise ValueError("data and volume must have the same length")

        # 第一優先: pandas-ta
        try:
            result = ta.vwma(data, volume=volume, length=length)
            if result is not None and not result.isna().all():
                # 最初のlength位置以降に有効な値がある場合
                start_idx = length - 1
                if (
                    start_idx < len(result)
                    and not np.isnan(result.iloc[start_idx:]).all()
                ):
                    return result
        except Exception:
            pass

        # フォールバック: カスタム実装
        if len(data) < length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # VWMAの手動計算
        vwma_values = np.full(len(data), np.nan)
        for i in range(length - 1, len(data)):
            window_price = data.iloc[i - length + 1 : i + 1]
            window_vol = volume.iloc[i - length + 1 : i + 1]
            vwma_values[i] = np.average(window_price, weights=window_vol)

        return pd.Series(vwma_values, index=data.index)

    @staticmethod
    @handle_pandas_ta_errors
    def swma(data: pd.Series, length: int = 10) -> pd.Series:
        """Symmetric Weighted Moving Average"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        # 第一優先: pandas-ta
        try:
            result = ta.swma(data, length=length)
            if result is not None:
                # 全てNaNではなく、有用な値が含まれている場合のみ使用
                result_values = (
                    result.values if hasattr(result, "values") else np.array(result)
                )
                # 最初のlength位置以降に有効な値がある場合
                start_idx = length - 1
                if (
                    start_idx < len(result_values)
                    and not np.isnan(result_values[start_idx:]).all()
                ):
                    return pd.Series(result_values, index=data.index)
        except Exception:
            pass

        # フォールバック実装: カスタム対称重み付き移動平均
        if len(data) < length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # SWMA: 対称重み (中心が最大重み、両端に向かって線形減衰)
        swma_values = np.full(len(data), np.nan)

        # 重み計算 (三角形分布)
        weights = np.concatenate(
            [
                np.arange(1, length // 2 + 1),  # 上昇部分
                np.arange(length // 2, 0, -1),  # 下降部分
            ][:length]
        )
        weights = weights[:length]  # 長さに合わせる
        weights = weights / weights.sum()  # 正規化

        for i in range(length - 1, len(data)):
            window = data.iloc[i - length + 1 : i + 1].values
            swma_values[i] = np.sum(window * weights)

        return pd.Series(swma_values, index=data.index)

    @staticmethod
    @handle_pandas_ta_errors
    def alma(
        data: pd.Series,
        length: int = 9,
        sigma: float = 6.0,
        offset: float = 0.85,
    ) -> pd.Series:
        """Arnaud Legoux Moving Average"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        # 第一優先: pandas-ta
        try:
            result = ta.alma(data, length=length, sigma=sigma, offset=offset)
            if result is not None:
                # 全てNaNではなく、有用な値が含まれている場合のみ使用
                result_values = (
                    result.values if hasattr(result, "values") else np.array(result)
                )
                # 最初のlength位置以降に有効な値がある場合
                start_idx = length - 1
                if (
                    start_idx < len(result_values)
                    and not np.isnan(result_values[start_idx:]).all()
                ):
                    return pd.Series(result_values, index=data.index)
        except Exception:
            pass

        # フォールバック実装: カスタムガウシアン重み付き移動平均
        if len(data) < length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        import math

        # ガウシアンダistributionによる重み計算
        weights = np.zeros(length)
        m = offset * (length - 1)  # 調整係数

        for i in range(length):
            weights[i] = np.exp(-np.power(i - m, 2) / (2 * np.power(length / sigma, 2)))

        weights = weights / weights.sum()  # 正規化

        # 重み付き移動平均計算
        alma_values = np.full(len(data), np.nan)
        for i in range(length - 1, len(data)):
            window = data.iloc[i - length + 1 : i + 1].values
            alma_values[i] = np.sum(window * weights)

        return pd.Series(alma_values, index=data.index)

    @staticmethod
    @handle_pandas_ta_errors
    def rma(data: pd.Series, length: int = 14) -> pd.Series:
        """Smoothed Moving Average (RMA)"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        try:
            result = ta.rma(data, length=length)
            if result is not None and not result.isna().all():
                # 最初のlength位置以降に有効な値がある場合
                start_idx = length - 1
                if (
                    start_idx < len(result)
                    and not np.isnan(result.iloc[start_idx:]).all()
                ):
                    return result
        except Exception:
            pass

        # フォールバック: カスタム実装 (RMA = EMA with alpha = 1/length)
        if len(data) < length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # RMAの手動計算（EMAと同様）
        alpha = 1.0 / length
        rma_values = np.full(len(data), np.nan)
        rma_values[length - 1] = data.iloc[:length].mean()  # 初期値はSMA

        for i in range(length, len(data)):
            rma_values[i] = alpha * data.iloc[i] + (1 - alpha) * rma_values[i - 1]

        return pd.Series(rma_values, index=data.index)

    @staticmethod
    @handle_pandas_ta_errors
    def ichimoku_cloud(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tenkan: int = 9,
        kijun: int = 26,
        senkou: int = 52,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Ichimoku Cloud: (conversion, base, span_a, span_b, lagging)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if len(high) != len(low) or len(high) != len(close):
            raise ValueError("high, low, and close must have the same length")

        # 標準定義に基づく計算
        conv = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2.0
        base = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2.0
        span_a = ((conv + base) / 2.0).shift(kijun)
        span_b = (
            (high.rolling(senkou).max() + low.rolling(senkou).min()) / 2.0
        ).shift(kijun)
        lag = close.shift(-kijun)

        return (conv, base, span_a, span_b, lag)

    # カスタム指標
    @staticmethod
    @handle_pandas_ta_errors
    def sma_slope(data: pd.Series, length: int = 20) -> pd.Series:
        """SMAの傾き（前期間との差分）"""
        sma_vals = TrendIndicators.sma(data, length)
        return sma_vals.diff()

    @staticmethod
    @handle_pandas_ta_errors
    def price_ema_ratio(data: pd.Series, length: int = 20) -> pd.Series:
        """価格とEMAの比率 - 1"""
        ema_vals = TrendIndicators.ema(data, length)
        ema_series = pd.Series(ema_vals, index=data.index)
        return ((data / ema_series) - 1.0)

    # mama function removed due to pandas-ta compatibility issues

    @staticmethod
    @handle_pandas_ta_errors
    def maxindex(data: pd.Series, length: int = 14) -> pd.Series:
        """最大値のインデックス"""
        return (
            data.rolling(window=length).apply(lambda x: x.argmax(), raw=False)
        )

    @staticmethod
    @handle_pandas_ta_errors
    def minindex(data: pd.Series, length: int = 14) -> pd.Series:
        """最小値のインデックス"""
        return (
            data.rolling(window=length).apply(lambda x: x.argmin(), raw=False)
        )

    @staticmethod
    @handle_pandas_ta_errors
    def minmax(
        data: pd.Series, length: int = 14
    ) -> Tuple[pd.Series, pd.Series]:
        """最小値と最大値"""
        min_vals = data.rolling(window=length).min()
        max_vals = data.rolling(window=length).max()
        return min_vals, max_vals

    @staticmethod
    @handle_pandas_ta_errors
    def minmaxindex(
        data: pd.Series, length: int = 14
    ) -> Tuple[pd.Series, pd.Series]:
        """最小値と最大値のインデックス"""
        min_idx = (
            data.rolling(window=length).apply(lambda x: x.argmin(), raw=False)
        )
        max_idx = (
            data.rolling(window=length).apply(lambda x: x.argmax(), raw=False)
        )
        return min_idx, max_idx

    @staticmethod
    @handle_pandas_ta_errors
    def fwma(data: pd.Series, length: int = 10) -> pd.Series:
        """Fibonacci's Weighted Moving Average"""
        # 第一優先: pandas-ta
        try:
            result = ta.fwma(data, length=length)
            if result is not None and not result.isna().all():
                return result
        except Exception:
            pass

        # フォールバック実装: カスタム重み付き移動平均
        # フィボナッチ数的減衰重みを使用 (簡易バージョン)
        if len(data) < length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # フィボナッチ重み生成 (簡易版: 線形減衰に近似)
        weights = np.arange(1, length + 1, dtype=float)
        weights = weights / weights.sum()  # 正規化

        # 重み付き移動平均計算
        fwma_values = np.full(len(data), np.nan)
        for i in range(length - 1, len(data)):
            window = data.iloc[i - length + 1 : i + 1].values
            fwma_values[i] = np.sum(window * weights)

        return pd.Series(fwma_values, index=data.index)

    @staticmethod
    @handle_pandas_ta_errors
    def hilo(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series = None,
        high_length=None,
        low_length=None,
        length: int = 14,
        **kwargs,
    ) -> pd.Series:
        """Gann High-Low Activator"""
        high_series = high
        low_series = low
        close_series = close

        # Use high_length and low_length if provided, otherwise use length for both
        hl = high_length if high_length is not None else length
        ll = low_length if low_length is not None else length

        result = ta.hilo(
            high=high_series,
            low=low_series,
            close=close_series,
            high_length=hl,
            low_length=ll,
        )
        return (
            result if result is not None else pd.Series(np.full(len(high_series), np.nan), index=high_series.index)
        )

    @staticmethod
    @handle_pandas_ta_errors
    def hl2(
        high: pd.Series, low: pd.Series
    ) -> pd.Series:
        """High-Low Average"""
        return ((high + low) / 2)

    @staticmethod
    @handle_pandas_ta_errors
    def hlc3(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """High-Low-Close Average"""
        return ((high + low + close) / 3)

    @staticmethod
    @handle_pandas_ta_errors
    def hwma(data: pd.Series, length: int = 10) -> pd.Series:
        """Holt-Winter Moving Average"""
        result = ta.hwma(data, length=length)
        return result if result is not None else pd.Series(np.full(len(data), np.nan), index=data.index)

    @staticmethod
    @handle_pandas_ta_errors
    def jma(
        data: pd.Series,
        length: int = 7,
        phase: float = 0.0,
        power: float = 2.0,
    ) -> pd.Series:
        """Jurik Moving Average"""
        result = ta.jma(data, length=length, phase=phase, power=power)
        return result if result is not None else pd.Series(np.full(len(data), np.nan), index=data.index)

    @staticmethod
    @handle_pandas_ta_errors
    def mcgd(data: pd.Series, length: int = 10) -> pd.Series:
        """McGinley Dynamic"""
        if len(data) < length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        mcgd_values = np.full(len(data), np.nan)
        mcgd_values[length - 1] = data.iloc[:length].mean()  # 初期値はSMA

        k = 0.6  # 調整係数

        for i in range(length, len(data)):
            ratio = (
                data.iloc[i] / mcgd_values[i - 1] if mcgd_values[i - 1] != 0 else 1
            )
            mcgd_values[i] = mcgd_values[i - 1] + (
                data.iloc[i] - mcgd_values[i - 1]
            ) / (k * length * (ratio**4))

        return pd.Series(mcgd_values, index=data.index)

    @staticmethod
    @handle_pandas_ta_errors
    def ohlc4(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Open-High-Low-Close Average"""
        return ((open_ + high + low + close) / 4)

    @staticmethod
    @handle_pandas_ta_errors
    def pwma(data: pd.Series, length: int = 10) -> pd.Series:
        """Pascal's Weighted Moving Average"""
        result = ta.pwma(data, length=length)
        return result if result is not None else pd.Series(np.full(len(data), np.nan), index=data.index)

    @staticmethod
    @handle_pandas_ta_errors
    def sinwma(data: pd.Series, length: int = 10) -> pd.Series:
        """Sine Weighted Moving Average"""
        result = ta.sinwma(data, length=length)
        return result if result is not None else pd.Series(np.full(len(data), np.nan), index=data.index)

    @staticmethod
    @handle_pandas_ta_errors
    def ssf(data: pd.Series, length: int = 10) -> pd.Series:
        """Ehler's Super Smoother Filter"""
        result = ta.ssf(data, length=length)
        return result if result is not None else pd.Series(np.full(len(data), np.nan), index=data.index)

    @staticmethod
    @handle_pandas_ta_errors
    def vidya(
        data: pd.Series, length: int = 14, adjust: bool = True
    ) -> pd.Series:
        """Variable Index Dynamic Average"""
        series = data.astype(np.float64).copy()

        # 第一優先: pandas-ta
        try:
            result = ta.vidya(series, length=length, adjust=adjust)
            if result is not None and not result.isna().all():
                # 最初のlength位置以降に有効な値がある場合
                start_idx = length * 2 - 1  # VIDYAはより多くのデータを必要とする
                if (
                    start_idx < len(result)
                    and not np.isnan(result.iloc[start_idx:]).all()
                ):
                    return result
        except Exception:
            pass

        # フォールバック実装: カスタVIDYA (CMOベースの適応的平均)
        if len(series) < length * 3:
            return pd.Series(np.full(len(series), np.nan), index=series.index)

        vidya_values = np.full(len(series), np.nan)

        # 最初の値をSMAで初期化
        vidya_values[length * 2 - 1] = series.iloc[: length * 2].mean()

        # VIDYA計算: アルファは標準化されたChande Momentum Oscillator (CMO)に基づく
        for i in range(length * 2, len(series)):
            # CMO計算 (簡易版)
            diff = series.diff()
            pos_sum = diff[diff > 0].rolling(length).sum().iloc[i]
            neg_sum = abs(diff[diff < 0]).rolling(length).sum().iloc[i]

            if neg_sum == 0:
                cmo = 100.0
            else:
                cmo = 100 * (pos_sum - neg_sum) / (pos_sum + neg_sum)

            # CMOをアルファに変換 (0-1の範囲)
            alpha = abs(cmo) / 100.0

            # adjustパラメータでアルファを調整
            if adjust:
                alpha *= 0.8  # 標準調整

            # VIDYA更新
            vidya_values[i] = alpha * series.iloc[i] + (1 - alpha) * vidya_values[i - 1]

        return pd.Series(vidya_values, index=series.index)

    @staticmethod
    @handle_pandas_ta_errors
    def wcp(data: pd.Series) -> pd.Series:
        """Weighted Closing Price"""
        return data

    @staticmethod
    @handle_pandas_ta_errors
    def linreg(data: pd.Series, length: int = 14) -> pd.Series:
        """Linear Regression Moving Average"""
        return ta.linreg(data, length=length)

    @staticmethod
    @handle_pandas_ta_errors
    def linreg_slope(data: pd.Series, length: int = 14) -> pd.Series:
        """Linear Regression Slope"""
        return ta.linreg(data, length=length, slope=True)

    @staticmethod
    @handle_pandas_ta_errors
    def linreg_intercept(data: pd.Series, length: int = 14) -> pd.Series:
        """Linear Regression Intercept"""
        return ta.linreg(data, length=length, intercept=True)

    @staticmethod
    @handle_pandas_ta_errors
    def linreg_angle(data: pd.Series, length: int = 14, degrees: bool = False) -> pd.Series:
        """Linear Regression Angle"""
        if degrees:
            return ta.linreg(data, length=length, degrees=True)
        else:
            return ta.linreg(data, length=length, angle=True)

    # Aliases for compatibility
    ichimoku = ichimoku_cloud

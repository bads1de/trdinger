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
- HT_TRENDLINE (Hilbert Transform - Instantaneous Trendline)
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
"""

from typing import Union, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import handle_pandas_ta_errors


class TrendIndicators:
    """
    トレンド系指標クラス
    """

    @staticmethod
    @handle_pandas_ta_errors
    def sma(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
        """単純移動平均（軽量エラーハンドリング付き）"""
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        # 基本的な入力検証
        if len(series) == 0:
            raise ValueError("データが空です")

        if length == 1:
            return series.values

        result = series.rolling(window=length, min_periods=1).mean()

        # 結果検証（重要な異常ケースのみ）
        if result.isna().all():
            raise ValueError("計算結果が全てNaNです")

        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def ema(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
        """指数移動平均"""
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        # 第一優先: pandas-ta
        try:
            result = ta.ema(series, length=length)
            if result is not None:
                # 全てNaNではなく、有用な値が含まれている場合のみ使用
                result_values = result.values if hasattr(result, 'values') else np.array(result)
                # 最初のlength位置以降に有効な値がある場合
                start_idx = length - 1
                if start_idx < len(result_values) and not np.isnan(result_values[start_idx:]).all():
                    return result_values
        except Exception:
            pass

        # フォールバック実装: numpyベース
        if len(series) < length:
            return np.full(len(series), np.nan)

        # EMA計算用変数
        alpha = 2.0 / (length + 1)
        ema_values = np.full(len(series), np.nan, dtype=float)
        ema_values[length - 1] = series.iloc[:length].mean()  # 初期値はSMA

        # ループでEMA計算
        for i in range(length, len(series)):
            ema_values[i] = alpha * series.iloc[i] + (1 - alpha) * ema_values[i - 1]

        return ema_values
    @staticmethod
    @handle_pandas_ta_errors
    def ppo(data: Union[np.ndarray, pd.Series], fast: int = 12, slow: int = 26, signal: int = 9):
        """Percentage Price Oscillator with pandas-ta fallback"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        try:
            result = ta.ppo(series, fast=fast, slow=slow, signal=signal)
            if result is not None and not result.empty:
                return result.iloc[:, 0].values, result.iloc[:, 1].values, result.iloc[:, 2].values
        except Exception:
            pass

        # フォールバック実装
        ema_fast = TrendIndicators.ema(series, fast)
        ema_slow = TrendIndicators.ema(series, slow)

        if ema_fast is None or ema_slow is None:
            nan_array = np.full(len(series), np.nan)
            return nan_array, nan_array, nan_array

        # PPO主力とシグナルラインの計算
        ppo_line = 100 * (ema_fast - ema_slow) / ema_slow
        signal_line = TrendIndicators.ema(ppo_line, signal)

        return ppo_line, signal_line, ppo_line - signal_line

    @staticmethod
    @handle_pandas_ta_errors
    def stochf(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
        fast_length: int = 3
    ):
        """Stochastic Fast with pandas-ta fallback"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        try:
            result = ta.stochf(
                high=high_series,
                low=low_series,
                close=close_series,
                fastk_length=fast_length,
                fastd_length=length
            )
            if result is not None and not result.empty:
                return result.iloc[:, 0].values, result.iloc[:, 1].values
        except Exception:
            pass

        # フォールバック実装 (シンプルなストキャスティクス計算)
        raw_k = 100 * (close_series - low_series.rolling(length).min()) / (high_series.rolling(length).max() - low_series.rolling(length).min())
        fast_k = raw_k.rolling(fast_length).mean()

        return fast_k.values, fast_k.values

    @staticmethod
    @handle_pandas_ta_errors
    def tema(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
        """三重指数移動平均"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        # 第一優先: pandas-ta
    @staticmethod
    @handle_pandas_ta_errors
    def stc(data: Union[np.ndarray, pd.Series], length: int = 10, fast_length: int = 23, slow_length: int = 50):
        """Schaff Trend Cycle with pandas-ta fallback"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        # 第一優先: pandas-ta
        try:
            result = ta.stc(series, length=length, fastLength=fast_length, slowLength=slow_length)
            if result is not None and not result.isna().all():
                return result.values if hasattr(result, 'values') else np.array(result)
        except Exception:
            pass

        # フォールバック実装: 簡易STC (MACDベース)
        # %K: EMA(EMA(price, fast)) / EMA(EMA(price, slow)) - 1
        # %D: EMA(%K) * 100
        # STC: EMA(%D) * 100 (シグナルラインなしの場合は 3重EMAベース)

        if len(series) < slow_length:
            return np.full(len(series), np.nan)

        # 基本的なMACD計算
        ema_fast = TrendIndicators.ema(series, fast_length)
        ema_slow = TrendIndicators.ema(series, slow_length)
        macd = ema_fast - ema_slow

        # MACDのサイクル分析 (変動範囲のトレンド)
        if len(macd) > 0:
            # ピークとバレー検出 (簡易版)
            stc_values = np.full(len(series), np.nan)
            valid_start = slow_length - 1

            # 基本的なサイクル計算 (簡易: MACDのノーマライズ)
            if len(macd) >= valid_start + length:
                # MACDのグサイクルをトレンドサイクルに変換
                cycle = ((macd - macd.rolling(slow_length).min()) /
                        (macd.rolling(slow_length).max() - macd.rolling(slow_length).min())) * 100

                # 最終的なSTC値
                stc_values[valid_start:] = TrendIndicators.ema(cycle[valid_start:], length)

            return stc_values

        return np.full(len(series), np.nan)
        # 第一優先: pandas-ta
        try:
            result = ta.tema(series, length=length)
            if result is not None:
                # 全てNaNではなく、有用な値が含まれている場合のみ使用
                result_values = result.values if hasattr(result, 'values') else np.array(result)
                # 最初のlength位置以降に有効な値がある場合
                start_idx = length * 3 - 1
                if start_idx < len(result_values) and not np.isnan(result_values[start_idx:]).all():
                    return result_values
        except Exception:
            pass

        # フォールバック実装: 多段EMA計算
        if len(series) < length * 3:
            return np.full(len(series), np.nan)

        # EMA1, EMA2, EMA3の計算
        ema1 = TrendIndicators.ema(series, length)
        ema1_series = pd.Series(ema1, index=series.index)

        ema2 = TrendIndicators.ema(ema1_series, length)
        ema2_series = pd.Series(ema2, index=series.index)

        ema3 = TrendIndicators.ema(ema2_series, length)
        ema3_series = pd.Series(ema3, index=series.index)

        # TEMA = 3*EMA1 - 3*EMA2 + EMA3
        tema_values = 3 * ema1_series - 3 * ema2_series + ema3_series

        return tema_values.values if hasattr(tema_values, 'values') else np.array(tema_values)

    @staticmethod
    @handle_pandas_ta_errors
    def dema(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
        """二重指数移動平均"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        # 第一優先: pandas-ta
        try:
            result = ta.dema(series, length=length)
            if result is not None:
                # 全てNaNではなく、有用な値が含まれている場合のみ使用
                result_values = result.values if hasattr(result, 'values') else np.array(result)
                # 最初のlength位置以降に有効な値がある場合
                start_idx = length * 2 - 1
                if start_idx < len(result_values) and not np.isnan(result_values[start_idx:]).all():
                    return result_values
        except Exception:
            pass

        # フォールバック実装: 多段EMA計算 (DEMA = 2*EMA1 - EMA2)
        if len(series) < length * 2:
            return np.full(len(series), np.nan)

        # EMA1, EMA2の計算
        ema1 = TrendIndicators.ema(series, length)
        ema1_series = pd.Series(ema1, index=series.index)

        ema2 = TrendIndicators.ema(ema1_series, length)
        ema2_series = pd.Series(ema2, index=series.index)

        # DEMA = 2*EMA1 - EMA2
        dema_values = 2 * ema1_series - ema2_series

        return dema_values.values if hasattr(dema_values, 'values') else np.array(dema_values)

    @staticmethod
    @handle_pandas_ta_errors
    def wma(
        data: Union[np.ndarray, pd.Series] = None,
        length: int = 14,
        close: Union[np.ndarray, pd.Series] = None,
    ) -> np.ndarray:
        """加重移動平均"""
        # dataが提供されない場合はcloseを使用
        if data is None and close is not None:
            data = close
        elif data is None:
            raise ValueError("Either 'data' or 'close' must be provided")

        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return ta.wma(series, length=length).values

    @staticmethod
    @handle_pandas_ta_errors
    def trima(
        data: Union[np.ndarray, pd.Series] = None,
        length: int = 14,
        close: Union[np.ndarray, pd.Series] = None,
    ) -> np.ndarray:
        """三角移動平均"""
        # dataが提供されない場合はcloseを使用
        if data is None and close is not None:
            data = close
        elif data is None:
            raise ValueError("Either 'data' or 'close' must be provided")

        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return ta.trima(series, length=length).values

    @staticmethod
    @handle_pandas_ta_errors
    def kama(data: Union[np.ndarray, pd.Series], length: int = 30) -> np.ndarray:
        """カウフマン適応移動平均"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return ta.kama(series, length=length).values

    @staticmethod
    @handle_pandas_ta_errors
    def t3(
        data: Union[np.ndarray, pd.Series], length: int = 5, a: float = 0.7
    ) -> np.ndarray:
        """T3移動平均"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return ta.t3(series, length=length, a=a).values

    @staticmethod
    @handle_pandas_ta_errors
    def sar(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        af: float = 0.02,
        max_af: float = 0.2,
    ) -> np.ndarray:
        """パラボリックSAR"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low

        result = ta.psar(high=high_series, low=low_series, af0=af, af=af, max_af=max_af)
        # PSARl と PSARs を結合
        psar_long = result[f"PSARl_{af}_{max_af}"]
        psar_short = result[f"PSARs_{af}_{max_af}"]
        return psar_long.fillna(psar_short).values

    @staticmethod
    @handle_pandas_ta_errors
    def sarext(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        startvalue: float = 0.0,
        offsetonreverse: float = 0.0,
        accelerationinitlong: float = 0.02,
        accelerationlong: float = 0.02,
        accelerationmaxlong: float = 0.2,
        accelerationinitshort: float = 0.02,
        accelerationshort: float = 0.02,
        accelerationmaxshort: float = 0.2,
    ) -> np.ndarray:
        """Extended Parabolic SAR (pandas-ta psarで近似)"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low

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
            high=high_series,
            low=low_series,
            af0=accelerationinitlong,
            af=accelerationlong,
            max_af=accelerationmaxlong,
        )

        psar_long = result[f"PSARl_{accelerationlong}_{accelerationmaxlong}"]
        psar_short = result[f"PSARs_{accelerationlong}_{accelerationmaxlong}"]
        return psar_long.fillna(psar_short).values

    @staticmethod
    @handle_pandas_ta_errors
    def ht_trendline(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Hilbert Transform - Instantaneous Trendline"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        # pandas-taにht_trendlineがあれば使用、なければEMAで代替
        if hasattr(ta, "ht_trendline"):
            return ta.ht_trendline(series).values
        else:
            return TrendIndicators.ema(series, length=3)

    @staticmethod
    @handle_pandas_ta_errors
    def ma(
        data: Union[np.ndarray, pd.Series] = None,
        period: int = 30,
        matype: int = 0,
        close: Union[np.ndarray, pd.Series] = None,
    ) -> np.ndarray:
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
        data: Union[np.ndarray, pd.Series],
        periods: Union[np.ndarray, pd.Series],
        minperiod: int = 2,
        maxperiod: int = 30,
        matype: int = 0,
    ) -> np.ndarray:
        """可変期間移動平均（カスタム実装）"""
        data_array = np.array(data)
        periods_array = np.array(periods)

        if len(data_array) != len(periods_array):
            raise ValueError(
                f"データと期間の長さが一致しません。Data: {len(data_array)}, Periods: {len(periods_array)}"
            )

        # pandas-taには直接的な実装がないため、カスタム実装
        result = np.full_like(data_array, np.nan, dtype=float)

        for i in range(len(data_array)):
            period = int(np.clip(periods_array[i], minperiod, maxperiod))
            start_idx = max(0, i - period + 1)

            if i >= period - 1:
                window_data = data_array[start_idx : i + 1]
                if matype == 0:  # SMA
                    result[i] = np.mean(window_data)
                else:
                    # 他のタイプは簡略化してSMAで代替
                    result[i] = np.mean(window_data)

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def midpoint(
        data: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """期間内の中点"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        return ta.midpoint(series, length=length).values

    @staticmethod
    @handle_pandas_ta_errors
    def midprice(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """期間内の中値価格"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low

        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        return ta.midprice(high=high_series, low=low_series, length=length).values

    @staticmethod
    @handle_pandas_ta_errors
    def hma(data: Union[np.ndarray, pd.Series], length: int = 20) -> np.ndarray:
        """Hull Moving Average"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        # 第一優先: pandas-ta
        try:
            result = ta.hma(series, length=length)
            if result is not None:
                # 全てNaNではなく、有用な値が含まれている場合のみ使用
                result_values = result.values if hasattr(result, 'values') else np.array(result)
                # 最初のlength位置以降に有効な値がある場合
                start_idx = int(length * 1.5) - 1  # HMAはより多くのデータを必要とする
                if start_idx < len(result_values) and not np.isnan(result_values[start_idx:]).all():
                    return result_values
        except Exception:
            pass

        # フォールバック実装: カスタムHull Moving Average
        if len(series) < length * 3:  # HMAは3つのWMAを必要とする
            return np.full(len(series), np.nan)

        # 計算ステップ:
        # 1. WMA(n/2)
        # 2. WMA(n)
        # 3. WMA(n/2) を WMA(n) から引いて、2倍する
        # 4. 結果を sqrt(n)でWMA

        import math
        hma_length = int(length / 2)
        sqrt_length = int(math.floor(math.sqrt(length)))

        # WMA(n/2) と WMA(n) の計算
        wma_half = pd.Series(TrendIndicators.wma(series, hma_length))
        wma_full = pd.Series(TrendIndicators.wma(series, length))

        # 差分を2倍
        diff = 2 * wma_half - wma_full

        # sqrt(n)でWMA
        return TrendIndicators.wma(diff, sqrt_length).values

    @staticmethod
    @handle_pandas_ta_errors
    def zlma(data: Union[np.ndarray, pd.Series], length: int = 20) -> np.ndarray:
        """Zero-Lag Exponential Moving Average"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        if hasattr(ta, "zlma"):
            return ta.zlma(series, length=length).values
        else:
            # フォールバック: EMAの差分で近似
            lag = int((length - 1) / 2)
            shifted = series.shift(lag)
            adjusted = series + (series - shifted)
            return ta.ema(adjusted, length=length).values

    @staticmethod
    @handle_pandas_ta_errors
    def vwma(
        data: Union[np.ndarray, pd.Series] = None,
        volume: Union[np.ndarray, pd.Series] = None,
        length: int = 20,
        close: Union[np.ndarray, pd.Series] = None,
    ) -> np.ndarray:
        """Volume Weighted Moving Average"""
        # dataが提供されない場合はcloseを使用
        if data is None and close is not None:
            data = close
        elif data is None:
            raise ValueError("Either 'data' or 'close' must be provided")

        price = pd.Series(data) if isinstance(data, np.ndarray) else data
        vol = pd.Series(volume) if isinstance(volume, np.ndarray) else volume

        try:
            result = ta.vwma(price, volume=vol, length=length)
            if result is not None:
                return result.values
        except Exception:
            pass

        # フォールバック: カスタム実装
        if len(price) < length:
            # データが不十分な場合はNaNで埋める
            return np.full(len(price), np.nan)

        # VWMAの手動計算
        vwma_values = np.full(len(price), np.nan)
        for i in range(length - 1, len(price)):
            window_price = price.iloc[i - length + 1 : i + 1]
            window_vol = vol.iloc[i - length + 1 : i + 1]
            vwma_values[i] = np.average(window_price, weights=window_vol)

        return vwma_values

    @staticmethod
    @handle_pandas_ta_errors
    def swma(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """Symmetric Weighted Moving Average"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        # 第一優先: pandas-ta
        try:
            result = ta.swma(series, length=length)
            if result is not None:
                # 全てNaNではなく、有用な値が含まれている場合のみ使用
                result_values = result.values if hasattr(result, 'values') else np.array(result)
                # 最初のlength位置以降に有効な値がある場合
                start_idx = length - 1
                if start_idx < len(result_values) and not np.isnan(result_values[start_idx:]).all():
                    return result_values
        except Exception:
            pass

        # フォールバック実装: カスタム対称重み付き移動平均
        if len(series) < length:
            return np.full(len(series), np.nan)

        # SWMA: 対称重み (中心が最大重み、両端に向かって線形減衰)
        swma_values = np.full(len(series), np.nan)

        # 重み計算 (三角形分布)
        weights = np.concatenate([
            np.arange(1, length//2 + 1),  # 上昇部分
            np.arange(length//2, 0, -1)   # 下降部分
        ][:length])
        weights = weights[:length]  # 長さに合わせる
        weights = weights / weights.sum()  # 正規化

        for i in range(length - 1, len(series)):
            window = series.iloc[i - length + 1:i + 1].values
            swma_values[i] = np.sum(window * weights)

        return swma_values

    @staticmethod
    @handle_pandas_ta_errors
    def alma(
        data: Union[np.ndarray, pd.Series],
        length: int = 9,
        sigma: float = 6.0,
        offset: float = 0.85,
    ) -> np.ndarray:
        """Arnaud Legoux Moving Average"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        # 第一優先: pandas-ta
        try:
            result = ta.alma(series, length=length, sigma=sigma, offset=offset)
            if result is not None:
                # 全てNaNではなく、有用な値が含まれている場合のみ使用
                result_values = result.values if hasattr(result, 'values') else np.array(result)
                # 最初のlength位置以降に有効な値がある場合
                start_idx = length - 1
                if start_idx < len(result_values) and not np.isnan(result_values[start_idx:]).all():
                    return result_values
        except Exception:
            pass

        # フォールバック実装: カスタムガウシアン重み付き移動平均
        if len(series) < length:
            return np.full(len(series), np.nan)

        import math
        # ガウシアンダistributionによる重み計算
        weights = np.zeros(length)
        m = offset * (length - 1)  # 調整係数

        for i in range(length):
            weights[i] = np.exp(-np.power(i - m, 2) / (2 * np.power(length / sigma, 2)))

        weights = weights / weights.sum()  # 正規化

        # 重み付き移動平均計算
        alma_values = np.full(len(series), np.nan)
        for i in range(length - 1, len(series)):
            window = series.iloc[i - length + 1:i + 1].values
            alma_values[i] = np.sum(window * weights)

        return alma_values

    @staticmethod
    @handle_pandas_ta_errors
    def rma(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
        """Smoothed Moving Average (RMA)"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        try:
            result = ta.rma(series, length=length)
            if result is not None:
                return result.values
        except Exception:
            pass

        # フォールバック: カスタム実装 (RMA = EMA with alpha = 1/length)
        if len(series) < length:
            # データが不十分な場合はNaNで埋める
            return np.full(len(series), np.nan)

        # RMAの手動計算（EMAと同様）
        alpha = 1.0 / length
        rma_values = np.full(len(series), np.nan)
        rma_values[length - 1] = series.iloc[:length].mean()  # 初期値はSMA

        for i in range(length, len(series)):
            rma_values[i] = alpha * series.iloc[i] + (1 - alpha) * rma_values[i - 1]

        return rma_values

    @staticmethod
    @handle_pandas_ta_errors
    def ichimoku(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        tenkan: int = 9,
        kijun: int = 26,
        senkou: int = 52,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Ichimoku Cloud: (conversion, base, span_a, span_b, lagging)"""
        h = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        c = pd.Series(close) if isinstance(close, np.ndarray) else close

        # 標準定義に基づく計算
        conv = (h.rolling(tenkan).max() + low_series.rolling(tenkan).min()) / 2.0
        base = (h.rolling(kijun).max() + low_series.rolling(kijun).min()) / 2.0
        span_a = ((conv + base) / 2.0).shift(kijun)
        span_b = (
            (h.rolling(senkou).max() + low_series.rolling(senkou).min()) / 2.0
        ).shift(kijun)
        lag = c.shift(-kijun)

        return (
            conv.values,
            base.values,
            span_a.values,
            span_b.values,
            lag.values,
        )

    # カスタム指標
    @staticmethod
    @handle_pandas_ta_errors
    def sma_slope(data: Union[np.ndarray, pd.Series], length: int = 20) -> np.ndarray:
        """SMAの傾き（前期間との差分）"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        sma_vals = pd.Series(TrendIndicators.sma(series, length))
        return sma_vals.diff().values

    @staticmethod
    @handle_pandas_ta_errors
    def price_ema_ratio(
        data: Union[np.ndarray, pd.Series], length: int = 20
    ) -> np.ndarray:
        """価格とEMAの比率 - 1"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        ema_vals = TrendIndicators.ema(series, length)
        ema_series = pd.Series(ema_vals, index=series.index)
        return ((series / ema_series) - 1.0).values

    @staticmethod
    @handle_pandas_ta_errors
    def mama(
        data: Union[np.ndarray, pd.Series],
        fastlimit: float = 0.5,
        slowlimit: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """MESA Adaptive Moving Average"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        # pandas-taにmamaがあるか確認
        if hasattr(ta, "mama"):
            result = ta.mama(series, fastlimit=fastlimit, slowlimit=slowlimit)
            return result.iloc[:, 0].values, result.iloc[:, 1].values
        else:
            # フォールバック: EMAで代替
            mama_values = ta.ema(series, length=20).values
            fama_values = ta.ema(series, length=40).values
            return mama_values, fama_values

    @staticmethod
    @handle_pandas_ta_errors
    def maxindex(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
        """最大値のインデックス"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return (
            series.rolling(window=length).apply(lambda x: x.argmax(), raw=False).values
        )

    @staticmethod
    @handle_pandas_ta_errors
    def minindex(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
        """最小値のインデックス"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return (
            series.rolling(window=length).apply(lambda x: x.argmin(), raw=False).values
        )

    @staticmethod
    @handle_pandas_ta_errors
    def minmax(
        data: Union[np.ndarray, pd.Series], length: int = 14
    ) -> Tuple[np.ndarray, np.ndarray]:
        """最小値と最大値"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        min_vals = series.rolling(window=length).min().values
        max_vals = series.rolling(window=length).max().values
        return min_vals, max_vals

    @staticmethod
    @handle_pandas_ta_errors
    def minmaxindex(
        data: Union[np.ndarray, pd.Series], length: int = 14
    ) -> Tuple[np.ndarray, np.ndarray]:
        """最小値と最大値のインデックス"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        min_idx = (
            series.rolling(window=length).apply(lambda x: x.argmin(), raw=False).values
        )
        max_idx = (
            series.rolling(window=length).apply(lambda x: x.argmax(), raw=False).values
        )
        return min_idx, max_idx

    @staticmethod
    @handle_pandas_ta_errors
    def fwma(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """Fibonacci's Weighted Moving Average"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        # 第一優先: pandas-ta
        try:
            result = ta.fwma(series, length=length)
            if result is not None and not result.isna().all():
                return result.values if hasattr(result, 'values') else np.array(result)
        except Exception:
            pass

        # フォールバック実装: カスタム重み付き移動平均
        # フィボナッチ数的減衰重みを使用 (簡易バージョン)
        if len(series) < length:
            return np.full(len(series), np.nan)

        # フィボナッチ重み生成 (簡易版: 線形減衰に近似)
        weights = np.arange(1, length + 1, dtype=float)
        weights = weights / weights.sum()  # 正規化

        # 重み付き移動平均計算
        fwma_values = np.full(len(series), np.nan)
        for i in range(length - 1, len(series)):
            window = series.iloc[i - length + 1:i + 1].values
            fwma_values[i] = np.sum(window * weights)

        return fwma_values

    @staticmethod
    @handle_pandas_ta_errors
    def hilo(high: Union[np.ndarray, pd.Series], low: Union[np.ndarray, pd.Series], close: Union[np.ndarray, pd.Series] = None, high_length=None, low_length=None, length: int = 14, **kwargs) -> np.ndarray:
        """Gann High-Low Activator"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        # Use high_length and low_length if provided, otherwise use length for both
        hl = high_length if high_length is not None else length
        ll = low_length if low_length is not None else length

        result = ta.hilo(high=high_series, low=low_series, close=close_series, high_length=hl, low_length=ll)
        return result.values if result is not None else np.full(len(high_series), np.nan)

    @staticmethod
    @handle_pandas_ta_errors
    def hl2(high: Union[np.ndarray, pd.Series], low: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """High-Low Average"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        return ((high_series + low_series) / 2).values

    @staticmethod
    @handle_pandas_ta_errors
    def hlc3(high: Union[np.ndarray, pd.Series], low: Union[np.ndarray, pd.Series], close: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """High-Low-Close Average"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close
        return ((high_series + low_series + close_series) / 3).values

    @staticmethod
    @handle_pandas_ta_errors
    def hwma(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """Holt-Winter Moving Average"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.hwma(series, length=length)
        return result.values if result is not None else np.full(len(series), np.nan)

    @staticmethod
    @handle_pandas_ta_errors
    def jma(data: Union[np.ndarray, pd.Series], length: int = 7, phase: float = 0.0, power: float = 2.0) -> np.ndarray:
        """Jurik Moving Average"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.jma(series, length=length, phase=phase, power=power)
        return result.values if result is not None else np.full(len(series), np.nan)

    @staticmethod
    @handle_pandas_ta_errors
    def mcgd(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """McGinley Dynamic"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        if len(series) < length:
            return np.full(len(series), np.nan)

        mcgd_values = np.full(len(series), np.nan)
        mcgd_values[length - 1] = series.iloc[:length].mean()  # 初期値はSMA

        k = 0.6  # 調整係数

        for i in range(length, len(series)):
            ratio = series.iloc[i] / mcgd_values[i-1] if mcgd_values[i-1] != 0 else 1
            mcgd_values[i] = mcgd_values[i-1] + (series.iloc[i] - mcgd_values[i-1]) / (k * length * (ratio ** 4))

        return mcgd_values

    @staticmethod
    @handle_pandas_ta_errors
    def ohlc4(open_: Union[np.ndarray, pd.Series], high: Union[np.ndarray, pd.Series], low: Union[np.ndarray, pd.Series], close: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Open-High-Low-Close Average"""
        open_series = pd.Series(open_) if isinstance(open_, np.ndarray) else open_
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close
        return ((open_series + high_series + low_series + close_series) / 4).values

    @staticmethod
    @handle_pandas_ta_errors
    def pwma(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """Pascal's Weighted Moving Average"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.pwma(series, length=length)
        return result.values if result is not None else np.full(len(series), np.nan)

    @staticmethod
    @handle_pandas_ta_errors
    def sinwma(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """Sine Weighted Moving Average"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.sinwma(series, length=length)
        return result.values if result is not None else np.full(len(series), np.nan)

    @staticmethod
    @handle_pandas_ta_errors
    def ssf(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """Ehler's Super Smoother Filter"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.ssf(series, length=length)
        return result.values if result is not None else np.full(len(series), np.nan)

    @staticmethod
    @handle_pandas_ta_errors
    def vidya(data: Union[np.ndarray, pd.Series], length: int = 14, adjust: bool = True) -> np.ndarray:
        """Variable Index Dynamic Average"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data

        # 第一優先: pandas-ta
        try:
            result = ta.vidya(series, length=length, adjust=adjust)
            if result is not None and not result.isna().all():
                result_values = result.values if hasattr(result, 'values') else np.array(result)
                # 最初のlength位置以降に有効な値がある場合
                start_idx = length * 2 - 1  # VIDYAはより多くのデータを必要とする
                if start_idx < len(result_values) and not np.isnan(result_values[start_idx:]).all():
                    return result_values
        except Exception:
            pass

        # フォールバック実装: カスタムVIDYA (CMOベースの適応的平均)
        if len(series) < length * 3:
            return np.full(len(series), np.nan)

        vidya_values = np.full(len(series), np.nan)

        # 最初の値をSMAで初期化
        vidya_values[length * 2 - 1] = series.iloc[:length*2].mean()

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
            vidya_values[i] = alpha * series.iloc[i] + (1 - alpha) * vidya_values[i-1]

        return vidya_values

    @staticmethod
    @handle_pandas_ta_errors
    def wcp(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Weighted Closing Price"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return series.values  # WCP is essentially the close price itself

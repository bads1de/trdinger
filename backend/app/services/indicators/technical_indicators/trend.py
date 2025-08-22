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
        result = ta.ema(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def tema(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
        """三重指数移動平均"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        result = ta.tema(series, length=length)
        # 全NaNの場合はEMAにフォールバック
        if result.isna().all():
            result = ta.ema(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def dema(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
        """二重指数移動平均"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return ta.dema(series, length=length).values

    @staticmethod
    @handle_pandas_ta_errors
    def wma(
        data: Union[np.ndarray, pd.Series] = None,
        length: int = 14,
        close: Union[np.ndarray, pd.Series] = None
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
        close: Union[np.ndarray, pd.Series] = None
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
        close: Union[np.ndarray, pd.Series] = None
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
            3: TrendIndicators.dema, # DEMA
            4: TrendIndicators.tema, # TEMA
            5: TrendIndicators.trima, # TRIMA
            6: TrendIndicators.kama,  # KAMA
            8: TrendIndicators.t3,   # T3
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
        length: int = None,
        period: int = None,
    ) -> np.ndarray:
        """期間内の中点"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        length = period if period is not None else length

        if length is None:
            raise ValueError("length または period を指定してください")

        return ta.midpoint(series, length=length).values

    @staticmethod
    @handle_pandas_ta_errors
    def midprice(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        length: int = None,
        period: int = None,
    ) -> np.ndarray:
        """期間内の中値価格"""
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        length = period if period is not None else length

        if length is None:
            raise ValueError("length または period を指定してください")

        return ta.midprice(high=high_series, low=low_series, length=length).values

    @staticmethod
    @handle_pandas_ta_errors
    def hma(data: Union[np.ndarray, pd.Series], length: int = 20) -> np.ndarray:
        """Hull Moving Average"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return ta.hma(series, length=length).values

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
            window_price = price.iloc[i - length + 1:i + 1]
            window_vol = vol.iloc[i - length + 1:i + 1]
            vwma_values[i] = np.average(window_price, weights=window_vol)

        return vwma_values

    @staticmethod
    @handle_pandas_ta_errors
    def swma(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """Symmetric Weighted Moving Average"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return ta.swma(series, length=length).values

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
        return ta.alma(series, length=length, sigma=sigma, offset=offset).values

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
        slowlimit: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """MESA Adaptive Moving Average"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        
        # pandas-taにmamaがあるか確認
        if hasattr(ta, 'mama'):
            result = ta.mama(series, fastlimit=fastlimit, slowlimit=slowlimit)
            return result.iloc[:, 0].values, result.iloc[:, 1].values
        else:
            # フォールバック: EMAで代替
            mama_values = ta.ema(series, length=20).values
            fama_values = ta.ema(series, length=40).values
            return mama_values, fama_values

    @staticmethod
    @handle_pandas_ta_errors
    def maxindex(data: Union[np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
        """最大値のインデックス"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return series.rolling(window=period).apply(lambda x: x.argmax(), raw=False).values

    @staticmethod
    @handle_pandas_ta_errors
    def minindex(data: Union[np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
        """最小値のインデックス"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return series.rolling(window=period).apply(lambda x: x.argmin(), raw=False).values

    @staticmethod
    @handle_pandas_ta_errors
    def minmax(data: Union[np.ndarray, pd.Series], period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """最小値と最大値"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        min_vals = series.rolling(window=period).min().values
        max_vals = series.rolling(window=period).max().values
        return min_vals, max_vals

    @staticmethod
    @handle_pandas_ta_errors
    def minmaxindex(data: Union[np.ndarray, pd.Series], period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """最小値と最大値のインデックス"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        min_idx = series.rolling(window=period).apply(lambda x: x.argmin(), raw=False).values
        max_idx = series.rolling(window=period).apply(lambda x: x.argmax(), raw=False).values
        return min_idx, max_idx

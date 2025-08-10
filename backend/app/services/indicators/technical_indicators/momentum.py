"""
モメンタム系テクニカル指標（pandas-ta移行版）

このモジュールはpandas-taライブラリを使用し、
backtesting.pyとの完全な互換性を提供します。
numpy配列ベースのインターフェースを維持しています。
"""

from typing import Tuple, cast, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import (
    PandasTAError,
    handle_pandas_ta_errors,
    to_pandas_series,
    validate_series_data,
    validate_indicator_parameters,
)


class MomentumIndicators:
    """
    モメンタム系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    @handle_pandas_ta_errors
    def rsi(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
        """相対力指数"""
        validate_indicator_parameters(length)
        series = to_pandas_series(data)
        validate_series_data(series, length + 1)
        result = ta.rsi(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def macd(
        data: Union[np.ndarray, pd.Series], fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD"""
        series = to_pandas_series(data)
        validate_series_data(series, slow + signal)
        result = ta.macd(series, fast=fast, slow=slow, signal=signal)

        macd_col = f"MACD_{fast}_{slow}_{signal}"
        signal_col = f"MACDs_{fast}_{slow}_{signal}"
        hist_col = f"MACDh_{fast}_{slow}_{signal}"

        return (result[macd_col].values, result[signal_col].values, result[hist_col].values)

    @staticmethod
    @handle_pandas_ta_errors
    def macdext(
        data: Union[np.ndarray, pd.Series],
        fastperiod: int = 12,
        fastmatype: int = 0,
        slowperiod: int = 26,
        slowmatype: int = 0,
        signalperiod: int = 9,
        signalmatype: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Approximate MACDEXT via pandas-ta by computing MACD and ignoring matype differences"""
        series = to_pandas_series(data)
        validate_series_data(series, max(fastperiod, slowperiod, signalperiod))
        result = ta.macd(series, fast=fastperiod, slow=slowperiod, signal=signalperiod)

        macd_col = f"MACD_{fastperiod}_{slowperiod}_{signalperiod}"
        signal_col = f"MACDs_{fastperiod}_{slowperiod}_{signalperiod}"
        hist_col = f"MACDh_{fastperiod}_{slowperiod}_{signalperiod}"

        return (result[macd_col].values, result[signal_col].values, result[hist_col].values)

    @staticmethod
    @handle_pandas_ta_errors
    def macdfix(
        data: Union[np.ndarray, pd.Series], signalperiod: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        series = to_pandas_series(data)
        validate_series_data(series, 26 + signalperiod)
        # pandas-ta does not have macdfix; approximate by standard macd with fixed periods
        result = ta.macd(series, fast=12, slow=26, signal=signalperiod)
        macd_col = f"MACD_12_26_{signalperiod}"
        signal_col = f"MACDs_12_26_{signalperiod}"
        hist_col = f"MACDh_12_26_{signalperiod}"
        return (result[macd_col].values, result[signal_col].values, result[hist_col].values)

    @staticmethod
    @handle_pandas_ta_errors
    def stoch(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        k: int = 14,
        d: int = 3,
        smooth_k: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ストキャスティクス"""
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        validate_series_data(high_series, k)
        validate_series_data(low_series, k)
        validate_series_data(close_series, k)

        result = ta.stoch(
            high=high_series,
            low=low_series,
            close=close_series,
            k=k,
            d=d,
            smooth_k=smooth_k,
        )

        k_col = f"STOCHk_{k}_{d}_{smooth_k}"
        d_col = f"STOCHd_{k}_{d}_{smooth_k}"

        return (result[k_col].values, result[d_col].values)

    @staticmethod
    @handle_pandas_ta_errors
    def stochf(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        k: int = 5,
        d: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """高速ストキャスティクス"""
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        validate_series_data(high_series, k)
        validate_series_data(low_series, k)
        validate_series_data(close_series, k)

        result = ta.stochf(
            high=high_series, low=low_series, close=close_series, k=k, d=d
        )
        return result[f"STOCHFk_{k}_{d}"].values, result[f"STOCHFd_{k}_{d}"].values

    @staticmethod
    @handle_pandas_ta_errors
    def stochrsi(
        data: Union[np.ndarray, pd.Series],
        length: int = 14,
        k: int = 5,
        d: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ストキャスティクスRSI"""
        series = to_pandas_series(data)
        validate_series_data(series, length + k + d)
        result = ta.stochrsi(series, length=length, k=k, d=d)
        return result[f"STOCHRSIk_{length}_{k}_{d}"].values, result[f"STOCHRSId_{length}_{k}_{d}"].values

    @staticmethod
    @handle_pandas_ta_errors
    def willr(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """ウィリアムズ%R"""
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)
        validate_series_data(close_series, length)

        result = ta.willr(
            high=high_series, low=low_series, close=close_series, length=length
        )
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def cci(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 20,
    ) -> np.ndarray:
        """商品チャネル指数"""
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)
        validate_series_data(close_series, length)

        result = ta.cci(high=high_series, low=low_series, close=close_series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def cmo(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
        """チェンジモメンタムオシレーター"""
        series = to_pandas_series(data)
        validate_series_data(series, length)
        result = ta.cmo(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def roc(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """変化率"""
        series = to_pandas_series(data)
        validate_indicator_parameters(length)
        validate_series_data(series, length)
        result = ta.roc(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def rocp(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """変化率（%）"""
        series = to_pandas_series(data)
        validate_series_data(series, length)
        result = ta.rocp(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def rocr(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """変化率（比率）"""
        series = to_pandas_series(data)
        validate_series_data(series, length)
        result = ta.rocr(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def rocr100(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """変化率（比率100スケール）"""
        series = to_pandas_series(data)
        validate_series_data(series, length)
        result = ta.rocr(series, length=length, scalar=100)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def mom(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """モメンタム"""
        series = to_pandas_series(data)
        validate_indicator_parameters(length)
        validate_series_data(series, length)
        result = ta.mom(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def adx(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """平均方向性指数"""
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)
        validate_series_data(close_series, length)

        result = ta.adx(high=high_series, low=low_series, close=close_series, length=length)
        return result[f"ADX_{length}"].values

    @staticmethod
    @handle_pandas_ta_errors
    def adxr(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """ADX評価"""
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)
        validate_series_data(close_series, length)

        result = ta.adx(
            high=high_series, low=low_series, close=close_series, length=length
        )
        return result[f"ADXR_{length}"].values

    @staticmethod
    @handle_pandas_ta_errors
    def aroon(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """アルーン"""
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)

        result = ta.aroon(high=high_series, low=low_series, length=length)
        return result[f"AROOND_{length}"].values, result[f"AROONU_{length}"].values

    @staticmethod
    @handle_pandas_ta_errors
    def aroonosc(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """アルーンオシレーター"""
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)

        result = ta.aroon(high=high_series, low=low_series, length=length)
        return result[f"AROONOSC_{length}"].values

    @staticmethod
    @handle_pandas_ta_errors
    def dx(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """Directional Movement Index wrapper (DX)"""
        # pandas-ta returns DX as part of adx; extract DX
        high_s = to_pandas_series(high)
        low_s = to_pandas_series(low)
        close_s = to_pandas_series(close)
        validate_series_data(high_s, length)
        validate_series_data(low_s, length)
        validate_series_data(close_s, length)
        result = ta.adx(high=high_s, low=low_s, close=close_s, length=length)
        # result contains DX_{length} column
        dx_col = f"DX_{length}"
        if dx_col in result.columns:
            return result[dx_col].values
        # fallback: compute difference between plus and minus DI
        plus = result[f"DMP_{length}"] if f"DMP_{length}" in result.columns else None
        minus = result[f"DMN_{length}"] if f"DMN_{length}" in result.columns else None
        if plus is not None and minus is not None:
            return (plus - minus).values
        raise PandasTAError("DX not available from pandas-ta in this version")

    @staticmethod
    @handle_pandas_ta_errors
    def mfi(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        volume: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """マネーフローインデックス"""
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)
        volume_series = to_pandas_series(volume)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)
        validate_series_data(close_series, length)
        validate_series_data(volume_series, length)

        result = ta.mfi(
            high=high_series,
            low=low_series,
            close=close_series,
            volume=volume_series,
            length=length,
        )
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def plus_di(high, low, close, length: int = 14) -> np.ndarray:
        high_s = to_pandas_series(high)
        low_s = to_pandas_series(low)
        close_s = to_pandas_series(close)
        result = ta.adx(high=high_s, low=low_s, close=close_s, length=length)
        col = f"DMP_{length}"
        if col in result.columns:
            return result[col].values
        raise PandasTAError("PLUS_DI not available in this pandas-ta version")

    @staticmethod
    @handle_pandas_ta_errors
    def minus_di(high, low, close, length: int = 14) -> np.ndarray:
        high_s = to_pandas_series(high)
        low_s = to_pandas_series(low)
        close_s = to_pandas_series(close)
        result = ta.adx(high=high_s, low=low_s, close=close_s, length=length)
        col = f"DMN_{length}"
        if col in result.columns:
            return result[col].values
        raise PandasTAError("MINUS_DI not available in this pandas-ta version")

    @staticmethod
    @handle_pandas_ta_errors
    def plus_dm(high, low, length: int = 14) -> np.ndarray:
        high_s = to_pandas_series(high)
        low_s = to_pandas_series(low)
        result = ta.dm(high=high_s, low=low_s, length=length)
        # pandas-ta dm returns DMP and DMN columns
        cols = [c for c in result.columns if c.startswith("DMP_")]
        if cols:
            return result[cols[0]].values
        raise PandasTAError("PLUS_DM not available in this pandas-ta version")

    @staticmethod
    @handle_pandas_ta_errors
    def minus_dm(high, low, length: int = 14) -> np.ndarray:
        high_s = to_pandas_series(high)
        low_s = to_pandas_series(low)
        result = ta.dm(high=high_s, low=low_s, length=length)
        cols = [c for c in result.columns if c.startswith("DMN_")]
        if cols:
            return result[cols[0]].values
        raise PandasTAError("MINUS_DM not available in this pandas-ta version")

    @staticmethod
    @handle_pandas_ta_errors
    def ppo(
        data: Union[np.ndarray, pd.Series],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> np.ndarray:
        """パーセンテージ価格オシレーター"""
        series = to_pandas_series(data)
        validate_series_data(series, slow)
        result = ta.ppo(series, fast=fast, slow=slow, signal=signal)
        return result[f"PPO_{fast}_{slow}_{signal}"].values

    @staticmethod
    @handle_pandas_ta_errors
    def trix(data: Union[np.ndarray, pd.Series], length: int = 30) -> np.ndarray:
        """TRIX"""
        series = to_pandas_series(data)
        validate_series_data(series, length)
        result = ta.trix(series, length=length)
        return result[f"TRIX_{length}_9"].values

    @staticmethod
    @handle_pandas_ta_errors
    def ultosc(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        fast: int = 7,
        medium: int = 14,
        slow: int = 28,
    ) -> np.ndarray:
        """アルティメットオシレーター"""
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        validate_series_data(high_series, slow)
        validate_series_data(low_series, slow)
        validate_series_data(close_series, slow)

        result = ta.ultosc(
            high=high_series,
            low=low_series,
            close=close_series,
            fast=fast,
            medium=medium,
            slow=slow,
        )
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def bop(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """バランスオブパワー"""
        open_series = to_pandas_series(open_data)
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        validate_series_data(open_series, 1)
        validate_series_data(high_series, 1)
        validate_series_data(low_series, 1)
        validate_series_data(close_series, 1)

        result = ta.bop(
            open=open_series, high=high_series, low=low_series, close=close_series
        )
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def apo(
        data: Union[np.ndarray, pd.Series],
        fast: int = 12,
        slow: int = 26,
    ) -> np.ndarray:
        """アブソリュートプライスオシレーター"""
        series = to_pandas_series(data)
        validate_series_data(series, slow)
        result = ta.apo(series, fast=fast, slow=slow)
        return result.values

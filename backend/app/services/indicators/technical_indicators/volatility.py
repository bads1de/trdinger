"""
ボラティリティ系テクニカル指標（pandas-ta移行版）

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
    ensure_series_minimal_conversion,
    validate_series_data,
    validate_indicator_parameters,
)


class VolatilityIndicators:
    """
    ボラティリティ系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    @handle_pandas_ta_errors
    def atr(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """平均真の値幅"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)
        validate_series_data(close_series, length)

        result = ta.atr(
            high=high_series, low=low_series, close=close_series, length=length
        )
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def natr(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """正規化平均実効値幅"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)
        validate_series_data(close_series, length)

        result = ta.natr(
            high=high_series, low=low_series, close=close_series, length=length
        )
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def trange(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """真の値幅"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        validate_series_data(high_series, 1)
        validate_series_data(low_series, 1)
        validate_series_data(close_series, 1)

        result = ta.true_range(high=high_series, low=low_series, close=close_series)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def bbands(
        data: Union[np.ndarray, pd.Series], length: int = 20, std: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ボリンジャーバンド"""
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, length)
        result = ta.bbands(series, length=length, std=std)

        # 列名を動的に取得（pandas-taのバージョンによって異なる可能性がある）
        columns = result.columns.tolist()

        # 上位、中位、下位バンドを特定
        upper_col = [col for col in columns if "BBU" in col][0]
        middle_col = [col for col in columns if "BBM" in col][0]
        lower_col = [col for col in columns if "BBL" in col][0]

        return (
            result[upper_col].values,
            result[middle_col].values,
            result[lower_col].values,
        )

    @staticmethod
    @handle_pandas_ta_errors
    def stddev(
        data: Union[np.ndarray, pd.Series], length: int = 5, ddof: int = 1
    ) -> np.ndarray:
        """標準偏差"""
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, length)
        result = ta.stdev(series, length=length, ddof=ddof)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def var(
        data: Union[np.ndarray, pd.Series], length: int = 5, ddof: int = 1
    ) -> np.ndarray:
        """分散"""
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, length)
        result = ta.variance(series, length=length, ddof=ddof)
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
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)
        validate_series_data(close_series, length)

        result = ta.adx(
            high=high_series, low=low_series, close=close_series, length=length
        )
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
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)
        validate_series_data(close_series, length)

        result = ta.adx(
            high=high_series, low=low_series, close=close_series, length=length
        )
        return result[f"ADXR_{length}"].values

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
        high_s = ensure_series_minimal_conversion(high)
        low_s = ensure_series_minimal_conversion(low)
        close_s = ensure_series_minimal_conversion(close)
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
    def minus_di(high, low, close, length: int = 14) -> np.ndarray:
        high_s = ensure_series_minimal_conversion(high)
        low_s = ensure_series_minimal_conversion(low)
        close_s = ensure_series_minimal_conversion(close)
        result = ta.adx(high=high_s, low=low_s, close=close_s, length=length)
        col = f"DMN_{length}"
        if col in result.columns:
            return result[col].values
        raise PandasTAError("MINUS_DI not available in this pandas-ta version")

    @staticmethod
    @handle_pandas_ta_errors
    def plus_di(high, low, close, length: int = 14) -> np.ndarray:
        high_s = ensure_series_minimal_conversion(high)
        low_s = ensure_series_minimal_conversion(low)
        close_s = ensure_series_minimal_conversion(close)
        result = ta.adx(high=high_s, low=low_s, close=close_s, length=length)
        col = f"DMP_{length}"
        if col in result.columns:
            return result[col].values
        raise PandasTAError("PLUS_DI not available in this pandas-ta version")

    @staticmethod
    @handle_pandas_ta_errors
    def minus_dm(high, low, length: int = 14) -> np.ndarray:
        high_s = ensure_series_minimal_conversion(high)
        low_s = ensure_series_minimal_conversion(low)
        result = ta.dm(high=high_s, low=low_s, length=length)
        # pandas-ta dm returns DMP and DMN columns
        cols = [c for c in result.columns if c.startswith("DMN_")]
        if cols:
            return result[cols[0]].values
        raise PandasTAError("MINUS_DM not available in this pandas-ta version")

    @staticmethod
    @handle_pandas_ta_errors
    def plus_dm(high, low, length: int = 14) -> np.ndarray:
        high_s = ensure_series_minimal_conversion(high)
        low_s = ensure_series_minimal_conversion(low)
        result = ta.dm(high=high_s, low=low_s, length=length)
        # pandas-ta dm returns DMP and DMN columns
        cols = [c for c in result.columns if c.startswith("DMP_")]
        if cols:
            return result[cols[0]].values
        raise PandasTAError("PLUS_DM not available in this pandas-ta version")

    @staticmethod
    @handle_pandas_ta_errors
    def aroon(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """アルーン"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)

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
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)

        result = ta.aroon(high=high_series, low=low_series, length=length)
        return result[f"AROONOSC_{length}"].values

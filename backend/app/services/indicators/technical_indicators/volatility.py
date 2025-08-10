"""
ボラティリティ系テクニカル指標（pandas-ta移行版）

このモジュールはpandas-taライブラリを使用し、
backtesting.pyとの完全な互換性を提供します。
numpy配列ベースのインターフェースを維持しています。
"""

from typing import Tuple, cast

import numpy as np

from ..utils import (
    PandasTAError,
    ensure_numpy_array,
    format_indicator_result,
    handle_pandas_ta_errors,
    validate_input,
    validate_multi_input,
)
from ..pandas_ta_utils import (
    atr as pandas_ta_atr,
    bbands as pandas_ta_bbands,
    stdev as pandas_ta_stdev,
    adx as pandas_ta_adx,
    natr as pandas_ta_natr,
    true_range as pandas_ta_true_range,
    variance as pandas_ta_variance,
    dx as pandas_ta_dx,
    plus_di as pandas_ta_plus_di,
    minus_di as pandas_ta_minus_di,
    plus_dm as pandas_ta_plus_dm,
    minus_dm as pandas_ta_minus_dm,
    aroon as pandas_ta_aroon,
    aroonosc as pandas_ta_aroonosc,
)


class VolatilityIndicators:
    """
    ボラティリティ系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    def atr(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Average True Range (平均真の値幅) - pandas-ta版

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            ATR値のnumpy配列
        """
        return pandas_ta_atr(high, low, close, period)

    @staticmethod
    def natr(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Normalized Average True Range (正規化平均真の値幅) - pandas-ta版

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            NATR値のnumpy配列
        """
        return pandas_ta_natr(high, low, close, period)

    @staticmethod
    def trange(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        True Range (真の値幅) - pandas-ta版

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            TRANGE値のnumpy配列
        """
        return pandas_ta_true_range(high, low, close)

    @staticmethod
    def bollinger_bands(
        data: np.ndarray, period: int = 20, std_dev: float = 2.0, matype: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bollinger Bands (ボリンジャーバンド) - pandas-ta版

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 20）
            std_dev: 標準偏差倍率（デフォルト: 2.0）
            matype: 移動平均種別（pandas-taでは無視される）

        Returns:
            (Upper Band, Middle Band, Lower Band)のtuple
        """
        return pandas_ta_bbands(data, period, std_dev)

    @staticmethod
    def stddev(data: np.ndarray, period: int = 5, nbdev: float = 1.0) -> np.ndarray:
        """
        Standard Deviation (標準偏差) - pandas-ta版

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 5）
            nbdev: 標準偏差倍率（デフォルト: 1.0）

        Returns:
            STDDEV値のnumpy配列
        """
        result = pandas_ta_stdev(data, period)
        # nbdevが1.0でない場合は倍率を適用
        if nbdev != 1.0:
            result = result * nbdev
        return result

    @staticmethod
    def var(data: np.ndarray, period: int = 5, nbdev: float = 1.0) -> np.ndarray:
        """
        Variance (分散) - pandas-ta版

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 5）
            nbdev: 標準偏差倍率（デフォルト: 1.0）

        Returns:
            VAR値のnumpy配列
        """
        result = pandas_ta_variance(data, period)
        # nbdevが1.0でない場合は倍率を適用
        if nbdev != 1.0:
            result = result * (nbdev**2)  # 分散なので二乗
        return result

    @staticmethod
    def adx(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Average Directional Movement Index (平均方向性指数) - pandas-ta版

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            ADX値のnumpy配列
        """
        return pandas_ta_adx(high, low, close, period)

    @staticmethod
    @handle_pandas_ta_errors
    def adxr(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Average Directional Movement Index Rating (ADX評価)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            ADXR値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        # ADXRはADXの変種として実装
        return pandas_ta_adx(high, low, close, period)

    @staticmethod
    @handle_pandas_ta_errors
    def dx(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Directional Movement Index (方向性指数)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            DX値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        return pandas_ta_dx(high, low, close, period)

    @staticmethod
    @handle_pandas_ta_errors
    def minus_di(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Minus Directional Indicator (マイナス方向性指標)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            MINUS_DI値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        return pandas_ta_minus_di(high, low, close, period)

    @staticmethod
    @handle_pandas_ta_errors
    def plus_di(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Plus Directional Indicator (プラス方向性指標)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            PLUS_DI値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        return pandas_ta_plus_di(high, low, close, period)

    @staticmethod
    def minus_dm(high: np.ndarray, low: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Minus Directional Movement (マイナス方向性移動) - pandas-ta版

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            MINUS_DM値のnumpy配列
        """
        return pandas_ta_minus_dm(high, low, period)

    @staticmethod
    def plus_dm(high: np.ndarray, low: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Plus Directional Movement (プラス方向性移動) - pandas-ta版

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            PLUS_DM値のnumpy配列
        """
        return pandas_ta_plus_dm(high, low, period)

    @staticmethod
    def aroon(
        high: np.ndarray, low: np.ndarray, period: int = 14
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aroon (アルーン) - pandas-ta版

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            (Aroon Down, Aroon Up)のtuple
        """
        return pandas_ta_aroon(high, low, period)

    @staticmethod
    def aroonosc(high: np.ndarray, low: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Aroon Oscillator (アルーンオシレーター) - pandas-ta版

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            AROONOSC値のnumpy配列
        """
        return pandas_ta_aroonosc(high, low, period)

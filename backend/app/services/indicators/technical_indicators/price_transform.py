"""
価格変換系テクニカル指標（pandas-ta移行版）

このモジュールはpandas-taライブラリを使用し、
backtesting.pyとの完全な互換性を提供します。
numpy配列ベースのインターフェースを維持しています。
"""

from typing import cast

import numpy as np

from ..utils import (
    TALibError,
    ensure_numpy_array,
    format_indicator_result,
    handle_talib_errors,
    validate_input,
    validate_multi_input,
)
from ..pandas_ta_utils import (
    ohlc4 as pandas_ta_ohlc4,
    hl2 as pandas_ta_hl2,
    hlc3 as pandas_ta_hlc3,
    wcp as pandas_ta_wcp,
)


class PriceTransformIndicators:
    """
    価格変換系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、pandas-taの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    def avgprice(
        open_data: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Average Price (平均価格) - pandas-ta版

        Args:
            open_data: 始値データ（numpy配列）
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            AVGPRICE値のnumpy配列
        """
        return pandas_ta_ohlc4(open_data, high, low, close)

    @staticmethod
    def medprice(high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """
        Median Price (中央値価格) - pandas-ta版

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）

        Returns:
            MEDPRICE値のnumpy配列
        """
        return pandas_ta_hl2(high, low)

    @staticmethod
    def typprice(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Typical Price (典型価格) - pandas-ta版

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            TYPPRICE値のnumpy配列
        """
        return pandas_ta_hlc3(high, low, close)

    @staticmethod
    def wclprice(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Weighted Close Price (加重終値価格) - pandas-ta版

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            WCLPRICE値のnumpy配列
        """
        return pandas_ta_wcp(high, low, close)

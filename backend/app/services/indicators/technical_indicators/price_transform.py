"""
価格変換系テクニカル指標

このモジュールはnumpy配列ベースでTa-libを直接使用し、
backtesting.pyとの完全な互換性を提供します。
pandas Seriesの変換は一切行いません。
"""

from typing import cast

import numpy as np
import talib

from ..utils import (
    TALibError,
    ensure_numpy_array,
    format_indicator_result,
    handle_talib_errors,
    validate_input,
    validate_multi_input,
)


class PriceTransformIndicators:
    """
    価格変換系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    @handle_talib_errors
    def avgprice(
        open_data: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Average Price (平均価格)

        Args:
            open_data: 始値データ（numpy配列）
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            AVGPRICE値のnumpy配列
        """
        open_data = ensure_numpy_array(open_data)
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)

        # 全データの長さが一致することを確認
        if not (len(open_data) == len(high) == len(low) == len(close)):
            raise TALibError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 1)
        result = talib.AVGPRICE(open_data, high, low, close)
        return cast(np.ndarray, format_indicator_result(result, "AVGPRICE"))

    @staticmethod
    @handle_talib_errors
    def medprice(high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """
        Median Price (中央値価格)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）

        Returns:
            MEDPRICE値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        validate_multi_input(high, low, high, 1)
        result = talib.MEDPRICE(high, low)
        return cast(np.ndarray, format_indicator_result(result, "MEDPRICE"))

    @staticmethod
    @handle_talib_errors
    def typprice(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Typical Price (典型価格)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            TYPPRICE値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, 1)
        result = talib.TYPPRICE(high, low, close)
        return cast(np.ndarray, format_indicator_result(result, "TYPPRICE"))

    @staticmethod
    @handle_talib_errors
    def wclprice(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Weighted Close Price (加重終値価格)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            WCLPRICE値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, 1)
        result = talib.WCLPRICE(high, low, close)
        return cast(np.ndarray, format_indicator_result(result, "WCLPRICE"))

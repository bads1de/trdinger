"""
数学演算子系テクニカル指標（NumPy標準関数版）

このモジュールはNumPy標準関数を使用し、
backtesting.pyとの完全な互換性を提供します。
TA-libの数学演算子関数をNumPy標準関数で置き換えています。
"""

import logging
import numpy as np
from typing import Union

logger = logging.getLogger(__name__)


def _validate_input(data: Union[np.ndarray, list], min_length: int = 1) -> np.ndarray:
    """入力データの検証とnumpy配列への変換"""
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=float)

    if len(data) < min_length:
        raise ValueError(f"データ長が不足: 必要{min_length}, 実際{len(data)}")

    return data


def _validate_dual_input(data0: np.ndarray, data1: np.ndarray) -> None:
    """2つの入力データの長さ一致確認"""
    if len(data0) != len(data1):
        raise ValueError(
            f"データの長さが一致しません。Data0: {len(data0)}, Data1: {len(data1)}"
        )


class MathOperatorsIndicators:
    """
    数学演算子系指標クラス（NumPy標準関数版）

    全ての指標はNumPy標準関数を使用し、TA-libへの依存を排除しています。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    def add(data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        """Vector Arithmetic Add (ベクトル算術加算)"""
        data0 = _validate_input(data0)
        data1 = _validate_input(data1)
        _validate_dual_input(data0, data1)
        return data0 + data1

    @staticmethod
    def sub(data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        """Vector Arithmetic Subtraction (ベクトル算術減算)"""
        data0 = _validate_input(data0)
        data1 = _validate_input(data1)
        _validate_dual_input(data0, data1)
        return data0 - data1

    @staticmethod
    def mult(data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        """Vector Arithmetic Multiplication (ベクトル算術乗算)"""
        data0 = _validate_input(data0)
        data1 = _validate_input(data1)
        _validate_dual_input(data0, data1)
        return data0 * data1

    @staticmethod
    def div(data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        """Vector Arithmetic Division (ベクトル算術除算)"""
        data0 = _validate_input(data0)
        data1 = _validate_input(data1)
        _validate_dual_input(data0, data1)

        # ゼロ除算の警告
        if np.any(data1 == 0):
            logger.warning("DIV: ゼロ除算が発生する可能性があります")

        return np.divide(data0, data1)

    @staticmethod
    def max_value(data: np.ndarray, period: int) -> np.ndarray:
        """Highest value over a specified period (指定期間の最大値)"""
        data = _validate_input(data, period)
        result = np.full_like(data, np.nan)

        for i in range(period - 1, len(data)):
            result[i] = np.max(data[i - period + 1 : i + 1])

        return result

    @staticmethod
    def min_value(data: np.ndarray, period: int) -> np.ndarray:
        """Lowest value over a specified period (指定期間の最小値)"""
        data = _validate_input(data, period)
        result = np.full_like(data, np.nan)

        for i in range(period - 1, len(data)):
            result[i] = np.min(data[i - period + 1 : i + 1])

        return result

    @staticmethod
    def sum_values(data: np.ndarray, period: int) -> np.ndarray:
        """Summation (合計)"""
        data = _validate_input(data, period)
        result = np.full_like(data, np.nan)

        for i in range(period - 1, len(data)):
            result[i] = np.sum(data[i - period + 1 : i + 1])

        return result

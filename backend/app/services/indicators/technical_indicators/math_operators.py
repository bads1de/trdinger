"""
数学演算子系テクニカル指標（NumPy標準関数版）

このモジュールはNumPy標準関数を使用し、
backtesting.pyとの完全な互換性を提供します。
TA-libの数学演算子関数をNumPy標準関数で置き換えています。
"""

import logging
import numpy as np
import pandas as pd
from typing import Union

from ..utils import validate_numpy_input, validate_numpy_dual_input

logger = logging.getLogger(__name__)


class MathOperatorsIndicators:
    """
    数学演算子系指標クラス（NumPy標準関数版）

    全ての指標はNumPy標準関数を使用し、TA-libへの依存を排除しています。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    def add(
        data0: Union[np.ndarray, pd.Series], data1: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Vector Arithmetic Add (ベクトル算術加算)"""
        data0 = validate_numpy_input(data0)
        data1 = validate_numpy_input(data1)
        validate_numpy_dual_input(data0, data1)
        return data0 + data1

    @staticmethod
    def sub(
        data0: Union[np.ndarray, pd.Series], data1: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Vector Arithmetic Subtraction (ベクトル算術減算)"""
        data0 = validate_numpy_input(data0)
        data1 = validate_numpy_input(data1)
        validate_numpy_dual_input(data0, data1)
        return data0 - data1

    @staticmethod
    def mult(
        data0: Union[np.ndarray, pd.Series], data1: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Vector Arithmetic Multiplication (ベクトル算術乗算)"""
        data0 = validate_numpy_input(data0)
        data1 = validate_numpy_input(data1)
        validate_numpy_dual_input(data0, data1)
        return data0 * data1

    @staticmethod
    def div(
        data0: Union[np.ndarray, pd.Series], data1: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Vector Arithmetic Division (ベクトル算術除算)"""
        data0 = validate_numpy_input(data0)
        data1 = validate_numpy_input(data1)
        validate_numpy_dual_input(data0, data1)

        # ゼロ除算の警告
        if np.any(data1 == 0):
            logger.warning("DIV: ゼロ除算が発生する可能性があります")

        return np.divide(data0, data1)

    @staticmethod
    def max_value(data: np.ndarray, period: int) -> np.ndarray:
        """Highest value over a specified period (指定期間の最大値)"""
        data = validate_numpy_input(data, period)
        result = np.full_like(data, np.nan)

        for i in range(period - 1, len(data)):
            result[i] = np.max(data[i - period + 1 : i + 1])

        return result

    @staticmethod
    def min_value(data: np.ndarray, period: int) -> np.ndarray:
        """Lowest value over a specified period (指定期間の最小値)"""
        data = validate_numpy_input(data, period)
        result = np.full_like(data, np.nan)

        for i in range(period - 1, len(data)):
            result[i] = np.min(data[i - period + 1 : i + 1])

        return result

    @staticmethod
    def sum_values(data: np.ndarray, period: int) -> np.ndarray:
        """Summation (合計)"""
        data = validate_numpy_input(data, period)
        result = np.full_like(data, np.nan)

        for i in range(period - 1, len(data)):
            result[i] = np.sum(data[i - period + 1 : i + 1])

        return result

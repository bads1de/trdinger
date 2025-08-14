"""
数学変換系テクニカル指標（NumPy標準関数版）

このモジュールはNumPy標準関数を使用し、
backtesting.pyとの完全な互換性を提供します。
数学変換関数をNumPy標準関数で置き換えています。
"""

import logging
import numpy as np
import pandas as pd
from typing import Union

from ..utils import validate_numpy_input

logger = logging.getLogger(__name__)


class MathTransformIndicators:
    """
    数学変換系指標クラス（NumPy標準関数版）

    全ての指標はNumPy標準関数を使用し、依存を排除しています。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    def acos(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Vector Trigonometric ACos (ベクトル三角関数ACos)

        Args:
            data: 入力データ（numpy配列）

        Returns:
            ACOS値のnumpy配列
        """
        data = validate_numpy_input(data)

        # 入力データの範囲チェックとクリッピング
        min_val, max_val = np.nanmin(data), np.nanmax(data)
        if min_val < -1.0 or max_val > 1.0:
            logger.warning(
                f"ACOS入力値が範囲外: [{min_val:.6f}, {max_val:.6f}] -> [-1, 1]にクリップ"
            )
            data = np.clip(data, -1.0, 1.0)

        return np.arccos(data)

    @staticmethod
    def asin(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Vector Trigonometric ASin (ベクトル三角関数ASin)"""
        data = validate_numpy_input(data)

        min_val, max_val = np.nanmin(data), np.nanmax(data)
        if min_val < -1.0 or max_val > 1.0:
            logger.warning(
                f"ASIN入力値が範囲外: [{min_val:.6f}, {max_val:.6f}] -> [-1, 1]にクリップ"
            )
            data = np.clip(data, -1.0, 1.0)

        return np.arcsin(data)

    @staticmethod
    def atan(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Vector Trigonometric ATan (ベクトル三角関数ATan)"""
        data = validate_numpy_input(data)
        return np.arctan(data)

    @staticmethod
    def ceil(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Vector Ceiling (ベクトル天井関数)"""
        data = validate_numpy_input(data)
        return np.ceil(data)

    @staticmethod
    def cos(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Vector Trigonometric Cos (ベクトル三角関数Cos)"""
        data = validate_numpy_input(data)
        return np.cos(data)

    @staticmethod
    def cosh(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Vector Trigonometric Cosh (ベクトル双曲線余弦)"""
        data = validate_numpy_input(data)
        return np.cosh(data)

    @staticmethod
    def exp(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Vector Arithmetic Exp (ベクトル指数関数)"""
        data = validate_numpy_input(data)
        return np.exp(data)

    @staticmethod
    def floor(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Vector Floor (ベクトル床関数)"""
        data = validate_numpy_input(data)
        return np.floor(data)

    @staticmethod
    def ln(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Vector Log Natural (ベクトル自然対数)"""
        data = validate_numpy_input(data)

        # 負の値や0の値をチェック
        if np.any(data <= 0):
            logger.warning(
                "LN: 負の値または0が含まれています。NaNが生成される可能性があります。"
            )

        return np.log(data)

    @staticmethod
    def log10(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Vector Log10 (ベクトル常用対数)"""
        data = validate_numpy_input(data)

        if np.any(data <= 0):
            logger.warning(
                "LOG10: 負の値または0が含まれています。NaNが生成される可能性があります。"
            )

        return np.log10(data)

    @staticmethod
    def sin(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Vector Trigonometric Sin (ベクトル三角関数Sin)"""
        data = validate_numpy_input(data)
        return np.sin(data)

    @staticmethod
    def sinh(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Vector Trigonometric Sinh (ベクトル双曲線正弦)"""
        data = validate_numpy_input(data)
        return np.sinh(data)

    @staticmethod
    def sqrt(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Vector Square Root (ベクトル平方根)"""
        data = validate_numpy_input(data)

        if np.any(data < 0):
            logger.warning(
                "SQRT: 負の値が含まれています。NaNが生成される可能性があります。"
            )

        return np.sqrt(data)

    @staticmethod
    def tan(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Vector Trigonometric Tan (ベクトル三角関数Tan)"""
        data = validate_numpy_input(data)
        return np.tan(data)

    @staticmethod
    def tanh(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Vector Trigonometric Tanh (ベクトル双曲線正接)"""
        data = validate_numpy_input(data)
        return np.tanh(data)

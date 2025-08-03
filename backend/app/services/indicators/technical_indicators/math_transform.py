"""
数学変換系テクニカル指標

このモジュールはnumpy配列ベースでTa-libを直接使用し、
backtesting.pyとの完全な互換性を提供します。
pandas Seriesの変換は一切行いません。
"""

import talib
import numpy as np
from typing import cast
from ..utils import (
    validate_input,
    handle_talib_errors,
    format_indicator_result,
    ensure_numpy_array,
)


class MathTransformIndicators:
    """
    数学変換系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    @handle_talib_errors
    def acos(data: np.ndarray) -> np.ndarray:
        """
        Vector Trigonometric ACos (ベクトル三角関数ACos)

        Args:
            data: 入力データ（numpy配列）

        Returns:
            ACOS値のnumpy配列
        """
        import logging

        logger = logging.getLogger(__name__)

        data = ensure_numpy_array(data)
        validate_input(data, 1)
        # 入力データの範囲チェックとクリッピング
        min_val, max_val = np.nanmin(data), np.nanmax(data)
        if min_val < -1.0 or max_val > 1.0:
            logger.warning(
                f"ACOS input data is out of the valid range [-1, 1]. Range: [{min_val:.6f}, {max_val:.6f}]. Clipping to valid range."
            )
            # 範囲外の値を[-1, 1]にクリップ
            data = np.clip(data, -1.0, 1.0)

        result = talib.ACOS(data)
        return cast(np.ndarray, format_indicator_result(result, "ACOS"))

    @staticmethod
    @handle_talib_errors
    def asin(data: np.ndarray) -> np.ndarray:
        """
        Vector Trigonometric ASin (ベクトル三角関数ASin)

        Args:
            data: 入力データ（numpy配列）

        Returns:
            ASIN値のnumpy配列
        """
        import logging

        logger = logging.getLogger(__name__)

        data = ensure_numpy_array(data)
        validate_input(data, 1)
        # 入力データの範囲チェックとクリッピング
        min_val, max_val = np.nanmin(data), np.nanmax(data)
        if min_val < -1.0 or max_val > 1.0:
            logger.warning(
                f"ASIN input data is out of the valid range [-1, 1]. Range: [{min_val:.6f}, {max_val:.6f}]. Clipping to valid range."
            )
            # 範囲外の値を[-1, 1]にクリップ
            data = np.clip(data, -1.0, 1.0)

        result = talib.ASIN(data)
        return cast(np.ndarray, format_indicator_result(result, "ASIN"))

    @staticmethod
    @handle_talib_errors
    def atan(data: np.ndarray) -> np.ndarray:
        """
        Vector Trigonometric ATan (ベクトル三角関数ATan)

        Args:
            data: 入力データ（numpy配列）

        Returns:
            ATAN値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, 1)
        result = talib.ATAN(data)
        return cast(np.ndarray, format_indicator_result(result, "ATAN"))

    @staticmethod
    @handle_talib_errors
    def ceil(data: np.ndarray) -> np.ndarray:
        """
        Vector Ceil (ベクトル天井関数)

        Args:
            data: 入力データ（numpy配列）

        Returns:
            CEIL値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, 1)
        result = talib.CEIL(data)
        return cast(np.ndarray, format_indicator_result(result, "CEIL"))

    @staticmethod
    @handle_talib_errors
    def cos(data: np.ndarray) -> np.ndarray:
        """
        Vector Trigonometric Cos (ベクトル三角関数Cos)

        Args:
            data: 入力データ（numpy配列）

        Returns:
            COS値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, 1)
        result = talib.COS(data)
        return cast(np.ndarray, format_indicator_result(result, "COS"))

    @staticmethod
    @handle_talib_errors
    def cosh(data: np.ndarray) -> np.ndarray:
        """
        Vector Trigonometric Cosh (ベクトル双曲線関数Cosh)

        Args:
            data: 入力データ（numpy配列）

        Returns:
            COSH値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, 1)
        result = talib.COSH(data)
        return cast(np.ndarray, format_indicator_result(result, "COSH"))

    @staticmethod
    @handle_talib_errors
    def exp(data: np.ndarray) -> np.ndarray:
        """
        Vector Arithmetic Exp (ベクトル指数関数)

        Args:
            data: 入力データ（numpy配列）

        Returns:
            EXP値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, 1)
        result = talib.EXP(data)
        return cast(np.ndarray, format_indicator_result(result, "EXP"))

    @staticmethod
    @handle_talib_errors
    def floor(data: np.ndarray) -> np.ndarray:
        """
        Vector Floor (ベクトル床関数)

        Args:
            data: 入力データ（numpy配列）

        Returns:
            FLOOR値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, 1)
        result = talib.FLOOR(data)
        return cast(np.ndarray, format_indicator_result(result, "FLOOR"))

    @staticmethod
    @handle_talib_errors
    def ln(data: np.ndarray) -> np.ndarray:
        """
        Vector Log Natural (ベクトル自然対数)

        Args:
            data: 入力データ（numpy配列）

        Returns:
            LN値のnumpy配列
        """
        import logging

        logger = logging.getLogger(__name__)

        data = ensure_numpy_array(data)
        validate_input(data, 1)
        # 入力データの範囲チェック（正の値のみ有効）
        min_val = np.nanmin(data)
        if min_val <= 0.0:
            logger.warning(
                f"LN input data contains non-positive values. Min value: {min_val:.6f}. Replacing with small positive value."
            )
            # 0以下の値を小さな正の値に置換
            data = np.where(data <= 0.0, 1e-10, data)

        result = talib.LN(data)
        return cast(np.ndarray, format_indicator_result(result, "LN"))

    @staticmethod
    @handle_talib_errors
    def log10(data: np.ndarray) -> np.ndarray:
        """
        Vector Log10 (ベクトル常用対数)

        Args:
            data: 入力データ（numpy配列）

        Returns:
            LOG10値のnumpy配列
        """
        import logging

        logger = logging.getLogger(__name__)

        data = ensure_numpy_array(data)
        validate_input(data, 1)
        # 入力データの範囲チェック（正の値のみ有効）
        min_val = np.nanmin(data)
        if min_val <= 0.0:
            logger.warning(
                f"LOG10 input data contains non-positive values. Min value: {min_val:.6f}. Replacing with small positive value."
            )
            # 0以下の値を小さな正の値に置換
            data = np.where(data <= 0.0, 1e-10, data)

        result = talib.LOG10(data)
        return cast(np.ndarray, format_indicator_result(result, "LOG10"))

    @staticmethod
    @handle_talib_errors
    def sin(data: np.ndarray) -> np.ndarray:
        """
        Vector Trigonometric Sin (ベクトル三角関数Sin)

        Args:
            data: 入力データ（numpy配列）

        Returns:
            SIN値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, 1)
        result = talib.SIN(data)
        return cast(np.ndarray, format_indicator_result(result, "SIN"))

    @staticmethod
    @handle_talib_errors
    def sinh(data: np.ndarray) -> np.ndarray:
        """
        Vector Trigonometric Sinh (ベクトル双曲線関数Sinh)

        Args:
            data: 入力データ（numpy配列）

        Returns:
            SINH値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, 1)
        result = talib.SINH(data)
        return cast(np.ndarray, format_indicator_result(result, "SINH"))

    @staticmethod
    @handle_talib_errors
    def sqrt(data: np.ndarray) -> np.ndarray:
        """
        Vector Square Root (ベクトル平方根)

        Args:
            data: 入力データ（numpy配列）

        Returns:
            SQRT値のnumpy配列
        """
        import logging

        logger = logging.getLogger(__name__)

        data = ensure_numpy_array(data)
        validate_input(data, 1)
        # 入力データの範囲チェック（非負の値のみ有効）
        min_val = np.nanmin(data)
        if min_val < 0.0:
            logger.warning(
                f"SQRT input data contains negative values. Min value: {min_val:.6f}. Replacing with 0."
            )
            # 負の値を0に置換
            data = np.where(data < 0.0, 0.0, data)

        result = talib.SQRT(data)
        return cast(np.ndarray, format_indicator_result(result, "SQRT"))

    @staticmethod
    @handle_talib_errors
    def tan(data: np.ndarray) -> np.ndarray:
        """
        Vector Trigonometric Tan (ベクトル三角関数Tan)

        Args:
            data: 入力データ（numpy配列）

        Returns:
            TAN値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, 1)
        result = talib.TAN(data)
        return cast(np.ndarray, format_indicator_result(result, "TAN"))

    @staticmethod
    @handle_talib_errors
    def tanh(data: np.ndarray) -> np.ndarray:
        """
        Vector Trigonometric Tanh (ベクトル双曲線関数Tanh)

        Args:
            data: 入力データ（numpy配列）

        Returns:
            TANH値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, 1)
        result = talib.TANH(data)
        return cast(np.ndarray, format_indicator_result(result, "TANH"))

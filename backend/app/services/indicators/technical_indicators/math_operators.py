"""
数学演算子系テクニカル指標

このモジュールはnumpy配列ベースでTa-libを直接使用し、
backtesting.pyとの完全な互換性を提供します。
pandas Seriesの変換は一切行いません。
"""

from typing import Tuple, cast

import numpy as np
import talib

from ..utils import (
    TALibError,
    ensure_numpy_array,
    format_indicator_result,
    handle_talib_errors,
    validate_input,
)


class MathOperatorsIndicators:
    """
    数学演算子系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    @handle_talib_errors
    def add(data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        """
        Vector Arithmetic Add (ベクトル算術加算)

        Args:
            data0: 第1データ（numpy配列）
            data1: 第2データ（numpy配列）

        Returns:
            ADD値のnumpy配列
        """
        data0 = ensure_numpy_array(data0)
        data1 = ensure_numpy_array(data1)

        if len(data0) != len(data1):
            raise TALibError(
                f"データの長さが一致しません。Data0: {len(data0)}, Data1: {len(data1)}"
            )

        validate_input(data0, 1)
        result = talib.ADD(data0, data1)
        return cast(np.ndarray, format_indicator_result(result, "ADD"))

    @staticmethod
    @handle_talib_errors
    def div(data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        """
        Vector Arithmetic Div (ベクトル算術除算)

        Args:
            data0: 第1データ（numpy配列）
            data1: 第2データ（numpy配列）

        Returns:
            DIV値のnumpy配列
        """
        data0 = ensure_numpy_array(data0)
        data1 = ensure_numpy_array(data1)

        if len(data0) != len(data1):
            raise TALibError(
                f"データの長さが一致しません。Data0: {len(data0)}, Data1: {len(data1)}"
            )

        validate_input(data0, 1)
        result = talib.DIV(data0, data1)
        return cast(np.ndarray, format_indicator_result(result, "DIV"))

    @staticmethod
    @handle_talib_errors
    def max(data: np.ndarray, period: int = 30) -> np.ndarray:
        """
        Highest value over a specified period (指定期間の最高値)

        Args:
            data: 入力データ（numpy配列）
            period: 期間（デフォルト: 30）

        Returns:
            MAX値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        result = talib.MAX(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "MAX"))

    @staticmethod
    @handle_talib_errors
    def maxindex(data: np.ndarray, period: int = 30) -> np.ndarray:
        """
        Index of highest value over a specified period (指定期間の最高値のインデックス)

        Args:
            data: 入力データ（numpy配列）
            period: 期間（デフォルト: 30）

        Returns:
            MAXINDEX値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        result = talib.MAXINDEX(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "MAXINDEX"))

    @staticmethod
    @handle_talib_errors
    def min(data: np.ndarray, period: int = 30) -> np.ndarray:
        """
        Lowest value over a specified period (指定期間の最低値)

        Args:
            data: 入力データ（numpy配列）
            period: 期間（デフォルト: 30）

        Returns:
            MIN値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        result = talib.MIN(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "MIN"))

    @staticmethod
    @handle_talib_errors
    def minindex(data: np.ndarray, period: int = 30) -> np.ndarray:
        """
        Index of lowest value over a specified period (指定期間の最低値のインデックス)

        Args:
            data: 入力データ（numpy配列）
            period: 期間（デフォルト: 30）

        Returns:
            MININDEX値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        result = talib.MININDEX(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "MININDEX"))

    @staticmethod
    @handle_talib_errors
    def minmax(data: np.ndarray, period: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lowest and highest values over a specified period (指定期間の最低値と最高値)

        Args:
            data: 入力データ（numpy配列）
            period: 期間（デフォルト: 30）

        Returns:
            (MIN, MAX)のtuple
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        min_val, max_val = talib.MINMAX(data, timeperiod=period)
        return cast(
            Tuple[np.ndarray, np.ndarray],
            format_indicator_result((min_val, max_val), "MINMAX"),
        )

    @staticmethod
    @handle_talib_errors
    def minmaxindex(
        data: np.ndarray, period: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Indexes of lowest and highest values over a specified period (指定期間の最低値と最高値のインデックス)

        Args:
            data: 入力データ（numpy配列）
            period: 期間（デフォルト: 30）

        Returns:
            (MININDEX, MAXINDEX)のtuple
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        min_idx, max_idx = talib.MINMAXINDEX(data, timeperiod=period)
        return cast(
            Tuple[np.ndarray, np.ndarray],
            format_indicator_result((min_idx, max_idx), "MINMAXINDEX"),
        )

    @staticmethod
    @handle_talib_errors
    def mult(data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        """
        Vector Arithmetic Mult (ベクトル算術乗算)

        Args:
            data0: 第1データ（numpy配列）
            data1: 第2データ（numpy配列）

        Returns:
            MULT値のnumpy配列
        """
        data0 = ensure_numpy_array(data0)
        data1 = ensure_numpy_array(data1)

        if len(data0) != len(data1):
            raise TALibError(
                f"データの長さが一致しません。Data0: {len(data0)}, Data1: {len(data1)}"
            )

        validate_input(data0, 1)
        result = talib.MULT(data0, data1)
        return cast(np.ndarray, format_indicator_result(result, "MULT"))

    @staticmethod
    @handle_talib_errors
    def sub(data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        """
        Vector Arithmetic Subtraction (ベクトル算術減算)

        Args:
            data0: 第1データ（numpy配列）
            data1: 第2データ（numpy配列）

        Returns:
            SUB値のnumpy配列
        """
        data0 = ensure_numpy_array(data0)
        data1 = ensure_numpy_array(data1)

        if len(data0) != len(data1):
            raise TALibError(
                f"データの長さが一致しません。Data0: {len(data0)}, Data1: {len(data1)}"
            )

        validate_input(data0, 1)
        result = talib.SUB(data0, data1)
        return cast(np.ndarray, format_indicator_result(result, "SUB"))

    @staticmethod
    @handle_talib_errors
    def sum(data: np.ndarray, period: int = 30) -> np.ndarray:
        """
        Summation (合計)

        Args:
            data: 入力データ（numpy配列）
            period: 期間（デフォルト: 30）

        Returns:
            SUM値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        result = talib.SUM(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "SUM"))

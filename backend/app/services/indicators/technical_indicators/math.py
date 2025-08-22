"""
数学系テクニカル指標

数学関数に基づくテクニカル指標を実装します。
"""

from typing import Union
import numpy as np
import pandas as pd

from ..utils import handle_pandas_ta_errors


class MathIndicators:
    """
    数学系指標クラス
    """

    @staticmethod
    @handle_pandas_ta_errors
    def acos(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """逆余弦関数"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        # データを[-1, 1]の範囲にクリップ
        clipped = np.clip(data_series.values, -1, 1)
        return np.arccos(clipped)

    @staticmethod
    @handle_pandas_ta_errors
    def asin(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """逆正弦関数"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        # データを[-1, 1]の範囲にクリップ
        clipped = np.clip(data_series.values, -1, 1)
        return np.arcsin(clipped)

    @staticmethod
    @handle_pandas_ta_errors
    def atan(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """逆正接関数"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return np.arctan(data_series.values)

    @staticmethod
    @handle_pandas_ta_errors
    def ceil(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """天井関数（小数点以下切り上げ）"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return np.ceil(data_series.values)

    @staticmethod
    @handle_pandas_ta_errors
    def cos(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """余弦関数"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return np.cos(data_series.values)

    @staticmethod
    @handle_pandas_ta_errors
    def cosh(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """双曲線余弦関数"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return np.cosh(data_series.values)

    @staticmethod
    @handle_pandas_ta_errors
    def exp(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """指数関数"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return np.exp(data_series.values)

    @staticmethod
    @handle_pandas_ta_errors
    def floor(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """床関数（小数点以下切り捨て）"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return np.floor(data_series.values)

    @staticmethod
    @handle_pandas_ta_errors
    def ln(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """自然対数"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        # 正の値のみに制限
        positive_data = np.where(data_series.values > 0, data_series.values, np.nan)
        return np.log(positive_data)

    @staticmethod
    @handle_pandas_ta_errors
    def log10(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """常用対数"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        # 正の値のみに制限
        positive_data = np.where(data_series.values > 0, data_series.values, np.nan)
        return np.log10(positive_data)

    @staticmethod
    @handle_pandas_ta_errors
    def sin(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """正弦関数"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return np.sin(data_series.values)

    @staticmethod
    @handle_pandas_ta_errors
    def sinh(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """双曲線正弦関数"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return np.sinh(data_series.values)

    @staticmethod
    @handle_pandas_ta_errors
    def sqrt(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """平方根"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        # 非負の値のみに制限
        non_negative_data = np.where(data_series.values >= 0, data_series.values, np.nan)
        return np.sqrt(non_negative_data)

    @staticmethod
    @handle_pandas_ta_errors
    def tan(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """正接関数"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return np.tan(data_series.values)

    @staticmethod
    @handle_pandas_ta_errors
    def tanh(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """双曲線正接関数"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return np.tanh(data_series.values)

    # 数学演算子
    @staticmethod
    @handle_pandas_ta_errors
    def add(data1: Union[np.ndarray, pd.Series], data2: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """加算"""
        data1_series = pd.Series(data1) if isinstance(data1, np.ndarray) else data1
        data2_series = pd.Series(data2) if isinstance(data2, np.ndarray) else data2
        return (data1_series + data2_series).values

    @staticmethod
    @handle_pandas_ta_errors
    def sub(data1: Union[np.ndarray, pd.Series], data2: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """減算"""
        data1_series = pd.Series(data1) if isinstance(data1, np.ndarray) else data1
        data2_series = pd.Series(data2) if isinstance(data2, np.ndarray) else data2
        return (data1_series - data2_series).values

    @staticmethod
    @handle_pandas_ta_errors
    def mult(data1: Union[np.ndarray, pd.Series], data2: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """乗算"""
        data1_series = pd.Series(data1) if isinstance(data1, np.ndarray) else data1
        data2_series = pd.Series(data2) if isinstance(data2, np.ndarray) else data2
        return (data1_series * data2_series).values

    @staticmethod
    @handle_pandas_ta_errors
    def div(data1: Union[np.ndarray, pd.Series], data2: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """除算"""
        data1_series = pd.Series(data1) if isinstance(data1, np.ndarray) else data1
        data2_series = pd.Series(data2) if isinstance(data2, np.ndarray) else data2
        # ゼロ除算を避ける
        result = data1_series / data2_series.replace(0, np.nan)
        return result.values

    # 統計系関数
    @staticmethod
    @handle_pandas_ta_errors
    def max_value(data: Union[np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
        """最大値"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return data_series.rolling(window=period).max().values

    @staticmethod
    @handle_pandas_ta_errors
    def min_value(data: Union[np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
        """最小値"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return data_series.rolling(window=period).min().values

    @staticmethod
    @handle_pandas_ta_errors
    def sum_value(data: Union[np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
        """合計値"""
        data_series = pd.Series(data) if isinstance(data, np.ndarray) else data
        return data_series.rolling(window=period).sum().values
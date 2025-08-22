"""
統計系テクニカル指標

登録してあるテクニカルの一覧:
- Beta
- Correl (Pearson's Correlation Coefficient)
- LinearReg (Linear Regression)
- LinearReg_Angle (Linear Regression Angle)
- LinearReg_Intercept (Linear Regression Intercept)
- LinearReg_Slope (Linear Regression Slope)
- StdDev (Standard Deviation)
- TSF (Time Series Forecast)
- Var (Variance)
"""

import logging
import numpy as np
import pandas as pd
from typing import Union

from ..utils import validate_numpy_input, validate_numpy_dual_input

logger = logging.getLogger(__name__)


class StatisticsIndicators:
    """
    統計系指標クラス
    """

    @staticmethod
    def beta(high: np.ndarray, low: np.ndarray, length: int = 5) -> np.ndarray:
        """Beta (ベータ) - 簡易実装"""
        high = validate_numpy_input(high, length)
        low = validate_numpy_input(low, length)
        validate_numpy_dual_input(high, low)

        result = np.full_like(high, np.nan)

        for i in range(length - 1, len(high)):
            high_window = high[i - length + 1 : i + 1]
            low_window = low[i - length + 1 : i + 1]

            # 簡易ベータ計算（高値と安値の相関）
            if np.std(low_window) > 0:
                result[i] = np.corrcoef(high_window, low_window)[0, 1]
            else:
                result[i] = 0.0

        return result

    @staticmethod
    def correl(high: np.ndarray, low: np.ndarray, length: int = 30) -> np.ndarray:
        """Pearson's Correlation Coefficient (ピアソン相関係数)"""
        high = validate_numpy_input(high, length)
        low = validate_numpy_input(low, length)
        validate_numpy_dual_input(high, low)

        result = np.full_like(high, np.nan)

        for i in range(length - 1, len(high)):
            high_window = high[i - length + 1 : i + 1]
            low_window = low[i - length + 1 : i + 1]

            if np.std(high_window) > 0 and np.std(low_window) > 0:
                result[i] = np.corrcoef(high_window, low_window)[0, 1]
            else:
                result[i] = 0.0

        return result

    @staticmethod
    def linearreg(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
        """Linear Regression (線形回帰)"""
        data = validate_numpy_input(data, length)
        result = np.full_like(data, np.nan)

        for i in range(length - 1, len(data)):
            y = data[i - length + 1 : i + 1]
            x = np.arange(length)

            # 線形回帰の計算
            slope, intercept = np.polyfit(x, y, 1)
            result[i] = slope * (length - 1) + intercept

        return result

    @staticmethod
    def linearreg_angle(
        data: Union[np.ndarray, pd.Series], length: int = 14
    ) -> np.ndarray:
        """Linear Regression Angle (線形回帰角度)"""
        data = validate_numpy_input(data, length)
        result = np.full_like(data, np.nan)

        for i in range(length - 1, len(data)):
            y = data[i - length + 1 : i + 1]
            x = np.arange(length)

            # 線形回帰の傾きを角度に変換
            slope, _ = np.polyfit(x, y, 1)
            result[i] = np.arctan(slope) * 180 / np.pi

        return result

    @staticmethod
    def linearreg_intercept(
        data: Union[np.ndarray, pd.Series], length: int = 14
    ) -> np.ndarray:
        """Linear Regression Intercept (線形回帰切片)"""
        data = validate_numpy_input(data, length)
        result = np.full_like(data, np.nan)

        for i in range(length - 1, len(data)):
            y = data[i - length + 1 : i + 1]
            x = np.arange(length)

            # 線形回帰の切片
            _, intercept = np.polyfit(x, y, 1)
            result[i] = intercept

        return result

    @staticmethod
    def linearreg_slope(
        data: Union[np.ndarray, pd.Series], length: int = 14
    ) -> np.ndarray:
        """Linear Regression Slope (線形回帰傾き)"""
        data = validate_numpy_input(data, length)
        result = np.full_like(data, np.nan)

        for i in range(length - 1, len(data)):
            y = data[i - length + 1 : i + 1]
            x = np.arange(length)

            # 線形回帰の傾き
            slope, _ = np.polyfit(x, y, 1)
            result[i] = slope

        return result

    @staticmethod
    def stddev(
        data: Union[np.ndarray, pd.Series], length: int = 5, nbdev: float = 1.0
    ) -> np.ndarray:
        """Standard Deviation (標準偏差)"""
        data = validate_numpy_input(data, length)
        result = np.full_like(data, np.nan)

        for i in range(length - 1, len(data)):
            window = data[i - length + 1 : i + 1]
            result[i] = np.std(window, ddof=1) * nbdev

        return result

    @staticmethod
    def tsf(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
        """Time Series Forecast (時系列予測)"""
        data = validate_numpy_input(data, length)
        result = np.full_like(data, np.nan)

        for i in range(length - 1, len(data)):
            y = data[i - length + 1 : i + 1]
            x = np.arange(length)

            # 線形回帰による次の値の予測
            slope, intercept = np.polyfit(x, y, 1)
            result[i] = slope * length + intercept

        return result

    @staticmethod
    def var(
        data: Union[np.ndarray, pd.Series], length: int = 5, nbdev: float = 1.0
    ) -> np.ndarray:
        """Variance (分散)"""
        data = validate_numpy_input(data, length)
        result = np.full_like(data, np.nan)

        for i in range(length - 1, len(data)):
            window = data[i - length + 1 : i + 1]
            result[i] = np.var(window, ddof=1) * nbdev

        return result

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
- Entropy
- Kurtosis
- MAD (Mean Absolute Deviation)
- Median
- Quantile
- Skew
- TOS_STDEVALL (Think or Swim Standard Deviation All)
- ZScore (Z Score)
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

    @staticmethod
    def entropy(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """Entropy (エントロピー)"""
        data = validate_numpy_input(data, length)
        result = np.full_like(data, np.nan)

        for i in range(length - 1, len(data)):
            window = data[i - length + 1 : i + 1]
            if np.all(window == window[0]):  # 全て同じ値の場合
                result[i] = 0.0
            else:
                # データをビン分けしてエントロピーを計算
                hist, _ = np.histogram(window, bins=min(length//2, 10))
                hist = hist[hist > 0]  # ゼロを除去
                prob = hist / len(window)
                result[i] = -np.sum(prob * np.log2(prob))

        return result

    @staticmethod
    def kurtosis(data: Union[np.ndarray, pd.Series], length: int = 30) -> np.ndarray:
        """Kurtosis (尖度)"""
        data = validate_numpy_input(data, length)
        result = np.full_like(data, np.nan)

        for i in range(length - 1, len(data)):
            window = data[i - length + 1 : i + 1]
            if len(window) < 4:  # 尖度計算には最低4つのデータが必要
                continue
            # 尖度の計算
            mean_val = np.mean(window)
            std_val = np.std(window, ddof=1)
            if std_val > 0:
                result[i] = np.mean(((window - mean_val) / std_val) ** 4) - 3
            else:
                result[i] = 0.0

        return result

    @staticmethod
    def mad(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """Mean Absolute Deviation (平均絶対偏差)"""
        data = validate_numpy_input(data, length)
        result = np.full_like(data, np.nan)

        for i in range(length - 1, len(data)):
            window = data[i - length + 1 : i + 1]
            median_val = np.median(window)
            result[i] = np.mean(np.abs(window - median_val))

        return result

    @staticmethod
    def median(data: Union[np.ndarray, pd.Series], length: int = 30) -> np.ndarray:
        """Median (中央値)"""
        data = validate_numpy_input(data, length)
        result = np.full_like(data, np.nan)

        for i in range(length - 1, len(data)):
            window = data[i - length + 1 : i + 1]
            result[i] = np.median(window)

        return result

    @staticmethod
    def quantile(data: Union[np.ndarray, pd.Series], length: int = 30, q: float = 0.5) -> np.ndarray:
        """Quantile (分位数)"""
        data = validate_numpy_input(data, length)
        result = np.full_like(data, np.nan)

        for i in range(length - 1, len(data)):
            window = data[i - length + 1 : i + 1]
            result[i] = np.quantile(window, q)

        return result

    @staticmethod
    def skew(data: Union[np.ndarray, pd.Series], length: int = 30) -> np.ndarray:
        """Skewness (歪度)"""
        data = validate_numpy_input(data, length)
        result = np.full_like(data, np.nan)

        for i in range(length - 1, len(data)):
            window = data[i - length + 1 : i + 1]
            if len(window) < 3:  # 歪度計算には最低3つのデータが必要
                continue
            # 歪度の計算
            mean_val = np.mean(window)
            std_val = np.std(window, ddof=1)
            if std_val > 0:
                result[i] = np.mean(((window - mean_val) / std_val) ** 3)
            else:
                result[i] = 0.0

        return result

    @staticmethod
    def tos_stdevall(data: Union[np.ndarray, pd.Series], length: int = 30) -> np.ndarray:
        """Think or Swim Standard Deviation All (Think or Swim 全標準偏差)"""
        data = validate_numpy_input(data, length)
        result = np.full_like(data, np.nan)

        for i in range(length - 1, len(data)):
            # 現在の位置までの全データを対象
            window = data[:i + 1]
            if len(window) > 1:
                result[i] = np.std(window, ddof=1)
            else:
                result[i] = 0.0

        return result

    @staticmethod
    def zscore(data: Union[np.ndarray, pd.Series], length: int = 30) -> np.ndarray:
        """Z-Score (Zスコア)"""
        data = validate_numpy_input(data, length)
        result = np.full_like(data, np.nan)

        for i in range(length - 1, len(data)):
            window = data[i - length + 1 : i + 1]
            mean_val = np.mean(window)
            std_val = np.std(window, ddof=1)
            current_val = data[i]
            if std_val > 0:
                result[i] = (current_val - mean_val) / std_val
            else:
                result[i] = 0.0

        return result

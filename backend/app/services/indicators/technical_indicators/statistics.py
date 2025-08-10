"""
統計系テクニカル指標（NumPy標準関数版）

このモジュールはNumPy標準関数を使用し、
backtesting.pyとの完全な互換性を提供します。
TA-libの統計関数をNumPy標準関数で置き換えています。
"""

import logging
import numpy as np
import pandas as pd
from typing import Union

logger = logging.getLogger(__name__)


def _validate_input(
    data: Union[np.ndarray, pd.Series, list], min_length: int = 1
) -> np.ndarray:
    """入力データの検証とnumpy配列への変換（pandas.Series対応版）"""
    # pandas.Seriesの場合はnumpy配列に変換
    if isinstance(data, pd.Series):
        data = data.to_numpy()
    elif not isinstance(data, np.ndarray):
        data = np.array(data, dtype=float)

    if len(data) < min_length:
        raise ValueError(f"データ長が不足: 必要{min_length}, 実際{len(data)}")

    return data


def _validate_dual_input(data0: Union[np.ndarray, pd.Series], data1: Union[np.ndarray, pd.Series]) -> None:
    """2つの入力データの長さ一致確認"""
    if len(data0) != len(data1):
        raise ValueError(
            f"データの長さが一致しません。Data0: {len(data0)}, Data1: {len(data1)}"
        )


class StatisticsIndicators:
    """
    統計系指標クラス（NumPy標準関数版）

    全ての指標はNumPy標準関数を使用し、TA-libへの依存を排除しています。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    def beta(high: np.ndarray, low: np.ndarray, period: int = 5) -> np.ndarray:
        """Beta (ベータ) - 簡易実装"""
        high = _validate_input(high, period)
        low = _validate_input(low, period)
        _validate_dual_input(high, low)

        result = np.full_like(high, np.nan)

        for i in range(period - 1, len(high)):
            high_window = high[i - period + 1 : i + 1]
            low_window = low[i - period + 1 : i + 1]

            # 簡易ベータ計算（高値と安値の相関）
            if np.std(low_window) > 0:
                result[i] = np.corrcoef(high_window, low_window)[0, 1]
            else:
                result[i] = 0.0

        return result

    @staticmethod
    def correl(high: np.ndarray, low: np.ndarray, period: int = 30) -> np.ndarray:
        """Pearson's Correlation Coefficient (ピアソン相関係数)"""
        high = _validate_input(high, period)
        low = _validate_input(low, period)
        _validate_dual_input(high, low)

        result = np.full_like(high, np.nan)

        for i in range(period - 1, len(high)):
            high_window = high[i - period + 1 : i + 1]
            low_window = low[i - period + 1 : i + 1]

            if np.std(high_window) > 0 and np.std(low_window) > 0:
                result[i] = np.corrcoef(high_window, low_window)[0, 1]
            else:
                result[i] = 0.0

        return result

    @staticmethod
    def linearreg(data: Union[np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
        """Linear Regression (線形回帰)"""
        data = _validate_input(data, period)
        result = np.full_like(data, np.nan)

        for i in range(period - 1, len(data)):
            y = data[i - period + 1 : i + 1]
            x = np.arange(period)

            # 線形回帰の計算
            slope, intercept = np.polyfit(x, y, 1)
            result[i] = slope * (period - 1) + intercept

        return result

    @staticmethod
    def linearreg_angle(data: Union[np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
        """Linear Regression Angle (線形回帰角度)"""
        data = _validate_input(data, period)
        result = np.full_like(data, np.nan)

        for i in range(period - 1, len(data)):
            y = data[i - period + 1 : i + 1]
            x = np.arange(period)

            # 線形回帰の傾きを角度に変換
            slope, _ = np.polyfit(x, y, 1)
            result[i] = np.arctan(slope) * 180 / np.pi

        return result

    @staticmethod
    def linearreg_intercept(data: Union[np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
        """Linear Regression Intercept (線形回帰切片)"""
        data = _validate_input(data, period)
        result = np.full_like(data, np.nan)

        for i in range(period - 1, len(data)):
            y = data[i - period + 1 : i + 1]
            x = np.arange(period)

            # 線形回帰の切片
            _, intercept = np.polyfit(x, y, 1)
            result[i] = intercept

        return result

    @staticmethod
    def linearreg_slope(data: Union[np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
        """Linear Regression Slope (線形回帰傾き)"""
        data = _validate_input(data, period)
        result = np.full_like(data, np.nan)

        for i in range(period - 1, len(data)):
            y = data[i - period + 1 : i + 1]
            x = np.arange(period)

            # 線形回帰の傾き
            slope, _ = np.polyfit(x, y, 1)
            result[i] = slope

        return result

    @staticmethod
    def stddev(data: Union[np.ndarray, pd.Series], period: int = 5, nbdev: float = 1.0) -> np.ndarray:
        """Standard Deviation (標準偏差)"""
        data = _validate_input(data, period)
        result = np.full_like(data, np.nan)

        for i in range(period - 1, len(data)):
            window = data[i - period + 1 : i + 1]
            result[i] = np.std(window, ddof=1) * nbdev

        return result

    @staticmethod
    def tsf(data: Union[np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
        """Time Series Forecast (時系列予測)"""
        data = _validate_input(data, period)
        result = np.full_like(data, np.nan)

        for i in range(period - 1, len(data)):
            y = data[i - period + 1 : i + 1]
            x = np.arange(period)

            # 線形回帰による次の値の予測
            slope, intercept = np.polyfit(x, y, 1)
            result[i] = slope * period + intercept

        return result

    @staticmethod
    def var(data: Union[np.ndarray, pd.Series], period: int = 5, nbdev: float = 1.0) -> np.ndarray:
        """Variance (分散)"""
        data = _validate_input(data, period)
        result = np.full_like(data, np.nan)

        for i in range(period - 1, len(data)):
            window = data[i - period + 1 : i + 1]
            result[i] = np.var(window, ddof=1) * nbdev

        return result

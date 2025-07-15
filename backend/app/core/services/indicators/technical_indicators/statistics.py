"""
統計系テクニカル指標（オートストラテジー最適化版）

このモジュールはnumpy配列ベースでTa-libを直接使用し、
backtesting.pyとの完全な互換性を提供します。
pandas Seriesの変換は一切行いません。
"""

import talib
import numpy as np
from typing import cast
from ..utils import (
    validate_input,
    validate_multi_input,
    handle_talib_errors,
    log_indicator_calculation,
    format_indicator_result,
    ensure_numpy_array,
)


class StatisticsIndicators:
    """
    統計系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    @handle_talib_errors
    def beta(high: np.ndarray, low: np.ndarray, period: int = 5) -> np.ndarray:
        """
        Beta (ベータ)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            period: 期間（デフォルト: 5）

        Returns:
            BETA値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        validate_multi_input(high, low, high, period)
        log_indicator_calculation("BETA", {"period": period}, len(high))

        result = talib.BETA(high, low, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "BETA"))

    @staticmethod
    @handle_talib_errors
    def correl(high: np.ndarray, low: np.ndarray, period: int = 30) -> np.ndarray:
        """
        Pearson's Correlation Coefficient (ピアソン相関係数)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            period: 期間（デフォルト: 30）

        Returns:
            CORREL値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        validate_multi_input(high, low, high, period)
        log_indicator_calculation("CORREL", {"period": period}, len(high))

        result = talib.CORREL(high, low, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "CORREL"))

    @staticmethod
    @handle_talib_errors
    def linearreg(data: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Linear Regression (線形回帰)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            LINEARREG値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("LINEARREG", {"period": period}, len(data))

        result = talib.LINEARREG(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "LINEARREG"))

    @staticmethod
    @handle_talib_errors
    def linearreg_angle(data: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Linear Regression Angle (線形回帰角度)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            LINEARREG_ANGLE値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("LINEARREG_ANGLE", {"period": period}, len(data))

        result = talib.LINEARREG_ANGLE(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "LINEARREG_ANGLE"))

    @staticmethod
    @handle_talib_errors
    def linearreg_intercept(data: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Linear Regression Intercept (線形回帰切片)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            LINEARREG_INTERCEPT値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("LINEARREG_INTERCEPT", {"period": period}, len(data))

        result = talib.LINEARREG_INTERCEPT(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "LINEARREG_INTERCEPT"))

    @staticmethod
    @handle_talib_errors
    def linearreg_slope(data: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Linear Regression Slope (線形回帰傾き)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            LINEARREG_SLOPE値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("LINEARREG_SLOPE", {"period": period}, len(data))

        result = talib.LINEARREG_SLOPE(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "LINEARREG_SLOPE"))

    @staticmethod
    @handle_talib_errors
    def stddev(data: np.ndarray, period: int = 5, nbdev: float = 1.0) -> np.ndarray:
        """
        Standard Deviation (標準偏差)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 5）
            nbdev: 偏差数（デフォルト: 1.0）

        Returns:
            STDDEV値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation(
            "STDDEV", {"period": period, "nbdev": nbdev}, len(data)
        )

        result = talib.STDDEV(data, timeperiod=period, nbdev=nbdev)
        return cast(np.ndarray, format_indicator_result(result, "STDDEV"))

    @staticmethod
    @handle_talib_errors
    def tsf(data: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Time Series Forecast (時系列予測)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            TSF値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("TSF", {"period": period}, len(data))

        result = talib.TSF(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "TSF"))

    @staticmethod
    @handle_talib_errors
    def var(data: np.ndarray, period: int = 5, nbdev: float = 1.0) -> np.ndarray:
        """
        Variance (分散)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 5）
            nbdev: 偏差数（デフォルト: 1.0）

        Returns:
            VAR値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("VAR", {"period": period, "nbdev": nbdev}, len(data))

        result = talib.VAR(data, timeperiod=period, nbdev=nbdev)
        return cast(np.ndarray, format_indicator_result(result, "VAR"))

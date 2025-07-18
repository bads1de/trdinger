"""
ボラティリティ系テクニカル指標

このモジュールはnumpy配列ベースでTa-libを直接使用し、
backtesting.pyとの完全な互換性を提供します。
pandas Seriesの変換は一切行いません。
"""

import talib
import numpy as np
from typing import Tuple, cast
from ..utils import (
    validate_input,
    validate_multi_input,
    handle_talib_errors,
    log_indicator_calculation,
    format_indicator_result,
    ensure_numpy_array,
)


class VolatilityIndicators:
    """
    ボラティリティ系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    @handle_talib_errors
    def atr(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Average True Range (平均真の値幅)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            ATR値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        log_indicator_calculation("ATR", {"period": period}, len(close))

        result = talib.ATR(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ATR"))

    @staticmethod
    @handle_talib_errors
    def natr(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Normalized Average True Range (正規化平均真の値幅)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            NATR値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        log_indicator_calculation("NATR", {"period": period}, len(close))

        result = talib.NATR(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "NATR"))

    @staticmethod
    @handle_talib_errors
    def trange(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        True Range (真の値幅)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            TRANGE値のnumpy配列
        """
        validate_multi_input(high, low, close, 1)
        log_indicator_calculation("TRANGE", {}, len(close))

        result = talib.TRANGE(high, low, close)
        return cast(np.ndarray, format_indicator_result(result, "TRANGE"))

    @staticmethod
    @handle_talib_errors
    def bollinger_bands(
        data: np.ndarray, period: int = 20, std_dev: float = 2.0, matype: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bollinger Bands (ボリンジャーバンド)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 20）
            std_dev: 標準偏差倍率（デフォルト: 2.0）
            matype: 移動平均種別（デフォルト: 0=SMA）

        Returns:
            (Upper Band, Middle Band, Lower Band)のtuple
        """
        validate_input(data, period)

        params = {
            "timeperiod": period,
            "nbdevup": std_dev,
            "nbdevdn": std_dev,
            "matype": matype,
        }
        log_indicator_calculation("BBANDS", params, len(data))

        upper, middle, lower = talib.BBANDS(data, **params)  # type: ignore
        return cast(
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            format_indicator_result((upper, middle, lower), "BBANDS"),
        )

    @staticmethod
    @handle_talib_errors
    def stddev(data: np.ndarray, period: int = 5, nbdev: float = 1.0) -> np.ndarray:
        """
        Standard Deviation (標準偏差)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 5）
            nbdev: 標準偏差倍率（デフォルト: 1.0）

        Returns:
            STDDEV値のnumpy配列
        """
        validate_input(data, period)
        log_indicator_calculation(
            "STDDEV", {"period": period, "nbdev": nbdev}, len(data)
        )

        result = talib.STDDEV(data, timeperiod=period, nbdev=nbdev)
        return cast(np.ndarray, format_indicator_result(result, "STDDEV"))

    @staticmethod
    @handle_talib_errors
    def var(data: np.ndarray, period: int = 5, nbdev: float = 1.0) -> np.ndarray:
        """
        Variance (分散)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 5）
            nbdev: 標準偏差倍率（デフォルト: 1.0）

        Returns:
            VAR値のnumpy配列
        """
        validate_input(data, period)
        log_indicator_calculation("VAR", {"period": period, "nbdev": nbdev}, len(data))

        result = talib.VAR(data, timeperiod=period, nbdev=nbdev)
        return cast(np.ndarray, format_indicator_result(result, "VAR"))

    @staticmethod
    @handle_talib_errors
    def adx(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Average Directional Movement Index (平均方向性指数)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            ADX値のnumpy配列
        """
        validate_multi_input(high, low, close, period)
        log_indicator_calculation("ADX", {"period": period}, len(close))

        result = talib.ADX(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ADX"))

    @staticmethod
    @handle_talib_errors
    def adxr(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Average Directional Movement Index Rating (ADX評価)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            ADXR値のnumpy配列
        """
        validate_multi_input(high, low, close, period)
        log_indicator_calculation("ADXR", {"period": period}, len(close))

        result = talib.ADXR(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ADXR"))

    @staticmethod
    @handle_talib_errors
    def dx(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Directional Movement Index (方向性指数)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            DX値のnumpy配列
        """
        validate_multi_input(high, low, close, period)
        log_indicator_calculation("DX", {"period": period}, len(close))

        result = talib.DX(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "DX"))

    @staticmethod
    @handle_talib_errors
    def minus_di(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Minus Directional Indicator (マイナス方向性指標)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            MINUS_DI値のnumpy配列
        """
        validate_multi_input(high, low, close, period)
        log_indicator_calculation("MINUS_DI", {"period": period}, len(close))

        result = talib.MINUS_DI(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "MINUS_DI"))

    @staticmethod
    @handle_talib_errors
    def plus_di(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Plus Directional Indicator (プラス方向性指標)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            PLUS_DI値のnumpy配列
        """
        validate_multi_input(high, low, close, period)
        log_indicator_calculation("PLUS_DI", {"period": period}, len(close))

        result = talib.PLUS_DI(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "PLUS_DI"))

    @staticmethod
    @handle_talib_errors
    def minus_dm(high: np.ndarray, low: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Minus Directional Movement (マイナス方向性移動)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            MINUS_DM値のnumpy配列
        """
        validate_multi_input(high, low, high, period)  # closeの代わりにhighを使用
        log_indicator_calculation("MINUS_DM", {"period": period}, len(high))

        result = talib.MINUS_DM(high, low, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "MINUS_DM"))

    @staticmethod
    @handle_talib_errors
    def plus_dm(high: np.ndarray, low: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Plus Directional Movement (プラス方向性移動)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            PLUS_DM値のnumpy配列
        """
        validate_multi_input(high, low, high, period)  # closeの代わりにhighを使用
        log_indicator_calculation("PLUS_DM", {"period": period}, len(high))

        result = talib.PLUS_DM(high, low, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "PLUS_DM"))

    @staticmethod
    @handle_talib_errors
    def aroon(
        high: np.ndarray, low: np.ndarray, period: int = 14
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aroon (アルーン)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            (Aroon Down, Aroon Up)のtuple
        """
        validate_multi_input(high, low, high, period)  # closeの代わりにhighを使用
        log_indicator_calculation("AROON", {"period": period}, len(high))

        aroondown, aroonup = talib.AROON(high, low, timeperiod=period)
        return cast(
            Tuple[np.ndarray, np.ndarray],
            format_indicator_result((aroondown, aroonup), "AROON"),
        )

    @staticmethod
    @handle_talib_errors
    def aroonosc(high: np.ndarray, low: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Aroon Oscillator (アルーンオシレーター)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            AROONOSC値のnumpy配列
        """
        validate_multi_input(high, low, high, period)  # closeの代わりにhighを使用
        log_indicator_calculation("AROONOSC", {"period": period}, len(high))

        result = talib.AROONOSC(high, low, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "AROONOSC"))

"""
トレンド系テクニカル指標（オートストラテジー最適化版）

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
    TALibError,
)


class TrendIndicators:
    """
    トレンド系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    @handle_talib_errors
    def sma(data: np.ndarray, period: int) -> np.ndarray:
        """
        Simple Moving Average (単純移動平均)

        Args:
            data: 価格データ（numpy配列）
            period: 期間

        Returns:
            SMA値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("SMA", {"period": period}, len(data))

        result = talib.SMA(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "SMA"))

    @staticmethod
    @handle_talib_errors
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        """
        Exponential Moving Average (指数移動平均)

        Args:
            data: 価格データ（numpy配列）
            period: 期間

        Returns:
            EMA値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("EMA", {"period": period}, len(data))

        result = talib.EMA(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "EMA"))

    @staticmethod
    @handle_talib_errors
    def tema(data: np.ndarray, period: int) -> np.ndarray:
        """
        Triple Exponential Moving Average (三重指数移動平均)

        Args:
            data: 価格データ（numpy配列）
            period: 期間

        Returns:
            TEMA値のnumpy配列
        """
        validate_input(data, period)
        log_indicator_calculation("TEMA", {"period": period}, len(data))

        result = talib.TEMA(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "TEMA"))

    @staticmethod
    @handle_talib_errors
    def dema(data: np.ndarray, period: int) -> np.ndarray:
        """
        Double Exponential Moving Average (二重指数移動平均)

        Args:
            data: 価格データ（numpy配列）
            period: 期間

        Returns:
            DEMA値のnumpy配列
        """
        validate_input(data, period)
        log_indicator_calculation("DEMA", {"period": period}, len(data))

        result = talib.DEMA(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "DEMA"))

    @staticmethod
    @handle_talib_errors
    def wma(data: np.ndarray, period: int) -> np.ndarray:
        """
        Weighted Moving Average (加重移動平均)

        Args:
            data: 価格データ（numpy配列）
            period: 期間

        Returns:
            WMA値のnumpy配列
        """
        validate_input(data, period)
        log_indicator_calculation("WMA", {"period": period}, len(data))

        result = talib.WMA(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "WMA"))

    @staticmethod
    @handle_talib_errors
    def trima(data: np.ndarray, period: int) -> np.ndarray:
        """
        Triangular Moving Average (三角移動平均)

        Args:
            data: 価格データ（numpy配列）
            period: 期間

        Returns:
            TRIMA値のnumpy配列
        """
        validate_input(data, period)
        log_indicator_calculation("TRIMA", {"period": period}, len(data))

        result = talib.TRIMA(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "TRIMA"))

    @staticmethod
    @handle_talib_errors
    def kama(data: np.ndarray, period: int = 30) -> np.ndarray:
        """
        Kaufman Adaptive Moving Average (カウフマン適応移動平均)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 30）

        Returns:
            KAMA値のnumpy配列
        """
        validate_input(data, period)
        log_indicator_calculation("KAMA", {"period": period}, len(data))

        result = talib.KAMA(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "KAMA"))

    @staticmethod
    @handle_talib_errors
    def mama(
        data: np.ndarray, fastlimit: float = 0.5, slowlimit: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        MESA Adaptive Moving Average (MESA適応移動平均)

        Args:
            data: 価格データ（numpy配列）
            fastlimit: 高速制限（デフォルト: 0.5）
            slowlimit: 低速制限（デフォルト: 0.05）

        Returns:
            (MAMA, FAMA)のtuple
        """
        validate_input(data, 2)  # 最小期間は2
        log_indicator_calculation(
            "MAMA", {"fastlimit": fastlimit, "slowlimit": slowlimit}, len(data)
        )

        mama, fama = talib.MAMA(data, fastlimit=fastlimit, slowlimit=slowlimit)
        return cast(
            Tuple[np.ndarray, np.ndarray],
            format_indicator_result((mama, fama), "MAMA"),
        )

    @staticmethod
    @handle_talib_errors
    def t3(data: np.ndarray, period: int = 5, vfactor: float = 0.7) -> np.ndarray:
        """
        Triple Exponential Moving Average (T3)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 5）
            vfactor: ボリューム因子（デフォルト: 0.7）

        Returns:
            T3値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation(
            "T3", {"period": period, "vfactor": vfactor}, len(data)
        )

        result = talib.T3(data, timeperiod=period, vfactor=vfactor)
        return cast(np.ndarray, format_indicator_result(result, "T3"))

    @staticmethod
    @handle_talib_errors
    def wma(data: np.ndarray, period: int) -> np.ndarray:
        """
        Weighted Moving Average (加重移動平均)

        Args:
            data: 価格データ（numpy配列）
            period: 期間

        Returns:
            WMA値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("WMA", {"period": period}, len(data))

        result = talib.WMA(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "WMA"))

    @staticmethod
    @handle_talib_errors
    def trima(data: np.ndarray, period: int) -> np.ndarray:
        """
        Triangular Moving Average (三角移動平均)

        Args:
            data: 価格データ（numpy配列）
            period: 期間

        Returns:
            TRIMA値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("TRIMA", {"period": period}, len(data))

        result = talib.TRIMA(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "TRIMA"))

    @staticmethod
    @handle_talib_errors
    def kama(data: np.ndarray, period: int = 30) -> np.ndarray:
        """
        Kaufman Adaptive Moving Average (カウフマン適応移動平均)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 30）

        Returns:
            KAMA値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("KAMA", {"period": period}, len(data))

        result = talib.KAMA(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "KAMA"))

    @staticmethod
    @handle_talib_errors
    def mama(data: np.ndarray, fastlimit: float = 0.5, slowlimit: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        MESA Adaptive Moving Average (MESA適応移動平均)

        Args:
            data: 価格データ（numpy配列）
            fastlimit: 高速制限（デフォルト: 0.5）
            slowlimit: 低速制限（デフォルト: 0.05）

        Returns:
            (MAMA, FAMA)のtuple
        """
        data = ensure_numpy_array(data)
        validate_input(data, 2)  # 最小期間は2
        log_indicator_calculation(
            "MAMA", {"fastlimit": fastlimit, "slowlimit": slowlimit}, len(data)
        )

        mama, fama = talib.MAMA(data, fastlimit=fastlimit, slowlimit=slowlimit)
        return cast(
            Tuple[np.ndarray, np.ndarray],
            format_indicator_result((mama, fama), "MAMA"),
        )

    @staticmethod
    @handle_talib_errors
    def sar(
        high: np.ndarray,
        low: np.ndarray,
        acceleration: float = 0.02,
        maximum: float = 0.2
    ) -> np.ndarray:
        """
        Parabolic SAR (パラボリックSAR)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            acceleration: 加速因子（デフォルト: 0.02）
            maximum: 最大値（デフォルト: 0.2）

        Returns:
            SAR値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        validate_multi_input(high, low, high, 2)  # closeの代わりにhighを使用
        log_indicator_calculation(
            "SAR", {"acceleration": acceleration, "maximum": maximum}, len(high)
        )

        result = talib.SAR(high, low, acceleration=acceleration, maximum=maximum)
        return cast(np.ndarray, format_indicator_result(result, "SAR"))

    @staticmethod
    @handle_talib_errors
    def sarext(
        high: np.ndarray,
        low: np.ndarray,
        startvalue: float = 0.0,
        offsetonreverse: float = 0.0,
        accelerationinitlong: float = 0.02,
        accelerationlong: float = 0.02,
        accelerationmaxlong: float = 0.2,
        accelerationinitshort: float = 0.02,
        accelerationshort: float = 0.02,
        accelerationmaxshort: float = 0.2,
    ) -> np.ndarray:
        """
        Parabolic SAR - Extended (拡張パラボリックSAR)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            startvalue: 開始値（デフォルト: 0.0）
            offsetonreverse: リバース時のオフセット（デフォルト: 0.0）
            accelerationinitlong: ロング初期加速（デフォルト: 0.02）
            accelerationlong: ロング加速（デフォルト: 0.02）
            accelerationmaxlong: ロング最大加速（デフォルト: 0.2）
            accelerationinitshort: ショート初期加速（デフォルト: 0.02）
            accelerationshort: ショート加速（デフォルト: 0.02）
            accelerationmaxshort: ショート最大加速（デフォルト: 0.2）

        Returns:
            SAREXT値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        validate_multi_input(high, low, high, 2)

        params = {
            "startvalue": startvalue,
            "offsetonreverse": offsetonreverse,
            "accelerationinitlong": accelerationinitlong,
            "accelerationlong": accelerationlong,
            "accelerationmaxlong": accelerationmaxlong,
            "accelerationinitshort": accelerationinitshort,
            "accelerationshort": accelerationshort,
            "accelerationmaxshort": accelerationmaxshort,
        }
        log_indicator_calculation("SAREXT", params, len(high))

        result = talib.SAREXT(high, low, **params)
        return cast(np.ndarray, format_indicator_result(result, "SAREXT"))

    @staticmethod
    @handle_talib_errors
    def ht_trendline(data: np.ndarray) -> np.ndarray:
        """
        Hilbert Transform - Instantaneous Trendline (ヒルベルト変換瞬時トレンドライン)

        Args:
            data: 価格データ（numpy配列）

        Returns:
            HT_TRENDLINE値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, 2)
        log_indicator_calculation("HT_TRENDLINE", {}, len(data))

        result = talib.HT_TRENDLINE(data)
        return cast(np.ndarray, format_indicator_result(result, "HT_TRENDLINE"))

    @staticmethod
    @handle_talib_errors
    def ma(data: np.ndarray, period: int, matype: int = 0) -> np.ndarray:
        """
        Moving Average (移動平均 - タイプ指定可能)

        Args:
            data: 価格データ（numpy配列）
            period: 期間
            matype: 移動平均タイプ（0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3）

        Returns:
            MA値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("MA", {"period": period, "matype": matype}, len(data))

        result = talib.MA(data, timeperiod=period, matype=matype)
        return cast(np.ndarray, format_indicator_result(result, "MA"))

    @staticmethod
    @handle_talib_errors
    def mavp(data: np.ndarray, periods: np.ndarray, minperiod: int = 2, maxperiod: int = 30, matype: int = 0) -> np.ndarray:
        """
        Moving Average with Variable Period (可変期間移動平均)

        Args:
            data: 価格データ（numpy配列）
            periods: 期間データ（numpy配列）
            minperiod: 最小期間（デフォルト: 2）
            maxperiod: 最大期間（デフォルト: 30）
            matype: 移動平均タイプ（デフォルト: 0=SMA）

        Returns:
            MAVP値のnumpy配列
        """
        data = ensure_numpy_array(data)
        periods = ensure_numpy_array(periods)

        if len(data) != len(periods):
            raise TALibError(f"データと期間の長さが一致しません。Data: {len(data)}, Periods: {len(periods)}")

        validate_input(data, minperiod)
        log_indicator_calculation(
            "MAVP", {"minperiod": minperiod, "maxperiod": maxperiod, "matype": matype}, len(data)
        )

        result = talib.MAVP(data, periods, minperiod=minperiod, maxperiod=maxperiod, matype=matype)
        return cast(np.ndarray, format_indicator_result(result, "MAVP"))

    @staticmethod
    @handle_talib_errors
    def midpoint(data: np.ndarray, period: int) -> np.ndarray:
        """
        MidPoint over period (期間中点)

        Args:
            data: 価格データ（numpy配列）
            period: 期間

        Returns:
            MIDPOINT値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("MIDPOINT", {"period": period}, len(data))

        result = talib.MIDPOINT(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "MIDPOINT"))

    @staticmethod
    @handle_talib_errors
    def midprice(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
        """
        Midpoint Price over period (期間中値価格)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            period: 期間

        Returns:
            MIDPRICE値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        validate_multi_input(high, low, high, period)
        log_indicator_calculation("MIDPRICE", {"period": period}, len(high))

        result = talib.MIDPRICE(high, low, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "MIDPRICE"))



    @staticmethod
    @handle_talib_errors
    def sar(
        high: np.ndarray,
        low: np.ndarray,
        acceleration: float = 0.02,
        maximum: float = 0.2,
    ) -> np.ndarray:
        """
        Parabolic SAR (パラボリックSAR)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            acceleration: 加速因子（デフォルト: 0.02）
            maximum: 最大値（デフォルト: 0.2）

        Returns:
            SAR値のnumpy配列
        """
        validate_multi_input(high, low, high, 2)
        log_indicator_calculation(
            "SAR", {"acceleration": acceleration, "maximum": maximum}, len(high)
        )

        result = talib.SAR(high, low, acceleration=acceleration, maximum=maximum)
        return cast(np.ndarray, format_indicator_result(result, "SAR"))

    @staticmethod
    @handle_talib_errors
    def sarext(
        high: np.ndarray,
        low: np.ndarray,
        startvalue: float = 0,
        offsetonreverse: float = 0,
        accelerationinitlong: float = 0.02,
        accelerationlong: float = 0.02,
        accelerationmaxlong: float = 0.2,
        accelerationinitshort: float = 0.02,
        accelerationshort: float = 0.02,
        accelerationmaxshort: float = 0.2,
    ) -> np.ndarray:
        """
        Parabolic SAR - Extended (拡張パラボリックSAR)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            その他: SAR拡張パラメータ

        Returns:
            SAREXT値のnumpy配列
        """
        validate_multi_input(high, low, high, 2)

        params = {
            "startvalue": startvalue,
            "offsetonreverse": offsetonreverse,
            "accelerationinitlong": accelerationinitlong,
            "accelerationlong": accelerationlong,
            "accelerationmaxlong": accelerationmaxlong,
            "accelerationinitshort": accelerationinitshort,
            "accelerationshort": accelerationshort,
            "accelerationmaxshort": accelerationmaxshort,
        }
        log_indicator_calculation("SAREXT", params, len(high))

        result = talib.SAREXT(high, low, **params)
        return cast(np.ndarray, format_indicator_result(result, "SAREXT"))

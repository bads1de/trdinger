"""
パターン認識系テクニカル指標

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
    log_indicator_calculation,
    format_indicator_result,
    ensure_numpy_array,
    TALibError,
)


class PatternRecognitionIndicators:
    """
    パターン認識系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    @handle_talib_errors
    def cdl_doji(
        open_data: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Doji (同事)

        Args:
            open_data: 始値データ（numpy配列）
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            CDL_DOJI値のnumpy配列
        """
        open_data = ensure_numpy_array(open_data)
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)

        if not (len(open_data) == len(high) == len(low) == len(close)):
            raise TALibError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 1)
        log_indicator_calculation("CDL_DOJI", {}, len(close))

        result = talib.CDLDOJI(open_data, high, low, close)
        return cast(np.ndarray, format_indicator_result(result, "CDL_DOJI"))

    @staticmethod
    @handle_talib_errors
    def cdl_hammer(
        open_data: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Hammer (ハンマー)

        Args:
            open_data: 始値データ（numpy配列）
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            CDL_HAMMER値のnumpy配列
        """
        open_data = ensure_numpy_array(open_data)
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)

        if not (len(open_data) == len(high) == len(low) == len(close)):
            raise TALibError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 1)
        log_indicator_calculation("CDL_HAMMER", {}, len(close))

        result = talib.CDLHAMMER(open_data, high, low, close)
        return cast(np.ndarray, format_indicator_result(result, "CDL_HAMMER"))

    @staticmethod
    @handle_talib_errors
    def cdl_hanging_man(
        open_data: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Hanging Man (首吊り線)

        Args:
            open_data: 始値データ（numpy配列）
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            CDL_HANGING_MAN値のnumpy配列
        """
        open_data = ensure_numpy_array(open_data)
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)

        if not (len(open_data) == len(high) == len(low) == len(close)):
            raise TALibError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 1)
        log_indicator_calculation("CDL_HANGING_MAN", {}, len(close))

        result = talib.CDLHANGINGMAN(open_data, high, low, close)
        return cast(np.ndarray, format_indicator_result(result, "CDL_HANGING_MAN"))

    @staticmethod
    @handle_talib_errors
    def cdl_shooting_star(
        open_data: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Shooting Star (流れ星)

        Args:
            open_data: 始値データ（numpy配列）
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            CDL_SHOOTING_STAR値のnumpy配列
        """
        open_data = ensure_numpy_array(open_data)
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)

        if not (len(open_data) == len(high) == len(low) == len(close)):
            raise TALibError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 1)
        log_indicator_calculation("CDL_SHOOTING_STAR", {}, len(close))

        result = talib.CDLSHOOTINGSTAR(open_data, high, low, close)
        return cast(np.ndarray, format_indicator_result(result, "CDL_SHOOTING_STAR"))

    @staticmethod
    @handle_talib_errors
    def cdl_engulfing(
        open_data: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Engulfing Pattern (包み線)

        Args:
            open_data: 始値データ（numpy配列）
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            CDL_ENGULFING値のnumpy配列
        """
        open_data = ensure_numpy_array(open_data)
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)

        if not (len(open_data) == len(high) == len(low) == len(close)):
            raise TALibError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 2)  # 包み線は2本のローソク足が必要
        log_indicator_calculation("CDL_ENGULFING", {}, len(close))

        result = talib.CDLENGULFING(open_data, high, low, close)
        return cast(np.ndarray, format_indicator_result(result, "CDL_ENGULFING"))

    @staticmethod
    @handle_talib_errors
    def cdl_harami(
        open_data: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Harami Pattern (はらみ線)

        Args:
            open_data: 始値データ（numpy配列）
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            CDL_HARAMI値のnumpy配列
        """
        open_data = ensure_numpy_array(open_data)
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)

        if not (len(open_data) == len(high) == len(low) == len(close)):
            raise TALibError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 2)  # はらみ線は2本のローソク足が必要
        log_indicator_calculation("CDL_HARAMI", {}, len(close))

        result = talib.CDLHARAMI(open_data, high, low, close)
        return cast(np.ndarray, format_indicator_result(result, "CDL_HARAMI"))

    @staticmethod
    @handle_talib_errors
    def cdl_piercing(
        open_data: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Piercing Pattern (切り込み線)

        Args:
            open_data: 始値データ（numpy配列）
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            CDL_PIERCING値のnumpy配列
        """
        open_data = ensure_numpy_array(open_data)
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)

        if not (len(open_data) == len(high) == len(low) == len(close)):
            raise TALibError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 2)
        log_indicator_calculation("CDL_PIERCING", {}, len(close))

        result = talib.CDLPIERCING(open_data, high, low, close)
        return cast(np.ndarray, format_indicator_result(result, "CDL_PIERCING"))

    @staticmethod
    @handle_talib_errors
    def cdl_dark_cloud_cover(
        open_data: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        penetration: float = 0.5,
    ) -> np.ndarray:
        """
        Dark Cloud Cover (かぶせ線)

        Args:
            open_data: 始値データ（numpy配列）
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            penetration: 浸透率（デフォルト: 0.5）

        Returns:
            CDL_DARK_CLOUD_COVER値のnumpy配列
        """
        open_data = ensure_numpy_array(open_data)
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)

        if not (len(open_data) == len(high) == len(low) == len(close)):
            raise TALibError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 2)
        log_indicator_calculation(
            "CDL_DARK_CLOUD_COVER", {"penetration": penetration}, len(close)
        )

        result = talib.CDLDARKCLOUDCOVER(
            open_data, high, low, close, penetration=penetration
        )
        return cast(np.ndarray, format_indicator_result(result, "CDL_DARK_CLOUD_COVER"))

    @staticmethod
    @handle_talib_errors
    def cdl_morning_star(
        open_data: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        penetration: float = 0.3,
    ) -> np.ndarray:
        """
        Morning Star (明けの明星)

        Args:
            open_data: 始値データ（numpy配列）
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            penetration: 浸透率（デフォルト: 0.3）

        Returns:
            CDL_MORNING_STAR値のnumpy配列
        """
        open_data = ensure_numpy_array(open_data)
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)

        if not (len(open_data) == len(high) == len(low) == len(close)):
            raise TALibError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 3)  # 明けの明星は3本のローソク足が必要
        log_indicator_calculation(
            "CDL_MORNING_STAR", {"penetration": penetration}, len(close)
        )

        result = talib.CDLMORNINGSTAR(
            open_data, high, low, close, penetration=penetration
        )
        return cast(np.ndarray, format_indicator_result(result, "CDL_MORNING_STAR"))

    @staticmethod
    @handle_talib_errors
    def cdl_evening_star(
        open_data: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        penetration: float = 0.3,
    ) -> np.ndarray:
        """
        Evening Star (宵の明星)

        Args:
            open_data: 始値データ（numpy配列）
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            penetration: 浸透率（デフォルト: 0.3）

        Returns:
            CDL_EVENING_STAR値のnumpy配列
        """
        open_data = ensure_numpy_array(open_data)
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)

        if not (len(open_data) == len(high) == len(low) == len(close)):
            raise TALibError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 3)  # 宵の明星は3本のローソク足が必要
        log_indicator_calculation(
            "CDL_EVENING_STAR", {"penetration": penetration}, len(close)
        )

        result = talib.CDLEVENINGSTAR(
            open_data, high, low, close, penetration=penetration
        )
        return cast(np.ndarray, format_indicator_result(result, "CDL_EVENING_STAR"))

    @staticmethod
    @handle_talib_errors
    def cdl_three_black_crows(
        open_data: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Three Black Crows (三羽烏)

        Args:
            open_data: 始値データ（numpy配列）
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            CDL_THREE_BLACK_CROWS値のnumpy配列
        """
        open_data = ensure_numpy_array(open_data)
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)

        if not (len(open_data) == len(high) == len(low) == len(close)):
            raise TALibError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 3)  # 三羽烏は3本のローソク足が必要
        log_indicator_calculation("CDL_THREE_BLACK_CROWS", {}, len(close))

        result = talib.CDL3BLACKCROWS(open_data, high, low, close)
        return cast(
            np.ndarray, format_indicator_result(result, "CDL_THREE_BLACK_CROWS")
        )

    @staticmethod
    @handle_talib_errors
    def cdl_three_white_soldiers(
        open_data: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Three Advancing White Soldiers (三兵)

        Args:
            open_data: 始値データ（numpy配列）
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            CDL_THREE_WHITE_SOLDIERS値のnumpy配列
        """
        open_data = ensure_numpy_array(open_data)
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)

        if not (len(open_data) == len(high) == len(low) == len(close)):
            raise TALibError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 3)  # 三兵は3本のローソク足が必要
        log_indicator_calculation("CDL_THREE_WHITE_SOLDIERS", {}, len(close))

        result = talib.CDL3WHITESOLDIERS(open_data, high, low, close)
        return cast(
            np.ndarray, format_indicator_result(result, "CDL_THREE_WHITE_SOLDIERS")
        )

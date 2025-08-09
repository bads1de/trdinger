"""
パターン認識系テクニカル指標

このモジュールはnumpy配列ベースでTa-libを直接使用し、
backtesting.pyとの完全な互換性を提供します。
pandas Seriesの変換は一切行いません。
"""

import numpy as np

from ..utils import (
    PandasTAError,
    ensure_numpy_array,
    validate_input,
)
from ..pandas_ta_utils import (
    cdl_doji as pandas_ta_cdl_doji,
    cdl_hammer as pandas_ta_cdl_hammer,
    cdl_hanging_man as pandas_ta_cdl_hanging_man,
    cdl_shooting_star as pandas_ta_cdl_shooting_star,
    cdl_engulfing as pandas_ta_cdl_engulfing,
    cdl_harami as pandas_ta_cdl_harami,
    cdl_piercing as pandas_ta_cdl_piercing,
    cdl_dark_cloud_cover as pandas_ta_cdl_dark_cloud_cover,
    cdl_morning_star as pandas_ta_cdl_morning_star,
    cdl_evening_star as pandas_ta_cdl_evening_star,
    cdl_three_black_crows as pandas_ta_cdl_three_black_crows,
    cdl_three_white_soldiers as pandas_ta_cdl_three_white_soldiers,
)


class PatternRecognitionIndicators:
    """
    パターン認識系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
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
            raise PandasTAError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 1)
        return pandas_ta_cdl_doji(open_data, high, low, close)

    @staticmethod
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
            raise PandasTAError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 1)
        return pandas_ta_cdl_hammer(open_data, high, low, close)

    @staticmethod
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
            raise PandasTAError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 1)
        return pandas_ta_cdl_hanging_man(open_data, high, low, close)

    @staticmethod
    def cdl_shooting_star(
        open_data: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Shooting Star (流れ星) - pandas-ta版

        Args:
            open_data: 始値データ（numpy配列）
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            CDL_SHOOTING_STAR値のnumpy配列
        """
        return pandas_ta_cdl_shooting_star(open_data, high, low, close)

    @staticmethod
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
            raise PandasTAError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 2)  # 包み線は2本のローソク足が必要
        return pandas_ta_cdl_engulfing(open_data, high, low, close)

    @staticmethod
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
            raise PandasTAError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        return pandas_ta_cdl_harami(open_data, high, low, close)

    @staticmethod
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
            raise PandasTAError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        return pandas_ta_cdl_piercing(open_data, high, low, close)

    @staticmethod
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
            raise PandasTAError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        return pandas_ta_cdl_dark_cloud_cover(open_data, high, low, close)

    @staticmethod
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
            raise PandasTAError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        return pandas_ta_cdl_morning_star(open_data, high, low, close)

    @staticmethod
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
            raise PandasTAError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        return pandas_ta_cdl_evening_star(open_data, high, low, close)

    @staticmethod
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
            raise PandasTAError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        return pandas_ta_cdl_three_black_crows(open_data, high, low, close)

    @staticmethod
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
            raise PandasTAError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        return pandas_ta_cdl_three_white_soldiers(open_data, high, low, close)

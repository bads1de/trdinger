"""
パターン認識系テクニカル指標

このモジュールはnumpy配列ベースでTa-libを直接使用し、
backtesting.pyとの完全な互換性を提供します。
pandas Seriesの変換は一切行いません。
"""

from typing import Union
import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import (
    PandasTAError,
    handle_pandas_ta_errors,
    to_pandas_series,
    validate_series_data,
)


class PatternRecognitionIndicators:
    """
    パターン認識系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_doji(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """同事"""
        open_series = to_pandas_series(open_data)
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        df = pd.DataFrame(
            {"open": open_series, "high": high_series, "low": low_series, "close": close_series}
        )
        result = df.ta.cdl_doji()
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_hammer(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """ハンマー"""
        open_series = to_pandas_series(open_data)
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        df = pd.DataFrame(
            {"open": open_series, "high": high_series, "low": low_series, "close": close_series}
        )
        result = df.ta.cdl_hammer()
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_hanging_man(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """首吊り線"""
        open_series = to_pandas_series(open_data)
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        df = pd.DataFrame(
            {"open": open_series, "high": high_series, "low": low_series, "close": close_series}
        )
        result = df.ta.cdl_hanging_man()
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_shooting_star(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """流れ星"""
        open_series = to_pandas_series(open_data)
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        df = pd.DataFrame(
            {"open": open_series, "high": high_series, "low": low_series, "close": close_series}
        )
        result = df.ta.cdl_shooting_star()
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_engulfing(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """包み線"""
        open_series = to_pandas_series(open_data)
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        df = pd.DataFrame(
            {"open": open_series, "high": high_series, "low": low_series, "close": close_series}
        )
        result = df.ta.cdl_engulfing()
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_harami(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """はらみ線"""
        open_series = to_pandas_series(open_data)
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        df = pd.DataFrame(
            {"open": open_series, "high": high_series, "low": low_series, "close": close_series}
        )
        result = df.ta.cdl_harami()
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_piercing(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """切り込み線"""
        open_series = to_pandas_series(open_data)
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        df = pd.DataFrame(
            {"open": open_series, "high": high_series, "low": low_series, "close": close_series}
        )
        result = df.ta.cdl_piercing()
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_dark_cloud_cover(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """かぶせ線"""
        open_series = to_pandas_series(open_data)
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        df = pd.DataFrame(
            {"open": open_series, "high": high_series, "low": low_series, "close": close_series}
        )
        result = df.ta.cdl_dark_cloud_cover()
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_morning_star(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """明けの明星"""
        open_series = to_pandas_series(open_data)
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        df = pd.DataFrame(
            {"open": open_series, "high": high_series, "low": low_series, "close": close_series}
        )
        result = df.ta.cdl_morning_star()
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_evening_star(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """宵の明星"""
        open_series = to_pandas_series(open_data)
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        df = pd.DataFrame(
            {"open": open_series, "high": high_series, "low": low_series, "close": close_series}
        )
        result = df.ta.cdl_evening_star()
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_three_black_crows(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """三羽烏"""
        open_series = to_pandas_series(open_data)
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        df = pd.DataFrame(
            {"open": open_series, "high": high_series, "low": low_series, "close": close_series}
        )
        result = df.ta.cdl_three_black_crows()
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_three_white_soldiers(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """三兵"""
        open_series = to_pandas_series(open_data)
        high_series = to_pandas_series(high)
        low_series = to_pandas_series(low)
        close_series = to_pandas_series(close)

        df = pd.DataFrame(
            {"open": open_series, "high": high_series, "low": low_series, "close": close_series}
        )
        result = df.ta.cdl_three_white_soldiers()
        return result.values

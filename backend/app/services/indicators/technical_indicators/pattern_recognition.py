"""
パターン認識系テクニカル指標

このモジュールはpandas-taを使用し、
backtesting.pyとの互換性を提供します。
内部で最小限のSeries変換を行います。
"""

from typing import Union
import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import (
    PandasTAError,
    handle_pandas_ta_errors,
    ensure_series_minimal_conversion,
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
        open_series = ensure_series_minimal_conversion(open_data)
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        df = pd.DataFrame(
            {
                "open": open_series,
                "high": high_series,
                "low": low_series,
                "close": close_series,
            }
        )
        # pandas-ta v0.3以降では cdl または cdl_pattern を利用
        if hasattr(df.ta, "cdl"):
            result = df.ta.cdl("doji")
            return result.values
        if hasattr(df.ta, "cdl_pattern"):
            result = df.ta.cdl_pattern(name="doji")
            return result.values
        raise PandasTAError("pandas-ta に cdl/cdl_pattern がありません")

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_hammer(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """ハンマー"""
        open_series = ensure_series_minimal_conversion(open_data)
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        df = pd.DataFrame(
            {
                "open": open_series,
                "high": high_series,
                "low": low_series,
                "close": close_series,
            }
        )
        if hasattr(df.ta, "cdl"):
            result = df.ta.cdl("hammer")
            return result.values
        if hasattr(df.ta, "cdl_pattern"):
            result = df.ta.cdl_pattern(name="hammer")
            return result.values
        raise PandasTAError("pandas-ta に cdl/cdl_pattern がありません")

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_hanging_man(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """首吊り線"""
        open_series = ensure_series_minimal_conversion(open_data)
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        df = pd.DataFrame(
            {
                "open": open_series,
                "high": high_series,
                "low": low_series,
                "close": close_series,
            }
        )
        raise PandasTAError("cdl_hanging_man はサポート対象外です")

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_shooting_star(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """流れ星"""
        raise PandasTAError("cdl_shooting_star はサポート対象外です")

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_engulfing(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """包み線"""
        open_series = ensure_series_minimal_conversion(open_data)
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        df = pd.DataFrame(
            {
                "open": open_series,
                "high": high_series,
                "low": low_series,
                "close": close_series,
            }
        )
        if hasattr(df.ta, "cdl_pattern"):
            result = df.ta.cdl_pattern(name="engulfing")
            return result.values
        raise PandasTAError("pandas-ta に cdl_pattern がありません")

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_harami(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """はらみ線"""
        open_series = ensure_series_minimal_conversion(open_data)
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        df = pd.DataFrame(
            {
                "open": open_series,
                "high": high_series,
                "low": low_series,
                "close": close_series,
            }
        )
        if hasattr(df.ta, "cdl_pattern"):
            # pandas-taでは"inside" が対応（harami相当）
            result = df.ta.cdl_pattern(name="inside")
            return result.values
        raise PandasTAError("pandas-ta に cdl_pattern がありません")

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_piercing(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """切り込み線"""
        raise PandasTAError("cdl_piercing はサポート対象外です")

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_dark_cloud_cover(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """かぶせ線"""
        raise PandasTAError("cdl_dark_cloud_cover はサポート対象外です")

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_morning_star(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """明けの明星"""
        raise PandasTAError("cdl_morning_star はサポート対象外です")

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_evening_star(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """宵の明星"""
        raise PandasTAError("cdl_evening_star はサポート対象外です")

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_three_black_crows(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """三羽烏"""
        open_series = ensure_series_minimal_conversion(open_data)
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        df = pd.DataFrame(
            {
                "open": open_series,
                "high": high_series,
                "low": low_series,
                "close": close_series,
            }
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
        open_series = ensure_series_minimal_conversion(open_data)
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        df = pd.DataFrame(
            {
                "open": open_series,
                "high": high_series,
                "low": low_series,
                "close": close_series,
            }
        )
        result = df.ta.cdl_three_white_soldiers()
        return result.values

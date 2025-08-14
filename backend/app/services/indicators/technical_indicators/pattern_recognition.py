"""
パターン認識系テクニカル指標

このモジュールはpandas-taを使用し、
backtesting.pyとの互換性を提供します。
内部で最小限のSeries変換を行います。
"""

from typing import Union
import numpy as np
import pandas as pd


from ..utils import (
    PandasTAError,
    handle_pandas_ta_errors,
    ensure_series_minimal_conversion,
)


class PatternRecognitionIndicators:
    """
    パターン認識系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    def _build_ohlc_df(
        open_series, high_series, low_series, close_series
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": open_series,
                "high": high_series,
                "low": low_series,
                "close": close_series,
            }
        )

    @staticmethod
    def _cdl_pattern_values(df: pd.DataFrame, name: str) -> np.ndarray:
        """pandas-ta の cdl_pattern を使って 1 次元 ndarray を返す共通処理"""
        if hasattr(df.ta, "cdl_pattern"):
            result = df.ta.cdl_pattern(name=name)
            # DataFrame/Series いずれにも対応して 1 次元に揃える
            if isinstance(result, pd.DataFrame):
                return result.iloc[:, 0].values
            if isinstance(result, pd.Series):
                return result.values
            return np.asarray(result).ravel()
        raise PandasTAError("pandas-ta に cdl_pattern がありません")

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

        df = PatternRecognitionIndicators._build_ohlc_df(
            open_series, high_series, low_series, close_series
        )
        # pandas-ta v0.3 互換: cdl または cdl_pattern
        if hasattr(df.ta, "cdl"):
            return df.ta.cdl("doji").values
        return PatternRecognitionIndicators._cdl_pattern_values(df, name="doji")

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

        df = PatternRecognitionIndicators._build_ohlc_df(
            open_series, high_series, low_series, close_series
        )
        if hasattr(df.ta, "cdl"):
            return df.ta.cdl("hammer").values
        return PatternRecognitionIndicators._cdl_pattern_values(df, name="hammer")

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

        df = PatternRecognitionIndicators._build_ohlc_df(
            open_series, high_series, low_series, close_series
        )
        return PatternRecognitionIndicators._cdl_pattern_values(df, name="hangingman")

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_shooting_star(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """流れ星"""
        open_series = ensure_series_minimal_conversion(open_data)
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)
        df = PatternRecognitionIndicators._build_ohlc_df(
            open_series, high_series, low_series, close_series
        )
        return PatternRecognitionIndicators._cdl_pattern_values(df, name="shootingstar")

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

        df = PatternRecognitionIndicators._build_ohlc_df(
            open_series, high_series, low_series, close_series
        )
        return PatternRecognitionIndicators._cdl_pattern_values(df, name="engulfing")

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

        df = PatternRecognitionIndicators._build_ohlc_df(
            open_series, high_series, low_series, close_series
        )
        # Harami を直接利用
        return PatternRecognitionIndicators._cdl_pattern_values(df, name="harami")

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_piercing(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """切り込み線"""
        open_series = ensure_series_minimal_conversion(open_data)
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)
        df = PatternRecognitionIndicators._build_ohlc_df(
            open_series, high_series, low_series, close_series
        )
        return PatternRecognitionIndicators._cdl_pattern_values(df, name="piercing")

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_dark_cloud_cover(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """かぶせ線"""
        open_series = ensure_series_minimal_conversion(open_data)
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)
        df = PatternRecognitionIndicators._build_ohlc_df(
            open_series, high_series, low_series, close_series
        )
        return PatternRecognitionIndicators._cdl_pattern_values(
            df, name="darkcloudcover"
        )

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_morning_star(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """明けの明星"""
        open_series = ensure_series_minimal_conversion(open_data)
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)
        df = PatternRecognitionIndicators._build_ohlc_df(
            open_series, high_series, low_series, close_series
        )
        return PatternRecognitionIndicators._cdl_pattern_values(df, name="morningstar")

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_evening_star(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """宵の明星"""
        open_series = ensure_series_minimal_conversion(open_data)
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)
        df = PatternRecognitionIndicators._build_ohlc_df(
            open_series, high_series, low_series, close_series
        )
        return PatternRecognitionIndicators._cdl_pattern_values(df, name="eveningstar")

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

        df = PatternRecognitionIndicators._build_ohlc_df(
            open_series, high_series, low_series, close_series
        )
        return PatternRecognitionIndicators._cdl_pattern_values(df, name="3blackcrows")

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

        df = PatternRecognitionIndicators._build_ohlc_df(
            open_series, high_series, low_series, close_series
        )
        return PatternRecognitionIndicators._cdl_pattern_values(
            df, name="3whitesoldiers"
        )

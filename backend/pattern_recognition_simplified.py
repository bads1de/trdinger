"""
パターン認識系テクニカル指標（簡素化版）

pandas-taを直接活用し、冗長なラッパーを削除した効率的な実装。
軽量エラーハンドリングで品質とパフォーマンスを両立。
"""

from typing import Union

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import handle_pandas_ta_errors


class PatternRecognitionIndicators:
    """
    パターン認識系指標クラス（簡素化版）

    重要なキャンドルスティックパターンには軽量エラーハンドリングを適用し、
    pandas-taを直接活用して効率的に実装。
    """

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_doji(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Doji（重要パターンのためエラーハンドリング付き）"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.cdl_doji(
            open_=open_series, high=high_series, low=low_series, close=close_series
        ).values

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_hammer(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Hammer（重要パターンのためエラーハンドリング付き）"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.cdl_hammer(
            open_=open_series, high=high_series, low=low_series, close=close_series
        ).values

    @staticmethod
    def cdl_hanging_man(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Hanging Man（軽量実装）"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.cdl_hangingman(
            open_=open_series, high=high_series, low=low_series, close=close_series
        ).values

    @staticmethod
    def cdl_shooting_star(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Shooting Star（軽量実装）"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.cdl_shootingstar(
            open_=open_series, high=high_series, low=low_series, close=close_series
        ).values

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_engulfing(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Engulfing Pattern（重要パターンのためエラーハンドリング付き）"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.cdl_engulfing(
            open_=open_series, high=high_series, low=low_series, close=close_series
        ).values

    @staticmethod
    def cdl_harami(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Harami Pattern（軽量実装）"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.cdl_harami(
            open_=open_series, high=high_series, low=low_series, close=close_series
        ).values

    @staticmethod
    def cdl_piercing(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Piercing Pattern（軽量実装）"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.cdl_piercing(
            open_=open_series, high=high_series, low=low_series, close=close_series
        ).values

    @staticmethod
    def cdl_dark_cloud_cover(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Dark Cloud Cover（軽量実装）"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.cdl_darkcloudcover(
            open_=open_series, high=high_series, low=low_series, close=close_series
        ).values

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_morning_star(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Morning Star（重要パターンのためエラーハンドリング付き）"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.cdl_morningstar(
            open_=open_series, high=high_series, low=low_series, close=close_series
        ).values

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_evening_star(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Evening Star（重要パターンのためエラーハンドリング付き）"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.cdl_eveningstar(
            open_=open_series, high=high_series, low=low_series, close=close_series
        ).values

    @staticmethod
    def cdl_three_black_crows(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Three Black Crows（軽量実装）"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.cdl_3blackcrows(
            open_=open_series, high=high_series, low=low_series, close=close_series
        ).values

    @staticmethod
    def cdl_three_white_soldiers(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Three White Soldiers（軽量実装）"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.cdl_3whitesoldiers(
            open_=open_series, high=high_series, low=low_series, close=close_series
        ).values

    # 追加の重要パターン
    @staticmethod
    def cdl_marubozu(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Marubozu（軽量実装）"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.cdl_marubozu(
            open_=open_series, high=high_series, low=low_series, close=close_series
        ).values

    @staticmethod
    def cdl_spinning_top(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Spinning Top（軽量実装）"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        return ta.cdl_spinningtop(
            open_=open_series, high=high_series, low=low_series, close=close_series
        ).values

    # 後方互換性のためのエイリアス
    @staticmethod
    def doji(*args, **kwargs):
        """Dojiのエイリアス"""
        return PatternRecognitionIndicators.cdl_doji(*args, **kwargs)

    @staticmethod
    def hammer(*args, **kwargs):
        """Hammerのエイリアス"""
        return PatternRecognitionIndicators.cdl_hammer(*args, **kwargs)

    @staticmethod
    def engulfing_pattern(*args, **kwargs):
        """Engulfing Patternのエイリアス"""
        return PatternRecognitionIndicators.cdl_engulfing(*args, **kwargs)

    @staticmethod
    def morning_star(*args, **kwargs):
        """Morning Starのエイリアス"""
        return PatternRecognitionIndicators.cdl_morning_star(*args, **kwargs)

    @staticmethod
    def evening_star(*args, **kwargs):
        """Evening Starのエイリアス"""
        return PatternRecognitionIndicators.cdl_evening_star(*args, **kwargs)

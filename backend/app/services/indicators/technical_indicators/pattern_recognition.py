"""
パターン認識系テクニカル指標

実装されている指標:
- cdl_doji
- cdl_hammer
- cdl_hanging_man
- cdl_shooting_star
- cdl_engulfing
- cdl_harami
- cdl_piercing
- cdl_dark_cloud_cover
- cdl_morning_star
- cdl_evening_star
- cdl_three_black_crows
- cdl_three_white_soldiers
- cdl_marubozu
- cdl_spinning_top
"""

from typing import Union

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import handle_pandas_ta_errors


class PatternRecognitionIndicators:
    """
    パターン認識系指標クラス
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

        result = ta.cdl_pattern(
            open_=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            name="doji",
        )
        return (
            result.iloc[:, 0].values
            if result is not None and not result.empty
            else np.zeros(len(open_series))
        )

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

        result = ta.cdl_pattern(
            open_=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            name="hammer",
        )
        return (
            result.iloc[:, 0].values
            if result is not None and not result.empty
            else np.zeros(len(open_series))
        )

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

        result = ta.cdl_pattern(
            open_=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            name="hangingman",
        )
        return (
            result.iloc[:, 0].values
            if result is not None and not result.empty
            else np.zeros(len(open_series))
        )

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

        result = ta.cdl_pattern(
            open_=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            name="shootingstar",
        )
        return (
            result.iloc[:, 0].values
            if result is not None and not result.empty
            else np.zeros(len(open_series))
        )

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

        result = ta.cdl_pattern(
            open_=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            name="engulfing",
        )
        return (
            result.iloc[:, 0].values
            if result is not None and not result.empty
            else np.zeros(len(open_series))
        )

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

        result = ta.cdl_pattern(
            open_=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            name="harami",
        )
        return (
            result.iloc[:, 0].values
            if result is not None and not result.empty
            else np.zeros(len(open_series))
        )

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

        result = ta.cdl_pattern(
            open_=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            name="piercing",
        )
        return (
            result.iloc[:, 0].values
            if result is not None and not result.empty
            else np.zeros(len(open_series))
        )

    @staticmethod
    def cdl_dark_cloud_cover(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Dark Cloud Cover"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        result = ta.cdl_pattern(
            open_=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            name="darkcloudcover",
        )
        return (
            result.iloc[:, 0].values
            if result is not None and not result.empty
            else np.zeros(len(open_series))
        )

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

        result = ta.cdl_pattern(
            open_=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            name="morningstar",
        )
        return (
            result.iloc[:, 0].values
            if result is not None and not result.empty
            else np.zeros(len(open_series))
        )

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

        result = ta.cdl_pattern(
            open_=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            name="eveningstar",
        )
        return (
            result.iloc[:, 0].values
            if result is not None and not result.empty
            else np.zeros(len(open_series))
        )

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

        result = ta.cdl_pattern(
            open_=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            name="3blackcrows",
        )
        return (
            result.iloc[:, 0].values
            if result is not None and not result.empty
            else np.zeros(len(open_series))
        )

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

        result = ta.cdl_pattern(
            open_=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            name="3whitesoldiers",
        )
        return (
            result.iloc[:, 0].values
            if result is not None and not result.empty
            else np.zeros(len(open_series))
        )

    # 簡易実装（pandas-taにない場合のフォールバック）
    @staticmethod
    def cdl_marubozu(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Marubozu（簡易実装）"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        # 簡易Marubozu判定（実体が全体の90%以上）
        body = np.abs(close_series - open_series)
        range_hl = high_series - low_series

        # ゼロ除算を避ける
        range_hl = np.where(range_hl == 0, 1e-10, range_hl)

        marubozu = np.where(
            body / range_hl > 0.9, np.where(close_series > open_series, 100, -100), 0
        )

        return marubozu

    @staticmethod
    def cdl_spinning_top(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Spinning Top（簡易実装）"""
        open_series = (
            pd.Series(open_data) if isinstance(open_data, np.ndarray) else open_data
        )
        high_series = pd.Series(high) if isinstance(high, np.ndarray) else high
        low_series = pd.Series(low) if isinstance(low, np.ndarray) else low
        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close

        # 簡易Spinning Top判定（小さな実体と長いヒゲ）
        body = np.abs(close_series - open_series)
        upper_shadow = high_series - np.maximum(open_series, close_series)
        lower_shadow = np.minimum(open_series, close_series) - low_series

        # 実体が小さく、上下のヒゲが長い
        spinning_top = np.where(
            (body < (upper_shadow + lower_shadow) * 0.3)
            & (upper_shadow > body * 2)
            & (lower_shadow > body * 2),
            100,
            0,
        )

        return spinning_top

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

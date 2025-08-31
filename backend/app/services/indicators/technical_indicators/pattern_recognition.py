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

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import handle_pandas_ta_errors


# TA-Libの利用可能性チェック
TA_LIB_AVAILABLE = False
try:
    import talib
    TA_LIB_AVAILABLE = True
except ImportError:
    TA_LIB_AVAILABLE = False


class PatternRecognitionIndicators:
    """
    パターン認識系指標クラス
    """

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_doji(
        open_data: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Doji（重要パターンのためエラーハンドリング付き）"""
        if not isinstance(open_data, pd.Series):
            raise TypeError("open_data must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.cdl_pattern(
            open_=open_data,
            high=high,
            low=low,
            close=close,
            name="doji",
        )
        return (
            result.iloc[:, 0]
            if result is not None and not result.empty
            else pd.Series(np.zeros(len(open_data)), index=open_data.index)
        )

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_hammer(
        open_data: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Hammer（重要パターンのためエラーハンドリング付き）"""
        if not isinstance(open_data, pd.Series):
            raise TypeError("open_data must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.cdl_pattern(
            open_=open_data,
            high=high,
            low=low,
            close=close,
            name="hammer",
        )
        return (
            result.iloc[:, 0]
            if result is not None and not result.empty
            else pd.Series(np.zeros(len(open_data)), index=open_data.index)
        )

    @staticmethod
    def cdl_hanging_man(
        open_data: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Hanging Man（軽量実装）"""
        if not isinstance(open_data, pd.Series):
            raise TypeError("open_data must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.cdl_pattern(
            open_=open_data,
            high=high,
            low=low,
            close=close,
            name="hangingman",
        )
        return (
            result.iloc[:, 0]
            if result is not None and not result.empty
            else pd.Series(np.zeros(len(open_data)), index=open_data.index)
        )

    @staticmethod
    def cdl_shooting_star(
        open_data: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Shooting Star（軽量実装）"""
        if not isinstance(open_data, pd.Series):
            raise TypeError("open_data must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.cdl_pattern(
            open_=open_data,
            high=high,
            low=low,
            close=close,
            name="shootingstar",
        )
        return (
            result.iloc[:, 0]
            if result is not None and not result.empty
            else pd.Series(np.zeros(len(open_data)), index=open_data.index)
        )

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_engulfing(
        open_data: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Engulfing Pattern（重要パターンのためエラーハンドリング付き）"""
        if not isinstance(open_data, pd.Series):
            raise TypeError("open_data must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        if not TA_LIB_AVAILABLE:
            return pd.Series(np.zeros(len(open_data)), index=open_data.index)

        result = ta.cdl_pattern(
            open_=open_data,
            high=high,
            low=low,
            close=close,
            name="engulfing",
        )
        return (
            result.iloc[:, 0]
            if result is not None and not result.empty
            else pd.Series(np.zeros(len(open_data)), index=open_data.index)
        )

    @staticmethod
    def cdl_harami(
        open_data: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Harami Pattern（軽量実装）"""
        if not isinstance(open_data, pd.Series):
            raise TypeError("open_data must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        if not TA_LIB_AVAILABLE:
            return pd.Series(np.zeros(len(open_data)), index=open_data.index)

        result = ta.cdl_pattern(
            open_=open_data,
            high=high,
            low=low,
            close=close,
            name="harami",
        )
        return (
            result.iloc[:, 0]
            if result is not None and not result.empty
            else pd.Series(np.zeros(len(open_data)), index=open_data.index)
        )

    @staticmethod
    def cdl_piercing(
        open_data: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Piercing Pattern（軽量実装）"""
        if not isinstance(open_data, pd.Series):
            raise TypeError("open_data must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.cdl_pattern(
            open_=open_data,
            high=high,
            low=low,
            close=close,
            name="piercing",
        )
        return (
            result.iloc[:, 0]
            if result is not None and not result.empty
            else pd.Series(np.zeros(len(open_data)), index=open_data.index)
        )

    @staticmethod
    def cdl_dark_cloud_cover(
        open_data: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Dark Cloud Cover"""
        if not isinstance(open_data, pd.Series):
            raise TypeError("open_data must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.cdl_pattern(
            open_=open_data,
            high=high,
            low=low,
            close=close,
            name="darkcloudcover",
        )
        return (
            result.iloc[:, 0]
            if result is not None and not result.empty
            else pd.Series(np.zeros(len(open_data)), index=open_data.index)
        )

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_morning_star(
        open_data: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Morning Star（重要パターンのためエラーハンドリング付き）"""
        if not isinstance(open_data, pd.Series):
            raise TypeError("open_data must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.cdl_pattern(
            open_=open_data,
            high=high,
            low=low,
            close=close,
            name="morningstar",
        )
        return (
            result.iloc[:, 0]
            if result is not None and not result.empty
            else pd.Series(np.zeros(len(open_data)), index=open_data.index)
        )

    @staticmethod
    @handle_pandas_ta_errors
    def cdl_evening_star(
        open_data: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Evening Star（重要パターンのためエラーハンドリング付き）"""
        if not isinstance(open_data, pd.Series):
            raise TypeError("open_data must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.cdl_pattern(
            open_=open_data,
            high=high,
            low=low,
            close=close,
            name="eveningstar",
        )
        return (
            result.iloc[:, 0]
            if result is not None and not result.empty
            else pd.Series(np.zeros(len(open_data)), index=open_data.index)
        )

    @staticmethod
    def cdl_three_black_crows(
        open_data: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Three Black Crows（軽量実装）"""
        if not isinstance(open_data, pd.Series):
            raise TypeError("open_data must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        if not TA_LIB_AVAILABLE:
            return pd.Series(np.zeros(len(open_data)), index=open_data.index)

        result = ta.cdl_pattern(
            open_=open_data,
            high=high,
            low=low,
            close=close,
            name="3blackcrows",
        )
        return (
            result.iloc[:, 0]
            if result is not None and not result.empty
            else pd.Series(np.zeros(len(open_data)), index=open_data.index)
        )

    @staticmethod
    def cdl_three_white_soldiers(
        open_data: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Three White Soldiers（軽量実装）"""
        if not isinstance(open_data, pd.Series):
            raise TypeError("open_data must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        if not TA_LIB_AVAILABLE:
            return pd.Series(np.zeros(len(open_data)), index=open_data.index)

        result = ta.cdl_pattern(
            open_=open_data,
            high=high,
            low=low,
            close=close,
            name="3whitesoldiers",
        )
        return (
            result.iloc[:, 0]
            if result is not None and not result.empty
            else pd.Series(np.zeros(len(open_data)), index=open_data.index)
        )

    # 簡易実装（pandas-taにない場合のフォールバック）
    @staticmethod
    def cdl_marubozu(
        open_data: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Marubozu（簡易実装）"""
        if not isinstance(open_data, pd.Series):
            raise TypeError("open_data must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # 簡易Marubozu判定（実体が全体の90%以上）
        body = np.abs(close - open_data)
        range_hl = high - low

        # ゼロ除算を避ける
        range_hl = range_hl.where(range_hl != 0, 1e-10)

        marubozu = np.where(
            body / range_hl > 0.9, np.where(close > open_data, 100, -100), 0
        )

        return pd.Series(marubozu, index=open_data.index)

    @staticmethod
    def cdl_spinning_top(
        open_data: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Spinning Top（簡易実装）"""
        if not isinstance(open_data, pd.Series):
            raise TypeError("open_data must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # 簡易Spinning Top判定（小さな実体と長いヒゲ）
        body = np.abs(close - open_data)
        upper_shadow = high - pd.concat([open_data, close], axis=1).max(axis=1)
        lower_shadow = pd.concat([open_data, close], axis=1).min(axis=1) - low

        # 実体が小さく、上下のヒゲが長い
        spinning_top = np.where(
            (body < (upper_shadow + lower_shadow) * 0.3)
            & (upper_shadow > body * 2)
            & (lower_shadow > body * 2),
            100,
            0,
        )

        return pd.Series(spinning_top, index=open_data.index)

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

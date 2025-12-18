"""
価格特徴量計算クラス

OHLCV価格データから基本的な価格関連特徴量を計算します。
単一責任原則に従い、価格特徴量の計算のみを担当します。
テクニカル指標（ATR, VWAPなど）はtechnical_features.pyに移動しました。
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from ....utils.error_handler import safe_ml_operation
from ...indicators.technical_indicators.momentum import MomentumIndicators
from ...indicators.technical_indicators.trend import TrendIndicators
from .base_feature_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


class PriceFeatureCalculator(BaseFeatureCalculator):
    """
    価格特徴量計算クラス

    OHLCV価格データから基本的な価格関連特徴量（変化率、実体、ヒゲなど）を計算します。
    """

    def __init__(self):
        """初期化"""
        super().__init__()

    def calculate_features(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """価格特徴量を計算"""
        res = self.create_result_dataframe(df)
        lookback = config.get("lookback_periods", {})

        # 各カテゴリの計算（resを直接更新）
        res = self.calculate_price_features(res, lookback)
        res = self.calculate_statistical_features(res, lookback)
        res = self.calculate_time_series_features(res, lookback)
        res = self.calculate_volatility_features(res, lookback)

        return res

    @safe_ml_operation(default_return=None, context="統計的特徴量計算エラー")
    def calculate_statistical_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """統計的特徴量を計算"""
        w = 20
        # 範囲、ボラティリティ、スキューネス
        df[f"Close_range_{w}"] = (df["high"].rolling(w).max() - df["low"].rolling(w).min()).fillna(0.0)
        log_rets = np.log(df["close"] / df["close"].shift(1))
        df[f"Historical_Volatility_{w}"] = (log_rets.rolling(w).std() * np.sqrt(252)).fillna(0.0)
        df[f"Price_Skewness_{w}"] = df["close"].rolling(w).skew().fillna(0.0)
        return df

    @safe_ml_operation(default_return=None, context="時系列特徴量計算エラー")
    def calculate_time_series_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """時系列特徴量を計算"""
        df["Trend_strength_20"] = TrendIndicators.linregslope(df["close"], length=20).fillna(0.0)
        return df

    @safe_ml_operation(default_return=None, context="ボラティリティ特徴量計算エラー")
    def calculate_volatility_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """ボラティリティ特徴量を計算"""
        hl_ratio = np.log(df["high"] / df["low"])
        df["Parkinson_Vol_20"] = (hl_ratio.rolling(20).var() * (1 / (4 * np.log(2)))).fillna(0.0)
        return df

    @safe_ml_operation(default_return=None, context="基本価格特徴量計算エラー")
    def calculate_price_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """基本価格特徴量を計算"""
        if not self.validate_input_data(df, ["open", "high", "low", "close", "volume"]):
            return df

        p_chg = MomentumIndicators.roc(df["close"], period=1).fillna(0.0)
        v_chg = MomentumIndicators.roc(df["volume"], period=1).fillna(0.0)
        df["Price_Volume_Trend"] = p_chg * v_chg
        return df

    def get_feature_names(self) -> list:
        """
        生成される価格特徴量名のリストを取得

        Returns:
            特徴量名のリスト
        """
        return [
            "Price_Volume_Trend",
            # 統計的特徴量
            "Close_range_20",
            "Historical_Volatility_20",
            "Price_Skewness_20",
            # 時系列特徴量
            "Trend_strength_20",
            # ボラティリティ特徴量
            "Parkinson_Vol_20",
        ]




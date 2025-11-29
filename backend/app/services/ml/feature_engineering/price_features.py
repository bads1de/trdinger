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

from ...indicators.technical_indicators.momentum import MomentumIndicators
from ....utils.error_handler import safe_ml_operation
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
        """
        価格特徴量を計算（BaseFeatureCalculatorの抽象メソッド実装）

        Args:
            df: OHLCV価格データ
            config: 計算設定

        Returns:
            価格特徴量が追加されたDataFrame
        """
        lookback_periods = config.get("lookback_periods", {})

        # 基本的な価格特徴量のみ計算
        df = self.calculate_price_features(df, lookback_periods)

        return df

    @safe_ml_operation(
        default_return=None, context="価格特徴量計算でエラーが発生しました"
    )
    def calculate_price_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """
        価格特徴量を計算

        Args:
            df: OHLCV価格データ
            lookback_periods: 計算期間設定

        Returns:
            価格特徴量が追加されたDataFrame
        """
        if not self.validate_input_data(df, ["open", "high", "low", "close", "volume"]):
            return df

        result_df = self.create_result_dataframe(df)

        # 価格変化率（MomentumIndicators使用）
        roc1 = MomentumIndicators.roc(result_df["close"], period=1)
        result_df["Price_Change_1"] = roc1.fillna(0.0)

        roc5 = MomentumIndicators.roc(result_df["close"], period=5)
        result_df["Price_Change_5"] = roc5.fillna(0.0)

        roc20 = MomentumIndicators.roc(result_df["close"], period=20)
        result_df["Price_Change_20"] = roc20.fillna(0.0)

        # ボディサイズ（実体の大きさ）
        result_df["Body_Size"] = (
            (abs(result_df["close"] - result_df["open"]) / result_df["close"])
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        # 下ヒゲのみ保持
        lower_shadow = (
            np.minimum(result_df["open"], result_df["close"]) - result_df["low"]
        ) / result_df["close"]
        lower_shadow = np.where(np.isinf(lower_shadow), np.nan, lower_shadow)
        result_df["Lower_Shadow"] = pd.Series(
            lower_shadow, index=result_df.index
        ).fillna(0.0)

        # 価格・出来高トレンド（MomentumIndicators使用）
        # これは価格と出来高の単純な積であり、特定の指標ではないためここに残す
        price_change = MomentumIndicators.roc(result_df["close"], period=1).fillna(0.0)
        volume_change = MomentumIndicators.roc(
            result_df["volume"], period=1
        ).fillna(0.0)

        result_df["Price_Volume_Trend"] = price_change * volume_change

        self.log_feature_calculation_complete("基本価格")
        return result_df

    def get_feature_names(self) -> list:
        """
        生成される価格特徴量名のリストを取得

        Returns:
            特徴量名のリスト
        """
        return [
            "Price_Change_1",
            "Price_Change_5",
            "Price_Change_20",
            "Body_Size",
            "Lower_Shadow",
            "Price_Volume_Trend",
        ]

"""
相互作用特徴量計算クラス

既存の特徴量同士を組み合わせて、より複雑な関係性を捉える
相互作用特徴量を計算します。
"""

import logging
import pandas as pd
import numpy as np
from typing import List

from ....utils.ml_error_handler import safe_ml_operation

logger = logging.getLogger(__name__)


class InteractionFeatureCalculator:
    """
    相互作用特徴量計算クラス

    既存の特徴量同士を組み合わせて相互作用特徴量を計算します。
    """

    def __init__(self):
        """初期化"""
        pass

    def calculate_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        相互作用特徴量を計算

        Args:
            df: 既存の特徴量を含むDataFrame

        Returns:
            相互作用特徴量が追加されたDataFrame
        """
        if df is None or df.empty:
            logger.warning("空のデータが提供されました")
            return df

        # 必要な特徴量が存在するかチェック
        if not self._check_required_features(df):
            logger.warning("相互作用特徴量の計算に必要な特徴量が不足しています")
            return df

        return self._calculate_interaction_features_internal(df)

    @safe_ml_operation(
        default_value=None, error_message="相互作用特徴量計算でエラーが発生しました"
    )
    def _calculate_interaction_features_internal(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        内部的な相互作用特徴量計算メソッド
        """
        result_df = df.copy()

        # ボラティリティ × モメンタム相互作用
        result_df = self._calculate_volatility_momentum_interactions(result_df)

        # 出来高 × トレンド相互作用
        result_df = self._calculate_volume_trend_interactions(result_df)

        # FR × RSI相互作用
        result_df = self._calculate_fr_rsi_interactions(result_df)

        # OI × 価格変動相互作用
        result_df = self._calculate_oi_price_interactions(result_df)

        logger.debug("相互作用特徴量計算完了")
        return result_df

    def _check_required_features(self, df: pd.DataFrame) -> bool:
        """必要な特徴量が存在するかチェック"""
        # 最低限必要な特徴量のリスト（実際の特徴量名に合わせて修正）
        required_features = [
            "Price_Momentum_14",
            "Price_Change_5",
            "Volume_Ratio",
            "Trend_Strength",
            "Breakout_Strength",
            "RSI",
            "FR_Normalized",
            "OI_Change_Rate",
            "OI_Trend",
        ]

        # ATRは複数の名前で存在する可能性があるため、別途チェック
        atr_variants = ["ATR", "ATR_20", "ATR_14"]
        has_atr = any(variant in df.columns for variant in atr_variants)

        missing_features = [
            feature for feature in required_features if feature not in df.columns
        ]

        if missing_features or not has_atr:
            if missing_features:
                logger.warning(f"不足している特徴量: {missing_features}")
            if not has_atr:
                logger.warning(
                    f"ATR系の特徴量が見つかりません。利用可能なATR: {[col for col in df.columns if 'ATR' in col]}"
                )
            return False

        return True

    def _calculate_volatility_momentum_interactions(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """ボラティリティ × モメンタム相互作用を計算"""
        result_df = df.copy()

        # ATR × Price_Momentum_14（ATRの名前を動的に検出）
        atr_column = None
        for atr_variant in ["ATR", "ATR_20", "ATR_14"]:
            if atr_variant in df.columns:
                atr_column = atr_variant
                break

        if atr_column and "Price_Momentum_14" in df.columns:
            atr_values = df[atr_column].replace([np.inf, -np.inf], np.nan)
            momentum_values = df["Price_Momentum_14"].replace([np.inf, -np.inf], np.nan)

            interaction = atr_values * momentum_values
            # 無限大値や極端に大きな値をクリップ
            interaction = np.clip(interaction, -1e6, 1e6)
            result_df["Volatility_Momentum_Interaction"] = interaction

        # Volatility_Spike × Price_Change_5
        if "Volatility_Spike" in df.columns and "Price_Change_5" in df.columns:
            price_change = df["Price_Change_5"].replace([np.inf, -np.inf], np.nan)
            interaction = df["Volatility_Spike"].astype(float) * price_change
            # 無限大値や極端に大きな値をクリップ
            interaction = np.clip(interaction, -1e6, 1e6)
            result_df["Volatility_Spike_Momentum"] = interaction

        return result_df

    def _calculate_volume_trend_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """出来高 × トレンド相互作用を計算"""
        result_df = df.copy()

        # Volume_Ratio × Trend_Strength
        if "Volume_Ratio" in df.columns and "Trend_Strength" in df.columns:
            volume_ratio = df["Volume_Ratio"].replace([np.inf, -np.inf], np.nan)
            trend_strength = df["Trend_Strength"].replace([np.inf, -np.inf], np.nan)

            interaction = volume_ratio * trend_strength
            # 無限大値や極端に大きな値をクリップ
            interaction = np.clip(interaction, -1e6, 1e6)
            result_df["Volume_Trend_Interaction"] = interaction

        # Volume_Spike × Breakout_Strength
        if "Volume_Spike" in df.columns and "Breakout_Strength" in df.columns:
            breakout_strength = df["Breakout_Strength"].replace(
                [np.inf, -np.inf], np.nan
            )
            interaction = df["Volume_Spike"].astype(float) * breakout_strength
            # 無限大値や極端に大きな値をクリップ
            interaction = np.clip(interaction, -1e6, 1e6)
            result_df["Volume_Breakout"] = interaction

        return result_df

    def _calculate_fr_rsi_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """FR × RSI相互作用を計算"""
        result_df = df.copy()

        # FR_Normalized × (RSI - 50)
        if "FR_Normalized" in df.columns and "RSI" in df.columns:
            fr_normalized = df["FR_Normalized"].replace([np.inf, -np.inf], np.nan)
            rsi_centered = df["RSI"] - 50

            interaction = fr_normalized * rsi_centered
            # 無限大値や極端に大きな値をクリップ
            interaction = np.clip(interaction, -1e6, 1e6)
            result_df["FR_RSI_Extreme"] = interaction

        # FR_Extreme_High × (RSI > 70)（実際の特徴量名に合わせて修正）
        if "FR_Extreme_High" in df.columns and "RSI" in df.columns:
            interaction = df["FR_Extreme_High"].astype(float) * (df["RSI"] > 70).astype(
                float
            )
            result_df["FR_Overbought"] = interaction

        # FR_Extreme_Low × (RSI < 30)（実際の特徴量名に合わせて修正）
        if "FR_Extreme_Low" in df.columns and "RSI" in df.columns:
            interaction = df["FR_Extreme_Low"].astype(float) * (df["RSI"] < 30).astype(
                float
            )
            result_df["FR_Oversold"] = interaction

        return result_df

    def _calculate_oi_price_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """OI × 価格変動相互作用を計算"""
        result_df = df.copy()

        # OI_Change_Rate × Price_Change_5
        if "OI_Change_Rate" in df.columns and "Price_Change_5" in df.columns:
            oi_change = df["OI_Change_Rate"].replace([np.inf, -np.inf], np.nan)
            price_change = df["Price_Change_5"].replace([np.inf, -np.inf], np.nan)

            interaction = oi_change * price_change
            # 無限大値や極端に大きな値をクリップ
            interaction = np.clip(interaction, -1e6, 1e6)
            result_df["OI_Price_Divergence"] = interaction

        # OI_Trend × Price_Momentum_14
        if "OI_Trend" in df.columns and "Price_Momentum_14" in df.columns:
            oi_trend = df["OI_Trend"].replace([np.inf, -np.inf], np.nan)
            momentum = df["Price_Momentum_14"].replace([np.inf, -np.inf], np.nan)

            interaction = oi_trend * momentum
            # 無限大値や極端に大きな値をクリップ
            interaction = np.clip(interaction, -1e6, 1e6)
            result_df["OI_Momentum_Alignment"] = interaction

        return result_df

    def get_feature_names(self) -> List[str]:
        """生成される特徴量名のリストを取得"""
        return [
            "Volatility_Momentum_Interaction",
            "Volatility_Spike_Momentum",
            "Volume_Trend_Interaction",
            "Volume_Breakout",
            "FR_RSI_Extreme",
            "FR_Overbought",
            "FR_Oversold",
            "OI_Price_Divergence",
            "OI_Momentum_Alignment",
        ]

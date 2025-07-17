"""
Fear & Greed Index 特徴量計算クラス

Fear & Greed Index データから市場センチメントを捉える特徴量を計算します。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

from ....utils.data_validation import DataValidator

logger = logging.getLogger(__name__)


class FearGreedFeatureCalculator:
    """
    Fear & Greed Index 特徴量計算クラス

    Fear & Greed Index データから特徴量を計算します。
    """

    def __init__(self):
        """初期化"""
        pass

    def calculate_fear_greed_features(
        self,
        df: pd.DataFrame,
        fear_greed_data: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        """
        Fear & Greed Index 特徴量を計算

        Args:
            df: OHLCV価格データ
            fear_greed_data: Fear & Greed Index データ
            lookback_periods: 計算期間設定

        Returns:
            Fear & Greed Index 特徴量が追加されたDataFrame
        """
        try:
            result_df = df.copy()

            if fear_greed_data is None or fear_greed_data.empty:
                logger.warning("Fear & Greed Index データが空です")
                return result_df

            # Fear & Greed Index データの準備
            fg_data = fear_greed_data.copy()
            
            # data_timestampをインデックスに設定
            if "data_timestamp" in fg_data.columns:
                fg_data = fg_data.set_index("data_timestamp")
            
            # valueカラムを確認
            if "value" not in fg_data.columns:
                logger.warning("Fear & Greed Index データにvalueカラムがありません")
                return result_df

            # インデックスを合わせてマージ
            merged_df = result_df.join(fg_data, how="left", rsuffix="_fg")

            # 欠損値を前方補完
            merged_df["value"] = merged_df["value"].ffill()

            # 基本的なFear & Greed特徴量を計算
            result_df = self._calculate_basic_features(result_df, merged_df, lookback_periods)
            
            # トレンド特徴量を計算
            result_df = self._calculate_trend_features(result_df, merged_df, lookback_periods)
            
            # 極値検出特徴量を計算
            result_df = self._calculate_extreme_features(result_df, merged_df, lookback_periods)
            
            # ボラティリティ特徴量を計算
            result_df = self._calculate_volatility_features(result_df, merged_df, lookback_periods)

            logger.debug("Fear & Greed Index 特徴量計算完了")
            return result_df

        except Exception as e:
            logger.error(f"Fear & Greed Index 特徴量計算エラー: {e}")
            return df

    def _calculate_basic_features(
        self, 
        result_df: pd.DataFrame, 
        merged_df: pd.DataFrame, 
        lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """基本的なFear & Greed特徴量を計算"""
        if "value" not in merged_df.columns:
            return result_df

        # Fear & Greed レベル（生値）
        result_df["FG_Level"] = merged_df["value"]

        # Fear & Greed 変化率
        result_df["FG_Change"] = DataValidator.safe_pct_change(merged_df["value"])

        # Fear & Greed 移動平均（短期・長期）
        result_df["FG_MA_7"] = DataValidator.safe_rolling_mean(
            merged_df["value"], window=7
        )
        result_df["FG_MA_30"] = DataValidator.safe_rolling_mean(
            merged_df["value"], window=30
        )

        # Fear & Greed 正規化値（0-1スケール）
        result_df["FG_Normalized"] = merged_df["value"] / 100.0

        return result_df

    def _calculate_trend_features(
        self, 
        result_df: pd.DataFrame, 
        merged_df: pd.DataFrame, 
        lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """トレンド特徴量を計算"""
        if "value" not in merged_df.columns:
            return result_df

        # Fear & Greed トレンド（短期MA vs 長期MA）
        if "FG_MA_7" in result_df.columns and "FG_MA_30" in result_df.columns:
            result_df["FG_Trend"] = DataValidator.safe_divide(
                result_df["FG_MA_7"], result_df["FG_MA_30"], default_value=1.0
            ) - 1

        # Fear & Greed モメンタム（5日間の変化）
        result_df["FG_Momentum_5"] = DataValidator.safe_pct_change(
            merged_df["value"], periods=5
        )

        # Fear & Greed 方向性（上昇・下降の判定）
        result_df["FG_Direction"] = np.where(
            result_df["FG_Change"] > 0, 1,
            np.where(result_df["FG_Change"] < 0, -1, 0)
        )

        return result_df

    def _calculate_extreme_features(
        self, 
        result_df: pd.DataFrame, 
        merged_df: pd.DataFrame, 
        lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """極値検出特徴量を計算"""
        if "value" not in merged_df.columns:
            return result_df

        # 極端な恐怖（0-25）
        result_df["FG_Extreme_Fear"] = (merged_df["value"] <= 25).astype(int)

        # 恐怖（26-45）
        result_df["FG_Fear"] = ((merged_df["value"] > 25) & (merged_df["value"] <= 45)).astype(int)

        # 中立（46-54）
        result_df["FG_Neutral"] = ((merged_df["value"] > 45) & (merged_df["value"] <= 54)).astype(int)

        # 強欲（55-74）
        result_df["FG_Greed"] = ((merged_df["value"] > 54) & (merged_df["value"] <= 74)).astype(int)

        # 極端な強欲（75-100）
        result_df["FG_Extreme_Greed"] = (merged_df["value"] > 74).astype(int)

        # 極値の継続期間（連続して極値にある日数）
        extreme_mask = (merged_df["value"] <= 25) | (merged_df["value"] > 74)
        result_df["FG_Extreme_Duration"] = self._calculate_consecutive_periods(extreme_mask)

        return result_df

    def _calculate_volatility_features(
        self, 
        result_df: pd.DataFrame, 
        merged_df: pd.DataFrame, 
        lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """ボラティリティ特徴量を計算"""
        if "value" not in merged_df.columns:
            return result_df

        # Fear & Greed ボラティリティ（20日間の標準偏差）
        result_df["FG_Volatility_20"] = DataValidator.safe_rolling_std(
            merged_df["value"], window=20
        )

        # Fear & Greed 変化率のボラティリティ
        if "FG_Change" in result_df.columns:
            result_df["FG_Change_Volatility"] = DataValidator.safe_rolling_std(
                result_df["FG_Change"], window=20
            )

        # Fear & Greed レンジ（期間内の最大値-最小値）
        result_df["FG_Range_20"] = (
            merged_df["value"].rolling(window=20, min_periods=1).max() -
            merged_df["value"].rolling(window=20, min_periods=1).min()
        )

        # Fear & Greed 位置（期間内での相対位置）
        fg_min = merged_df["value"].rolling(window=20, min_periods=1).min()
        fg_max = merged_df["value"].rolling(window=20, min_periods=1).max()
        result_df["FG_Position"] = DataValidator.safe_divide(
            merged_df["value"] - fg_min, fg_max - fg_min, default_value=0.5
        )

        return result_df

    def _calculate_consecutive_periods(self, mask: pd.Series) -> pd.Series:
        """連続期間を計算するヘルパーメソッド"""
        try:
            # 連続するTrueの期間を計算
            groups = (mask != mask.shift()).cumsum()
            consecutive = mask.groupby(groups).cumsum()
            return consecutive.where(mask, 0)
        except Exception as e:
            logger.warning(f"連続期間計算エラー: {e}")
            return pd.Series(0, index=mask.index)

    def get_feature_names(self) -> list:
        """
        生成されるFear & Greed Index特徴量名のリストを取得

        Returns:
            特徴量名のリスト
        """
        return [
            # 基本特徴量
            "FG_Level",
            "FG_Change",
            "FG_MA_7",
            "FG_MA_30",
            "FG_Normalized",
            # トレンド特徴量
            "FG_Trend",
            "FG_Momentum_5",
            "FG_Direction",
            # 極値検出特徴量
            "FG_Extreme_Fear",
            "FG_Fear",
            "FG_Neutral",
            "FG_Greed",
            "FG_Extreme_Greed",
            "FG_Extreme_Duration",
            # ボラティリティ特徴量
            "FG_Volatility_20",
            "FG_Change_Volatility",
            "FG_Range_20",
            "FG_Position",
        ]

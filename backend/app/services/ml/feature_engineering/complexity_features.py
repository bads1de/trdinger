"""
市場の複雑性（Complexity）と需給の不均衡（Order Flow）を捉える特徴量計算モジュール

Sample Entropy, Fractal Dimension, VPIN Approximation など、
相場の「質」を定量化する高度な指標を生成します。
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from ...indicators.technical_indicators.advanced_features import AdvancedFeatures
from .base_feature_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


class ComplexityFeatureCalculator(BaseFeatureCalculator):
    """
    複雑性・需給不均衡特徴量計算クラス
    """

    def calculate_features(
        self, df: pd.DataFrame, config: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        複雑性に関連する特徴量を計算

        Args:
            df: 入力データ (OHLCV)
            config: 設定辞書

        Returns:
            特徴量が追加されたDataFrame
        """
        if not self.validate_input_data(df, ["close", "volume"]):
            return pd.DataFrame(index=df.index)

        lookback_periods = config.get("lookback_periods", {}) if config else {}
        
        # 主要な計算期間
        short_p = lookback_periods.get("short", 20)
        mid_p = lookback_periods.get("mid", 50)
        long_p = lookback_periods.get("long", 100)

        new_features = {}

        # 1. ハースト指数 (Hurst Exponent) - 再計算または統合
        # すでにTechnicalFeatureCalculatorにあるかもしれないが、ここでは複数の期間で算出
        logger.info("Complexity: Hurst Exponent を計算中...")
        new_features[f"Hurst_{mid_p}"] = AdvancedFeatures.hurst_exponent(df["close"], window=mid_p)
        new_features[f"Hurst_{long_p}"] = AdvancedFeatures.hurst_exponent(df["close"], window=long_p)

        # 2. フラクタル次元 (Fractal Dimension)
        logger.info("Complexity: Fractal Dimension を計算中...")
        new_features[f"Fractal_Dim_{short_p}"] = AdvancedFeatures.fractal_dimension(df["close"], window=short_p)
        new_features[f"Fractal_Dim_{mid_p}"] = AdvancedFeatures.fractal_dimension(df["close"], window=mid_p)

        # 3. サンプル・エントロピー (Sample Entropy)
        # 計算コストが高いため、比較的小さな窓幅で計算
        logger.info("Complexity: Sample Entropy を計算中...")
        new_features[f"Sample_Entropy_{short_p}"] = AdvancedFeatures.sample_entropy(df["close"], window=short_p)
        
        # 4. 近似 VPIN (Order Flow Imbalance)
        logger.info("Complexity: VPIN Approximation を計算中...")
        new_features[f"VPIN_{short_p}"] = AdvancedFeatures.vpin_approximation(df["close"], df["volume"], window=short_p)
        new_features[f"VPIN_{mid_p}"] = AdvancedFeatures.vpin_approximation(df["close"], df["volume"], window=mid_p)

        # 5. 複合指標 (Interaction)
        # Hurstが高い(トレンド)かつEntropyが低い(秩序)時、トレンドの信頼性が高い
        new_features["Complexity_Trend_Trust"] = new_features[f"Hurst_{mid_p}"] / (new_features[f"Sample_Entropy_{short_p}"] + 1e-9)
        
        # 6. 効率性レシオの高度版 (Complexity-Adjusted Efficiency)
        # 従来のEfficiency Ratioをフラクタル次元で重み付け
        er = (df["close"].diff(mid_p).abs() / (df["close"].diff().abs().rolling(window=mid_p).sum() + 1e-9))
        new_features["Complexity_Adjusted_ER"] = er * (2.0 - new_features[f"Fractal_Dim_{mid_p}"])

        # DataFrameとして統合
        result_df = pd.DataFrame(new_features, index=df.index)
        
        # クリーニング
        result_df = result_df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        
        logger.info(f"複雑性特徴量を追加: {len(result_df.columns)}個")
        return result_df

    def get_feature_names(self) -> list:
        """
        生成される特徴量名のリストを取得
        """
        return [
            "Hurst_50", "Hurst_100",
            "Fractal_Dim_20", "Fractal_Dim_50",
            "Sample_Entropy_20",
            "VPIN_20", "VPIN_50",
            "Complexity_Trend_Trust",
            "Complexity_Adjusted_ER"
        ]

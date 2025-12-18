"""
暗号通貨特化の特徴量エンジニアリング

実際の取引データの特性を考慮した、効果的な特徴量を生成します。
OHLCV、OI、FR、FGデータの期間不一致を適切に処理し、
予測精度の向上に寄与する特徴量を作成します。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_feature_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


class CryptoFeatureCalculator(BaseFeatureCalculator):
    """暗号通貨特化の特徴量エンジニアリング"""

    def calculate_features(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """特徴量を計算"""
        return self.create_crypto_features(df)

    def create_crypto_features(
        self,
        df: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """暗号通貨特化特徴量を生成"""
        logger.info("暗号通貨特化特徴量を計算中...")
        
        # 必須カラムチェック
        if not self.validate_input_data(df, ["open", "high", "low", "close", "volume"]):
            return df

        result_df = df.copy()
        # 価格レベル特徴量 (24h)
        result_df["price_vs_low_24h"] = df["close"] / df["low"].rolling(24).min().fillna(df["low"])
        
        # クリーニング (inf除外)
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        
        logger.info("暗号通貨特化特徴量を追加: 1個")
        return result_df


# 後方互換性のためのクラス別名
CryptoFeatures = CryptoFeatureCalculator




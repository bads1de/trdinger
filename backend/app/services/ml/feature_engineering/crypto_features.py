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
    """暗号通貨特化の特徴量エンジニアリング（BaseFeatureCalculator継承）"""

    def __init__(self):
        """初期化"""
        super().__init__()
        self.feature_groups = {
            "price": [],
            "volume": [],
            "open_interest": [],
            "funding_rate": [],
            "technical": [],
            "composite": [],
            "temporal": [],
        }

    def calculate_features(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        暗号通貨特徴量を計算（BaseFeatureCalculatorの抽象メソッド実装）

        Args:
            df: OHLCV価格データ
            config: 計算設定（lookback_periodsを含む）

        Returns:
            暗号通貨特徴量が追加されたDataFrame
        """
        # create_crypto_featuresを呼び出し（後方互換性のため）
        result_df = self.create_crypto_features(df)

        return result_df

    def _ensure_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ品質を確保"""
        result_df = df.copy()

        # 必要なカラムの存在確認と補完
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in result_df.columns:
                logger.warning(f"必須カラム {col} が見つかりません")
                return result_df

        # オプショナルカラムの補完（削除: 生データは使用しない）
        # Removed: open_interest and funding_rate raw data columns
        # (低寄与度特徴量削除: 2025-01-07)
        # 理由: 加工済み特徴量のみを使用する設計に変更

        # 統計的手法による補完
        optional_columns = ["open_interest", "funding_rate"]
        for col in optional_columns:
            if col in result_df.columns and result_df[col].isna().any():
                result_df[col] = result_df[col].fillna(result_df[col].median())

        return result_df

    def _create_price_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """価格関連特徴量（PriceFeaturesと重複しない暗号通貨特化特徴量のみ）"""
        result_df = df.copy()

        # 新しい特徴量を辞書で収集（DataFrame断片化対策）
        new_features = {}

        # 価格レベル特徴量（暗号通貨特化）- 24hのみに削減（1週間は冗長）
        # price_vs_high_24h は削除
        new_features["price_vs_low_24h"] = df["close"] / df["low"].rolling(24).min()

        # 一括で結合（DataFrame断片化回避）
        result_df = pd.concat(
            [result_df, pd.DataFrame(new_features, index=result_df.index)], axis=1
        )

        # feature_groups更新
        self.feature_groups["price"].extend(
            [col for col in result_df.columns if col.startswith("price_vs_")]
        )

        return result_df

    # _create_volume_features は削除されました
    # _create_open_interest_features は削除されました
    # _create_funding_rate_features は削除されました
    # _create_technical_features は削除されました
    # _create_composite_features は削除されました
    # _create_temporal_features は削除されました

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量のクリーニング"""
        result_df = df.copy()

        # 無限値をNaNに変換
        result_df = result_df.replace([np.inf, -np.inf], np.nan)

        # データリーク防止のため、ここでの中央値補完は行わない
        # 補完はFeatureEngineeringServiceの最後で一括してffillで行う

        return result_df

    def create_crypto_features(
        self,
        df: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """暗号通貨特化特徴量を生成するパブリックメソッド"""
        logger.info("暗号通貨特化特徴量を計算中...")

        # デフォルトの計算期間
        periods = {
            "short": 14,
            "medium": 24,
            "long": 72,
            "very_long": 168,
        }

        result_df = self._ensure_data_quality(df)

        # 各特徴量グループを計算
        result_df = self._create_price_features(result_df, periods)
        # 他の特徴量生成メソッドは削除されました

        # 特徴量をクリーニング
        result_df = self._clean_features(result_df)

        # 統計情報を記録
        feature_count = len(result_df.columns) - len(df.columns)
        logger.info(f"暗号通貨特化特徴量を追加: {feature_count}個")

        return result_df


# 後方互換性のためのクラス別名
CryptoFeatures = CryptoFeatureCalculator



"""
相互作用特徴量計算クラス

既存の特徴量同士を組み合わせて、より複雑な関係性を捉える
相互作用特徴量を計算します。
"""

import logging
from typing import List

import pandas as pd


logger = logging.getLogger(__name__)


class InteractionFeatureCalculator:
    """
    相互作用特徴量計算クラス

    既存の特徴量同士を組み合わせて相互作用特徴量を計算します。
    """

    def __init__(self):
        """初期化"""

    def calculate_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        相互作用特徴量を計算

        現在の設定では相互作用特徴量は使用されていません。
        """
        return df

    def get_feature_names(self) -> List[str]:
        """生成される特徴量名のリストを取得"""
        return []

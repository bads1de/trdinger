"""
特徴量選択戦略の基底クラス
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

from ..config import FeatureSelectionConfig


class BaseSelectionStrategy(ABC):
    """特徴量選択戦略の基底クラス"""

    @abstractmethod
    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: FeatureSelectionConfig,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        特徴量選択を実行

        Returns:
            (support_mask, details): 選択マスクと詳細情報
        """
        pass

    def _limit_features(
        self,
        mask: np.ndarray,
        scores: np.ndarray,
        config: FeatureSelectionConfig,
    ) -> np.ndarray:
        """最大個数制限を適用するヘルパー"""
        if config.max_features is None or mask.sum() <= config.max_features:
            return mask

        support_indices = np.where(mask)[0]
        support_scores = scores[support_indices]

        top_k_indices = support_indices[
            np.argsort(support_scores)[-config.max_features :]
        ]

        new_mask = np.zeros_like(mask, dtype=bool)
        new_mask[top_k_indices] = True
        return new_mask

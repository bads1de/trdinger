"""
分散に基づくフィルタリング戦略（定数・準定数の削除）
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_selection import VarianceThreshold

from ..config import FeatureSelectionConfig
from .base import BaseSelectionStrategy


class VarianceStrategy(BaseSelectionStrategy):
    """分散に基づくフィルタ（定数・準定数の削除）"""

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: FeatureSelectionConfig,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        selector = VarianceThreshold(threshold=config.variance_threshold)
        selector.fit(X)
        mask = selector.get_support()
        return mask, {
            "method": "variance",
            "variances": selector.variances_.tolist(),
            "threshold": config.variance_threshold,
        }

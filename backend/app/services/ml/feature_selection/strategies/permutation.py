"""
Permutation Importance による特徴量選択戦略
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

from ..config import FeatureSelectionConfig
from ..utils import get_default_estimator
from .base import BaseSelectionStrategy


class PermutationStrategy(BaseSelectionStrategy):
    """Permutation Importance による選択"""

    def __init__(self, estimator: Optional[BaseEstimator] = None):
        self.estimator = estimator

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: FeatureSelectionConfig,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        model = self.estimator or get_default_estimator(
            n_estimators=50,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )
        model.fit(X, y)

        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=10,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )

        importances = result.importances_mean
        mask = importances > config.importance_threshold

        if config.max_features and mask.sum() > config.max_features:
            mask = self._limit_features(mask, importances, config)

        if mask.sum() < config.min_features:
            top_k = np.argsort(importances)[-config.min_features :]
            mask = np.zeros(len(feature_names), dtype=bool)
            mask[top_k] = True

        return mask, {
            "method": "permutation",
            "importances_mean": importances.tolist(),
            "importances_std": result.importances_std.tolist(),
        }

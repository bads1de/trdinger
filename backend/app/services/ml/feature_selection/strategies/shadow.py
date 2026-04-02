"""
シャドウ特徴量ベースの選択戦略（Boruta風）
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator

from ..config import FeatureSelectionConfig
from ..utils import get_default_estimator
from .base import BaseSelectionStrategy


class ShadowFeatureStrategy(BaseSelectionStrategy):
    """
    シャドウ特徴量ベースの選択（Boruta風）

    ランダムにシャッフルした「シャドウ特徴量」より重要度が高い
    特徴量のみを選択。統計的に有意なノイズ除去が可能。
    """

    def __init__(self, estimator: Optional[BaseEstimator] = None):
        self.estimator = estimator

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: FeatureSelectionConfig,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        n_features = X.shape[1]
        hit_counts = np.zeros(n_features)

        rng = np.random.RandomState(config.random_state)

        for iteration in range(config.shadow_iterations):
            X_shadow = X.copy()
            for col in range(n_features):
                rng.shuffle(X_shadow[:, col])

            X_extended = np.hstack([X, X_shadow])

            model = self.estimator or get_default_estimator(
                n_estimators=50,
                random_state=config.random_state + iteration,
                n_jobs=config.n_jobs,
            )
            model.fit(X_extended, y)  # type: ignore[reportAttributeAccessIssue]

            importances = model.feature_importances_  # type: ignore[reportAttributeAccessIssue]
            real_importances = importances[:n_features]
            shadow_importances = importances[n_features:]

            shadow_max = np.percentile(shadow_importances, config.shadow_percentile)

            hit_counts[real_importances > shadow_max] += 1

        threshold = config.shadow_iterations / 2
        mask = hit_counts > threshold

        if mask.sum() < config.min_features:
            top_k = np.argsort(hit_counts)[-config.min_features :]
            mask = np.zeros(n_features, dtype=bool)
            mask[top_k] = True

        return mask, {
            "method": "shadow",
            "hit_counts": hit_counts.tolist(),
            "threshold": threshold,
            "confirmed_count": int(mask.sum()),
        }

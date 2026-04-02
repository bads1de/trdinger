"""
L1正則化による埋め込み選択戦略
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

from ..config import FeatureSelectionConfig
from .base import BaseSelectionStrategy


class LassoStrategy(BaseSelectionStrategy):
    """L1正則化による埋め込み選択"""

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: FeatureSelectionConfig,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        model = LassoCV(
            cv=config.cv_folds, random_state=config.random_state, n_jobs=config.n_jobs
        )
        model.fit(X, y)

        selector = SelectFromModel(
            model, prefit=True, threshold=config.importance_threshold
        )
        mask = selector.get_support()
        if mask is None:
            mask = np.ones(len(feature_names), dtype=bool)

        if mask.sum() < config.min_features:
            top_k = np.argsort(np.abs(model.coef_))[-config.min_features :]
            mask = np.zeros(len(feature_names), dtype=bool)
            mask[top_k] = True

        return mask, {
            "method": "lasso",
            "coefficients": model.coef_.tolist(),
            "alpha": model.alpha_,
        }

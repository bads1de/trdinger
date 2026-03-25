"""
単変量統計テストによる特徴量選択戦略
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
)

from ..config import FeatureSelectionConfig
from .base import BaseSelectionStrategy


class UnivariateStrategy(BaseSelectionStrategy):
    """単変量統計テストによる選択"""

    def __init__(self, score_func: str = "f_classif"):
        self.score_func_name = score_func
        self.score_funcs = {
            "f_classif": f_classif,
            "chi2": chi2,
            "mutual_info": mutual_info_classif,
        }

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: FeatureSelectionConfig,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        k = (
            config.target_k
            or config.max_features
            or max(config.min_features, int(len(feature_names) * 0.5))
        )
        k = min(k, len(feature_names))

        score_func = self.score_funcs.get(self.score_func_name, f_classif)
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)

        pvalues = None
        if hasattr(selector, "pvalues_") and selector.pvalues_ is not None:
            pvalues = selector.pvalues_.tolist()

        return selector.get_support(), {
            "method": f"univariate_{self.score_func_name}",
            "scores": selector.scores_.tolist(),
            "pvalues": pvalues,
        }

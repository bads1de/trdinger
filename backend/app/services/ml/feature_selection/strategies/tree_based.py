"""
ツリーベースモデルの特徴量重要度による選択戦略
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel

from ..config import FeatureSelectionConfig
from ..utils import get_task_appropriate_estimator
from .base import BaseSelectionStrategy


class TreeBasedStrategy(BaseSelectionStrategy):
    """
    ツリーベースモデルの特徴量重要度による選択

    デフォルトはLightGBM。カスタムestimatorも注入可能。
    ターゲットが回帰か分類かを自動判定して適切なestimatorを選択する。
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
        from typing import cast

        model = cast(
            Any,
            self.estimator
            or get_task_appropriate_estimator(
                y,
                n_estimators=100,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
            ),
        )
        model.fit(X, y)

        selector = SelectFromModel(
            model, prefit=True, threshold=config.importance_threshold
        )
        mask = cast(np.ndarray, selector.get_support())

        if config.max_features and mask.sum() > config.max_features:
            mask = self._limit_features(mask, model.feature_importances_, config)

        if config.min_features and mask.sum() < config.min_features:
            top_k = np.argsort(model.feature_importances_)[-config.min_features :]
            mask = np.zeros(len(feature_names), dtype=bool)
            mask[top_k] = True

        return mask, {
            "method": "tree_based",
            "importances": model.feature_importances_.tolist(),
            "model_type": type(model).__name__,
        }

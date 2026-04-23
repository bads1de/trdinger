"""
再帰的特徴量削減（クロスバリデーション付き）戦略
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

from ..config import FeatureSelectionConfig
from ..utils import get_default_estimator
from .base import BaseSelectionStrategy


class RFECVStrategy(BaseSelectionStrategy):
    """再帰的特徴量削減（クロスバリデーション付き）"""

    def __init__(self, estimator: Optional[BaseEstimator] = None):
        self.estimator = estimator

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: FeatureSelectionConfig,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        estimator = self.estimator or get_default_estimator(
            n_estimators=50,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )

        if config.cv_strategy == "timeseries":
            from sklearn.model_selection import TimeSeriesSplit

            cv = TimeSeriesSplit(n_splits=config.cv_folds)
        else:
            cv = StratifiedKFold(
                n_splits=config.cv_folds,
                shuffle=True,
                random_state=config.random_state,
            )

        min_features = config.target_k or config.min_features
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=cv,
            scoring="accuracy",
            min_features_to_select=min_features,
            n_jobs=config.n_jobs,
        )
        rfecv.fit(X, y)

        mask = rfecv.support_
        if config.max_features and mask.sum() > config.max_features:
            ranking = rfecv.ranking_
            mask = ranking <= config.max_features

        return mask, {
            "method": "rfecv",
            "n_features": int(mask.sum()),
            "ranking": rfecv.ranking_.tolist(),
            "cv_results": (
                rfecv.cv_results_ if hasattr(rfecv, "cv_results_") else None
            ),
        }

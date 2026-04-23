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
    """L1正則化（Lasso）による埋め込み特徴量選択戦略。

    LassoCV（Cross-Validation付きLasso）を使用して、
    各特徴量の係数を自動調整します。係数が0になる特徴量を除去し、
    重要な特徴量のみを選択します。CVにより最適な正則化強度（alpha）が
    自動的に選択されます。
    """

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: FeatureSelectionConfig,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Lasso回帰の係数に基づいて特徴量を選択する。

        LassoCVでモデルを学習し、係数の絶対値がthreshold以上の特徴量を
        選択します。min_features未満の場合は、係数の大きい方から
        min_features個を強制的に選択します。

        Args:
            X: 特徴量配列（n_samples, n_features）。
            y: ターゲット配列。
            feature_names: 特徴量名のリスト。
            config: 特徴量選択設定。cv_folds、importance_threshold、
                min_features、random_state、n_jobsが使用される。

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - 選択マスク（True=選択、False=除去）
                - メタデータ（係数、alpha値、メソッド名を含む）
        """
        model = LassoCV(
            cv=config.cv_folds,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
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

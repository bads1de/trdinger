"""
分散に基づくフィルタリング戦略（定数・準定数の削除）
"""

from typing import Any, Dict, List, Tuple, cast

import numpy as np
from sklearn.feature_selection import VarianceThreshold

from ..config import FeatureSelectionConfig
from .base import BaseSelectionStrategy


class VarianceStrategy(BaseSelectionStrategy):
    """分散に基づく特徴量選択戦略。

    sklearn.feature_selection.VarianceThresholdを使用して、
    分散が閾値未満の特徴量（定数や準定数）をフィルタリングします。
    変動の少ない特徴量を除去することで、モデルの学習効率を向上させます。
    """

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: FeatureSelectionConfig,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """分散閾値に基づいて特徴量を選択する。

        各特徴量の分散を計算し、config.variance_threshold以上の分散を持つ
        特徴量のみを選択します。

        Args:
            X: 特徴量配列（n_samples, n_features）。
            y: ターゲット配列（この戦略では使用しない）。
            feature_names: 特徴量名のリスト。
            config: 特徴量選択設定。variance_thresholdが使用される。

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - 選択マスク（True=選択、False=除去）
                - メタデータ（分散値、閾値、メソッド名を含む）
        """
        selector = VarianceThreshold(threshold=config.variance_threshold)
        selector.fit(X)
        mask = cast(np.ndarray, selector.get_support())
        return mask, {
            "method": "variance",
            "variances": (
                cast(np.ndarray, selector.variances_).tolist()
                if selector.variances_ is not None
                else []
            ),
            "threshold": config.variance_threshold,
        }

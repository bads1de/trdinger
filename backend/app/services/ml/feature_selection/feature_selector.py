"""
特徴量選択システム（プロフェッショナル・エディション v2）

scikit-learn完全互換のカスタム特徴量選択器。
BaseEstimator + SelectorMixinを継承し、Pipelineとシームレスに統合可能。

ベストプラクティス:
- Filter, Wrapper, Embedded手法の階層的適用
- RFECV による最適な特徴量数の自動発見
- シャドウ特徴量ベースのノイズ検出（Boruta風）
- 完全なクロスバリデーション対応
"""

import logging
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from .config import FeatureSelectionConfig, SelectionMethod
from .strategies import (
    BaseSelectionStrategy,
    LassoStrategy,
    PermutationStrategy,
    RFECVStrategy,
    ShadowFeatureStrategy,
    StagedStrategy,
    TreeBasedStrategy,
    UnivariateStrategy,
    VarianceStrategy,
)

logger = logging.getLogger(__name__)


class FeatureSelector(SelectorMixin, BaseEstimator):
    """
    scikit-learn互換の特徴量選択器

    Pipeline内での使用を想定し、以下の機能を提供:
    - fit(X, y): 特徴量選択ルールを学習
    - transform(X): 選択された特徴量のみを返す
    - get_support(): 選択マスクを取得
    - get_feature_names_out(): 選択された特徴量名を取得

    Example:
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.ensemble import RandomForestClassifier
        >>>
        >>> pipe = Pipeline([
        ...     ('selector', FeatureSelector(method='staged')),
        ...     ('clf', RandomForestClassifier())
        ... ])
        >>> pipe.fit(X, y)
    """

    def __init__(
        self,
        method: Union[str, SelectionMethod] = "staged",
        variance_threshold: float = 0.0,
        correlation_threshold: float = 0.90,
        target_k: Optional[int] = None,
        min_features: int = 5,
        max_features: Optional[int] = None,
        cumulative_importance: float = 0.95,
        min_relative_importance: float = 0.01,
        importance_threshold: float = 0.001,
        cv_folds: int = 5,
        cv_strategy: str = "stratified",
        random_state: int = 42,
        n_jobs: int = 1,
        shadow_iterations: int = 20,
        staged_methods: Optional[List[str]] = None,
    ):
        self.method = method
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.target_k = target_k
        self.min_features = min_features
        self.max_features = max_features
        self.cumulative_importance = cumulative_importance
        self.min_relative_importance = min_relative_importance
        self.importance_threshold = importance_threshold
        self.cv_folds = cv_folds
        self.cv_strategy = cv_strategy
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.shadow_iterations = shadow_iterations
        self.staged_methods = staged_methods

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> "FeatureSelector":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X = np.asarray(X)
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]

        y = np.asarray(y)

        X, y = self._validate_input(X, y)
        config = self._build_config()

        X_processed, correlation_mask = self._remove_correlated_features(X)
        processed_names = [
            self.feature_names_in_[i] for i, m in enumerate(correlation_mask) if m
        ]

        strategy = self._get_strategy()
        selection_mask, self.selection_details_ = strategy.select(
            X_processed, y, processed_names, config
        )

        self.support_ = np.zeros(len(self.feature_names_in_), dtype=bool)
        processed_indices = np.where(correlation_mask)[0]
        for i, selected in enumerate(selection_mask):
            if selected:
                self.support_[processed_indices[i]] = True

        n_original = len(self.feature_names_in_)
        n_selected = self.support_.sum()
        logger.info(
            f"Feature selection complete: {n_original} -> {n_selected} features"
        )

        return self

    def _get_support_mask(self) -> np.ndarray:  # type: ignore[override]
        check_is_fitted(self, "support_")
        return self.support_

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """選択された特徴量を抽出する"""
        X_selected = np.asarray(super().transform(X))

        if isinstance(X, pd.DataFrame):
            selected_features = self.get_feature_names_out()
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

        return X_selected

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **_fit_params: Any,
    ) -> Union[pd.DataFrame, np.ndarray]:
        if y is None:
            raise ValueError("FeatureSelector.fit_transform requires y")
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> np.ndarray:
        if input_features is None:
            input_features = self.feature_names_in_
        return np.array([input_features[i] for i, s in enumerate(self.support_) if s])

    def _validate_input(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Empty input data")

        if X.shape[0] != len(y):
            raise ValueError(
                f"X and y have inconsistent samples: {X.shape[0]} vs {len(y)}"
            )

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, y

    def _build_config(self) -> FeatureSelectionConfig:
        staged_methods = None
        if self.staged_methods:
            staged_methods = [SelectionMethod(m) for m in self.staged_methods]

        method = (
            self.method
            if isinstance(self.method, SelectionMethod)
            else SelectionMethod(self.method)
        )

        return FeatureSelectionConfig(
            method=method,
            variance_threshold=self.variance_threshold,
            correlation_threshold=self.correlation_threshold,
            target_k=self.target_k,
            min_features=self.min_features,
            max_features=self.max_features,
            cumulative_importance=self.cumulative_importance,
            min_relative_importance=self.min_relative_importance,
            importance_threshold=self.importance_threshold,
            cv_folds=self.cv_folds,
            cv_strategy=self.cv_strategy,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            shadow_iterations=self.shadow_iterations,
            staged_methods=staged_methods
            or [
                SelectionMethod.VARIANCE,
                SelectionMethod.MUTUAL_INFO,
                SelectionMethod.RFECV,
            ],
        )

    def _remove_correlated_features(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.ones(X.shape[1], dtype=bool)

        if self.correlation_threshold >= 1.0:
            return X, mask

        try:
            std = np.nanstd(X, axis=0)
            constant_mask = std < 1e-10

            if constant_mask.all():
                return X, mask

            non_constant_idx = np.where(~constant_mask)[0]

            if len(non_constant_idx) < 2:
                return X, mask

            X_non_const = X[:, non_constant_idx]
            corr_matrix = np.corrcoef(X_non_const, rowvar=False)

            if np.isnan(corr_matrix).any():
                return X, mask

            local_mask = np.ones(len(non_constant_idx), dtype=bool)
            abs_corr = np.abs(corr_matrix)

            for i in range(len(non_constant_idx)):
                if not local_mask[i]:
                    continue
                if i + 1 < len(non_constant_idx):
                    local_mask[i + 1 :] &= (
                        abs_corr[i, i + 1 :] <= self.correlation_threshold
                    )

            for local_idx, global_idx in enumerate(non_constant_idx):
                if not local_mask[local_idx]:
                    mask[global_idx] = False

            n_removed = (~mask).sum()
            if n_removed > 0:
                logger.info(f"Removed {n_removed} highly correlated features")

        except Exception as e:
            logger.warning(f"Correlation removal failed: {e}")

        return X[:, mask], mask

    def _get_strategy(self) -> BaseSelectionStrategy:
        method = (
            self.method
            if isinstance(self.method, SelectionMethod)
            else SelectionMethod(self.method)
        )

        strategy_map = {
            SelectionMethod.VARIANCE: VarianceStrategy(),
            SelectionMethod.UNIVARIATE_F: UnivariateStrategy("f_classif"),
            SelectionMethod.UNIVARIATE_CHI2: UnivariateStrategy("chi2"),
            SelectionMethod.MUTUAL_INFO: UnivariateStrategy("mutual_info"),
            SelectionMethod.RFE: RFECVStrategy(),
            SelectionMethod.RFECV: RFECVStrategy(),
            SelectionMethod.LASSO: LassoStrategy(),
            SelectionMethod.RANDOM_FOREST: TreeBasedStrategy(),
            SelectionMethod.PERMUTATION: PermutationStrategy(),
            SelectionMethod.SHADOW: ShadowFeatureStrategy(),
            SelectionMethod.STAGED: StagedStrategy(),
        }

        if method not in strategy_map:
            raise ValueError(f"Unknown selection method: {method}")

        return strategy_map[method]


def create_feature_selector(
    method: str = "staged",
    **kwargs,
) -> FeatureSelector:
    """
    特徴量選択器のファクトリー関数

    後方互換性のために提供。新規コードでは直接 FeatureSelector を使用してください。
    """
    return FeatureSelector(method=method, **kwargs)

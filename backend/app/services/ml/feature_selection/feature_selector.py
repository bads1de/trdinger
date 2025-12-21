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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

# LightGBM をデフォルトモデルとして使用（高速・高精度）
try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
from sklearn.feature_selection import (
    RFECV,
    SelectFromModel,
    SelectKBest,
    SelectorMixin,
    VarianceThreshold,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def get_default_estimator(
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = 1,
) -> BaseEstimator:
    """
    デフォルトのestimatorを取得

    LightGBMが利用可能ならLightGBM、そうでなければRandomForestを返す。
    """
    if LIGHTGBM_AVAILABLE:
        return LGBMClassifier(
            n_estimators=n_estimators,
            importance_type="gain",
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=-1,  # 警告抑制
            force_col_wise=True,  # 警告抑制
        )
    else:
        logger.warning("LightGBM not available, falling back to RandomForest")
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )


# =============================================================================
# Enums & Config
# =============================================================================


class SelectionMethod(Enum):
    """特徴量選択手法"""

    # Filter手法（高速・モデル非依存）
    VARIANCE = "variance"
    UNIVARIATE_F = "univariate_f"
    UNIVARIATE_CHI2 = "univariate_chi2"
    MUTUAL_INFO = "mutual_info"

    # Wrapper手法（計算コスト高・精度重視）
    RFE = "rfe"
    RFECV = "rfecv"
    PERMUTATION = "permutation"

    # Embedded手法（モデル学習と同時）
    LASSO = "lasso"
    RANDOM_FOREST = "random_forest"

    # 組み合わせ手法（推奨）
    SHADOW = "shadow"  # Boruta風シャドウ特徴量ベース
    STAGED = "staged"  # 段階的フィルタリング


@dataclass
class FeatureSelectionConfig:
    """
    特徴量選択設定

    ベストプラクティスに基づくデフォルト値を提供。
    """

    method: SelectionMethod = SelectionMethod.STAGED

    # --- Filter設定 ---
    variance_threshold: float = 0.0  # 定数・準定数の削除
    correlation_threshold: float = 0.90  # 高相関ペアの削除

    # --- Wrapper/Embedded設定 ---
    target_k: Optional[int] = None  # None = 自動決定 (RFECV)
    min_features: int = 5  # 最小特徴量数

    # --- 質による選別 ---
    cumulative_importance: float = 0.95  # 累積重要度閾値
    min_relative_importance: float = 0.01  # トップ比での足切り
    importance_threshold: float = 0.001  # 絶対閾値

    # --- クロスバリデーション ---
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # "stratified" or "timeseries"

    # --- 並列処理 ---
    random_state: int = 42
    n_jobs: int = 1  # デフォルトは安全のため1

    # --- シャドウ特徴量設定 (Boruta風) ---
    shadow_iterations: int = 20
    shadow_percentile: float = 100.0  # シャドウ最大値のパーセンタイル

    # --- Staged選択の段階 ---
    staged_methods: List[SelectionMethod] = field(
        default_factory=lambda: [
            SelectionMethod.VARIANCE,
            SelectionMethod.MUTUAL_INFO,
            SelectionMethod.RFECV,
        ]
    )


# =============================================================================
# Selector Strategies (Strategy Pattern)
# =============================================================================


class BaseSelectionStrategy(ABC):
    """特徴量選択戦略の基底クラス"""

    @abstractmethod
    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: FeatureSelectionConfig,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        特徴量選択を実行

        Returns:
            (support_mask, details): 選択マスクと詳細情報
        """
        pass


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
        k = config.target_k or max(config.min_features, int(len(feature_names) * 0.5))
        k = min(k, len(feature_names))

        score_func = self.score_funcs.get(self.score_func_name, f_classif)
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)

        # pvalues_ は mutual_info では None になる
        pvalues = None
        if hasattr(selector, "pvalues_") and selector.pvalues_ is not None:
            pvalues = selector.pvalues_.tolist()

        return selector.get_support(), {
            "method": f"univariate_{self.score_func_name}",
            "scores": selector.scores_.tolist(),
            "pvalues": pvalues,
        }


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
                n_splits=config.cv_folds, shuffle=True, random_state=config.random_state
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

        return rfecv.support_, {
            "method": "rfecv",
            "n_features": rfecv.n_features_,
            "ranking": rfecv.ranking_.tolist(),
            "cv_results": rfecv.cv_results_ if hasattr(rfecv, "cv_results_") else None,
        }


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

        # 最低限の特徴量を確保
        if mask.sum() < config.min_features:
            top_k = np.argsort(np.abs(model.coef_))[-config.min_features :]
            mask = np.zeros(len(feature_names), dtype=bool)
            mask[top_k] = True

        return mask, {
            "method": "lasso",
            "coefficients": model.coef_.tolist(),
            "alpha": model.alpha_,
        }


class TreeBasedStrategy(BaseSelectionStrategy):
    """
    ツリーベースモデルの特徴量重要度による選択

    デフォルトはLightGBM。カスタムestimatorも注入可能。
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
        model = self.estimator or get_default_estimator(
            n_estimators=100,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )
        model.fit(X, y)

        selector = SelectFromModel(
            model, prefit=True, threshold=config.importance_threshold
        )
        mask = selector.get_support()

        # 最低限の特徴量を確保
        if mask.sum() < config.min_features:
            top_k = np.argsort(model.feature_importances_)[-config.min_features :]
            mask = np.zeros(len(feature_names), dtype=bool)
            mask[top_k] = True

        return mask, {
            "method": "tree_based",
            "importances": model.feature_importances_.tolist(),
            "model_type": type(model).__name__,
        }


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

        # 最低限の特徴量を確保
        if mask.sum() < config.min_features:
            top_k = np.argsort(importances)[-config.min_features :]
            mask = np.zeros(len(feature_names), dtype=bool)
            mask[top_k] = True

        return mask, {
            "method": "permutation",
            "importances_mean": importances.tolist(),
            "importances_std": result.importances_std.tolist(),
        }


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
            # シャドウ特徴量を生成（各列を独立にシャッフル）
            X_shadow = X.copy()
            for col in range(n_features):
                rng.shuffle(X_shadow[:, col])

            # 元特徴量とシャドウを結合
            X_extended = np.hstack([X, X_shadow])

            # LightGBM/RandomForestで重要度を計算
            model = self.estimator or get_default_estimator(
                n_estimators=50,
                random_state=config.random_state + iteration,
                n_jobs=config.n_jobs,
            )
            model.fit(X_extended, y)

            importances = model.feature_importances_
            real_importances = importances[:n_features]
            shadow_importances = importances[n_features:]

            # シャドウ特徴量の最大値を閾値として使用
            shadow_max = np.percentile(shadow_importances, config.shadow_percentile)

            # 閾値を超えた特徴量をヒットとしてカウント
            hit_counts[real_importances > shadow_max] += 1

        # 過半数のイテレーションでヒットした特徴量を選択
        threshold = config.shadow_iterations / 2
        mask = hit_counts > threshold

        # 最低限の特徴量を確保
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


class StagedStrategy(BaseSelectionStrategy):
    """
    段階的特徴量選択（推奨）

    複数の手法を順番に適用し、段階的に絞り込む。
    Filter -> Wrapper -> Embedded の順序が推奨。
    """

    def __init__(
        self, strategies: Optional[Dict[SelectionMethod, BaseSelectionStrategy]] = None
    ):
        self.strategy_map = strategies or self._default_strategies()

    def _default_strategies(self) -> Dict[SelectionMethod, BaseSelectionStrategy]:
        return {
            SelectionMethod.VARIANCE: VarianceStrategy(),
            SelectionMethod.UNIVARIATE_F: UnivariateStrategy("f_classif"),
            SelectionMethod.UNIVARIATE_CHI2: UnivariateStrategy("chi2"),
            SelectionMethod.MUTUAL_INFO: UnivariateStrategy("mutual_info"),
            SelectionMethod.RFECV: RFECVStrategy(),
            SelectionMethod.LASSO: LassoStrategy(),
            SelectionMethod.RANDOM_FOREST: TreeBasedStrategy(),
            SelectionMethod.PERMUTATION: PermutationStrategy(),
            SelectionMethod.SHADOW: ShadowFeatureStrategy(),
        }

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: FeatureSelectionConfig,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        current_mask = np.ones(X.shape[1], dtype=bool)
        current_X = X
        current_names = feature_names.copy()
        stage_results = []

        for method in config.staged_methods:
            if method not in self.strategy_map:
                logger.warning(f"Unknown method in staged selection: {method}")
                continue

            strategy = self.strategy_map[method]

            try:
                stage_mask, stage_details = strategy.select(
                    current_X, y, current_names, config
                )

                # グローバルマスクを更新
                global_indices = np.where(current_mask)[0]
                for i, selected in enumerate(stage_mask):
                    if not selected:
                        current_mask[global_indices[i]] = False

                # 次の段階用にデータを絞り込み
                current_X = current_X[:, stage_mask]
                current_names = [
                    current_names[i] for i, s in enumerate(stage_mask) if s
                ]

                stage_results.append(
                    {
                        "method": method.value,
                        "selected_count": int(stage_mask.sum()),
                        "details": stage_details,
                    }
                )

                logger.info(
                    f"Stage [{method.value}]: {len(feature_names)} -> {current_mask.sum()} features"
                )

            except Exception as e:
                logger.warning(f"Stage [{method.value}] failed: {e}")
                stage_results.append({"method": method.value, "error": str(e)})

        return current_mask, {
            "method": "staged",
            "stages": stage_results,
            "final_count": int(current_mask.sum()),
        }


# =============================================================================
# Main Feature Selector (sklearn Compatible)
# =============================================================================


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
        cumulative_importance: float = 0.95,
        min_relative_importance: float = 0.01,
        importance_threshold: float = 0.001,
        cv_folds: int = 5,
        cv_strategy: str = "stratified",  # Added
        random_state: int = 42,
        n_jobs: int = 1,
        shadow_iterations: int = 20,
        staged_methods: Optional[List[str]] = None,
    ):
        """
        初期化

        Args:
            method: 選択手法 ('staged', 'shadow', 'rfecv', 'lasso', 'random_forest' など)
            variance_threshold: 分散閾値（これ以下は削除）
            correlation_threshold: 相関閾値（これ以上は片方削除）
            target_k: 目標特徴量数（Noneで自動決定）
            min_features: 最低保証する特徴量数
            cv_folds: クロスバリデーションのフォールド数
            cv_strategy: "stratified" (default) or "timeseries"
            random_state: 乱数シード
            n_jobs: 並列ジョブ数
            shadow_iterations: シャドウ特徴量のイテレーション数
            staged_methods: 段階的選択で使用する手法リスト
        """
        self.method = method
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.target_k = target_k
        self.min_features = min_features
        self.cumulative_importance = cumulative_importance
        self.min_relative_importance = min_relative_importance
        self.importance_threshold = importance_threshold
        self.cv_folds = cv_folds
        self.cv_strategy = cv_strategy  # Added
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.shadow_iterations = shadow_iterations
        self.staged_methods = staged_methods

    # ... (fit method)

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> "FeatureSelector":
        """
        特徴量選択ルールを学習

        Args:
            X: 特徴量データ (n_samples, n_features)
            y: ターゲット変数 (n_samples,)

        Returns:
            self
        """
        # DataFrame対応
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X = X.values
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        # 入力検証
        X, y = self._validate_input(X, y)

        # 設定オブジェクトを構築
        config = self._build_config()

        # 前処理: 高相関特徴量の削除
        X_processed, correlation_mask = self._remove_correlated_features(X)
        processed_names = [
            self.feature_names_in_[i] for i, m in enumerate(correlation_mask) if m
        ]

        # 選択戦略を取得・実行
        strategy = self._get_strategy()
        selection_mask, self.selection_details_ = strategy.select(
            X_processed, y, processed_names, config
        )

        # グローバルマスクを構築
        self.support_ = np.zeros(len(self.feature_names_in_), dtype=bool)
        processed_indices = np.where(correlation_mask)[0]
        for i, selected in enumerate(selection_mask):
            if selected:
                self.support_[processed_indices[i]] = True

        # 選択された特徴量数のログ
        n_original = len(self.feature_names_in_)
        n_selected = self.support_.sum()
        logger.info(
            f"Feature selection complete: {n_original} -> {n_selected} features"
        )

        return self

    def _get_support_mask(self) -> np.ndarray:
        """SelectorMixin用: 選択マスクを返す"""
        return self.support_

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        選択された特徴量を抽出する

        Args:
            X: 特徴量データ

        Returns:
            選択された特徴量データ
        """
        # 親クラス(SelectorMixin)のtransformを呼び出し（マスク適用）
        X_selected = super().transform(X)

        # 入力がDataFrameの場合、DataFrameとして返す（カラム名付与）
        if isinstance(X, pd.DataFrame):
            selected_features = self.get_feature_names_out()
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        return X_selected

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **fit_params
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        特徴量選択を実行し、選択済みデータを返す

        sklearn標準のインターフェースに準拠。
        詳細情報は self.selection_details_ 属性から取得可能です。

        Args:
            X: 特徴量データ
            y: ターゲット変数

        Returns:
            選択後の特徴量データ (DataFrame または ndarray)
        """
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> np.ndarray:
        """選択された特徴量名を返す"""
        if input_features is None:
            input_features = self.feature_names_in_
        return np.array([input_features[i] for i, s in enumerate(self.support_) if s])

    def _validate_input(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """入力データの検証と前処理"""
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Empty input data")

        if X.shape[0] != len(y):
            raise ValueError(
                f"X and y have inconsistent samples: {X.shape[0]} vs {len(y)}"
            )

        # 欠損値・無限値の処理
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y

    def _build_config(self) -> FeatureSelectionConfig:
        """パラメータから設定オブジェクトを構築"""
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
            cumulative_importance=self.cumulative_importance,
            min_relative_importance=self.min_relative_importance,
            importance_threshold=self.importance_threshold,
            cv_folds=self.cv_folds,
            cv_strategy=self.cv_strategy,  # Added
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
        """高相関特徴量を削除"""
        mask = np.ones(X.shape[1], dtype=bool)

        if self.correlation_threshold >= 1.0:
            return X, mask

        try:
            # 定数列を事前に検出（相関計算でNaNになる原因）
            std = np.nanstd(X, axis=0)
            constant_mask = std < 1e-10

            if constant_mask.all():
                # 全て定数の場合はそのまま返す
                return X, mask

            # 定数でない列のみで相関を計算
            non_constant_idx = np.where(~constant_mask)[0]

            if len(non_constant_idx) < 2:
                # 相関計算できる列が1つ以下
                return X, mask

            X_non_const = X[:, non_constant_idx]
            corr_matrix = np.corrcoef(X_non_const, rowvar=False)

            if np.isnan(corr_matrix).any():
                # それでもNaNがある場合は諦める
                return X, mask

            # 上三角行列で高相関ペアを検出
            # corr_matrixはX_non_constに対するものなので、インデックス変換が必要
            local_mask = np.ones(len(non_constant_idx), dtype=bool)

            for i in range(corr_matrix.shape[0]):
                if not local_mask[i]:
                    continue
                for j in range(i + 1, corr_matrix.shape[1]):
                    if (
                        local_mask[j]
                        and abs(corr_matrix[i, j]) > self.correlation_threshold
                    ):
                        local_mask[j] = False

            # ローカルマスクを元のインデックスに反映
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
        """選択手法に対応する戦略を取得"""
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
            SelectionMethod.RFE: RFECVStrategy(),  # RFEはRFECVにフォールバック
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


# =============================================================================
# Backward Compatibility
# =============================================================================


def create_feature_selector(
    method: str = "staged",
    **kwargs,
) -> FeatureSelector:
    """
    特徴量選択器のファクトリー関数

    後方互換性のために提供。新規コードでは直接 FeatureSelector を使用してください。
    """
    return FeatureSelector(method=method, **kwargs)

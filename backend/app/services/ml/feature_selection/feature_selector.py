"""
特徴量選択システム

分析報告書で提案された統計的特徴量選択とML-based特徴量選択を実装。
高次元データから重要な特徴量を効率的に選択します。
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV

logger = logging.getLogger(__name__)


class SelectionMethod(Enum):
    """特徴量選択手法"""

    # 統計的手法
    UNIVARIATE_F = "univariate_f"
    UNIVARIATE_CHI2 = "univariate_chi2"
    MUTUAL_INFO = "mutual_info"

    # ML-based手法
    LASSO = "lasso"
    RANDOM_FOREST = "random_forest"
    RFE = "rfe"
    RFECV = "rfecv"
    PERMUTATION = "permutation"

    # 組み合わせ手法
    ENSEMBLE = "ensemble"


@dataclass
class FeatureSelectionConfig:
    """特徴量選択設定"""

    method: SelectionMethod = SelectionMethod.ENSEMBLE
    k_features: Optional[int] = None  # 選択する特徴量数
    percentile: float = 50  # 上位何%を選択するか
    cv_folds: int = 5  # RFECVでの分割数
    random_state: int = 42
    n_jobs: int = -1

    # 閾値設定
    importance_threshold: float = 0.01
    correlation_threshold: float = 0.95

    # アンサンブル設定
    ensemble_methods: Optional[List[SelectionMethod]] = None
    ensemble_voting: str = "majority"  # "majority" or "unanimous"


class FeatureSelector:
    """
    特徴量選択器

    複数の手法を組み合わせて最適な特徴量セットを選択します。
    """

    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        """
        初期化

        Args:
            config: 特徴量選択設定
        """
        self.config = config or FeatureSelectionConfig()
        if self.config.ensemble_methods is None:
            self.config.ensemble_methods = [
                SelectionMethod.MUTUAL_INFO,
                SelectionMethod.RANDOM_FOREST,
                SelectionMethod.LASSO,
            ]

        self.selected_features_ = None
        self.feature_scores_ = None
        self.selection_results_ = {}

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        特徴量選択を実行

        Args:
            X: 特徴量DataFrame
            y: ターゲットSeries

        Returns:
            選択された特徴量のDataFrameと選択結果の辞書
        """
        logger.info(f"🎯 特徴量選択開始: {self.config.method.value}")
        logger.info(f"入力特徴量数: {X.shape[1]}, サンプル数: {X.shape[0]}")

        # データの前処理
        X_processed, feature_names = self._preprocess_data(X)

        # 特徴量選択実行
        if self.config.method == SelectionMethod.ENSEMBLE:
            selected_features, results = self._ensemble_selection(
                X_processed, y, feature_names
            )
        else:
            selected_features, results = self._single_method_selection(
                X_processed, y, feature_names, self.config.method
            )

        # 結果の保存
        self.selected_features_ = selected_features
        self.selection_results_ = results

        # 選択された特徴量でDataFrameを作成
        if selected_features:
            X_selected = X[selected_features].copy()
        else:
            # 選択された特徴量がない場合は空のDataFrameを返す
            X_selected = pd.DataFrame(index=X.index)

        # 結果がDataFrameであることを保証
        if not isinstance(X_selected, pd.DataFrame):
            X_selected = pd.DataFrame(X_selected)

        logger.info(f"✅ 特徴量選択完了: {len(selected_features)}個の特徴量を選択")
        logger.info(f"選択率: {len(selected_features) / X.shape[1] * 100:.1f}%")

        return X_selected, results

    def _preprocess_data(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """データの前処理"""
        # 欠損値の処理
        X_filled = X.fillna(X.median())

        # 無限値の処理
        X_filled = X_filled.replace([np.inf, -np.inf], np.nan)
        X_filled = X_filled.fillna(X_filled.median())

        # 定数特徴量の除去
        constant_features = X_filled.columns[X_filled.nunique() <= 1].tolist()
        if constant_features:
            logger.info(f"定数特徴量を除去: {len(constant_features)}個")
            X_filled = X_filled.drop(columns=constant_features)

        # 高相関特徴量の除去
        X_filled = self._remove_highly_correlated_features(X_filled)

        return X_filled.values, X_filled.columns.tolist()

    def _remove_highly_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """高相関特徴量を除去"""
        try:
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            to_drop = [
                column
                for column in upper_triangle.columns
                if any(upper_triangle[column] > self.config.correlation_threshold)
            ]

            if to_drop:
                logger.info(f"高相関特徴量を除去: {len(to_drop)}個")
                X = X.drop(columns=to_drop)

        except Exception as e:
            logger.warning(f"相関除去エラー: {e}")

        return X

    def _ensemble_selection(
        self, X: np.ndarray, y: pd.Series, feature_names: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """アンサンブル特徴量選択"""
        logger.info("🔄 アンサンブル特徴量選択を実行")

        method_results = {}
        feature_votes = {name: 0 for name in feature_names}

        # ensemble_methods が None の場合の処理
        ensemble_methods = self.config.ensemble_methods or [
            SelectionMethod.MUTUAL_INFO,
            SelectionMethod.RANDOM_FOREST,
            SelectionMethod.LASSO,
        ]

        # 各手法で特徴量選択を実行
        for method in ensemble_methods:
            try:
                selected_features, result = self._single_method_selection(
                    X, y, feature_names, method
                )
                method_results[method.value] = result

                # 投票
                for feature in selected_features:
                    feature_votes[feature] += 1

                logger.info(f"{method.value}: {len(selected_features)}個選択")

            except Exception as e:
                logger.warning(f"{method.value}でエラー: {e}")
                continue

        # 投票結果に基づいて最終選択
        n_methods = len(ensemble_methods)

        if self.config.ensemble_voting == "unanimous":
            # 全手法で選択された特徴量のみ
            threshold = n_methods
        else:
            # 過半数で選択された特徴量
            threshold = max(1, n_methods // 2)

        selected_features = [
            feature for feature, votes in feature_votes.items() if votes >= threshold
        ]

        # 最小特徴量数の保証
        if len(selected_features) < 5:
            # 投票数順でトップ5を選択
            sorted_features = sorted(
                feature_votes.items(), key=lambda x: x[1], reverse=True
            )
            selected_features = [f[0] for f in sorted_features[:5]]

        results = {
            "method": "ensemble",
            "ensemble_methods": [m.value for m in ensemble_methods],
            "method_results": method_results,
            "feature_votes": feature_votes,
            "voting_threshold": threshold,
            "selected_features": selected_features,
        }

        return selected_features, results

    def _single_method_selection(
        self,
        X: np.ndarray,
        y: pd.Series,
        feature_names: List[str],
        method: SelectionMethod,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """単一手法による特徴量選択"""

        if method == SelectionMethod.UNIVARIATE_F:
            return self._univariate_selection(X, y, feature_names, f_classif)
        elif method == SelectionMethod.UNIVARIATE_CHI2:
            return self._univariate_selection(X, y, feature_names, chi2)
        elif method == SelectionMethod.MUTUAL_INFO:
            return self._mutual_info_selection(X, y, feature_names)
        elif method == SelectionMethod.LASSO:
            return self._lasso_selection(X, y, feature_names)
        elif method == SelectionMethod.RANDOM_FOREST:
            return self._random_forest_selection(X, y, feature_names)
        elif method == SelectionMethod.RFE:
            return self._rfe_selection(X, y, feature_names)
        elif method == SelectionMethod.RFECV:
            return self._rfecv_selection(X, y, feature_names)
        elif method == SelectionMethod.PERMUTATION:
            return self._permutation_selection(X, y, feature_names)
        else:
            raise ValueError(f"未対応の手法: {method}")

    def _univariate_selection(
        self, X: np.ndarray, y: pd.Series, feature_names: List[str], score_func
    ) -> Tuple[List[str], Dict[str, Any]]:
        """単変量統計的選択"""
        try:
            # Chi2の場合は非負値に変換
            if score_func == chi2:
                X_transformed = X - X.min(axis=0) + 1e-8
            else:
                X_transformed = X

            k = self.config.k_features or max(5, int(len(feature_names) * 0.3))
            selector = SelectKBest(score_func=score_func, k=k)
            selector.fit(X_transformed, y)

            selected_mask = selector.get_support()
            # selected_mask が None の場合の処理を追加
            if selected_mask is None:
                selected_mask = np.zeros(len(feature_names), dtype=bool)
                selected_mask[: min(k, len(feature_names))] = True

            selected_features = [
                feature_names[i] for i in range(len(feature_names)) if selected_mask[i]
            ]

            results = {
                "method": score_func.__name__,
                "scores": (
                    selector.scores_.tolist() if hasattr(selector, "scores_") else []
                ),
                "selected_features": selected_features,
                "k": k,
            }

            return selected_features, results

        except Exception as e:
            logger.error(f"単変量選択エラー: {e}")
            # エラー時は最初の5つの特徴量を返す
            fallback_features = (
                feature_names[: min(5, len(feature_names))] if feature_names else []
            )
            return fallback_features, {"error": str(e)}

    def _mutual_info_selection(
        self, X: np.ndarray, y: pd.Series, feature_names: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """相互情報量による選択"""
        try:
            k = self.config.k_features or max(5, int(len(feature_names) * 0.3))
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            selector.fit(X, y)

            selected_mask = selector.get_support()
            # selected_mask が None の場合の処理を追加
            if selected_mask is None:
                selected_mask = np.zeros(len(feature_names), dtype=bool)
                selected_mask[: min(k, len(feature_names))] = True

            selected_features = [
                feature_names[i] for i in range(len(feature_names)) if selected_mask[i]
            ]

            results = {
                "method": "mutual_info",
                "scores": (
                    selector.scores_.tolist() if hasattr(selector, "scores_") else []
                ),
                "selected_features": selected_features,
                "k": k,
            }

            return selected_features, results

        except Exception as e:
            logger.error(f"相互情報量選択エラー: {e}")
            # エラー時は最初の5つの特徴量を返す
            fallback_features = (
                feature_names[: min(5, len(feature_names))] if feature_names else []
            )
            return fallback_features, {"error": str(e)}

    def _lasso_selection(
        self, X: np.ndarray, y: pd.Series, feature_names: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Lasso回帰による選択"""
        try:
            lasso = LassoCV(
                cv=self.config.cv_folds, random_state=self.config.random_state
            )
            lasso.fit(X, y)  # 明示的にfitを実行

            selector = SelectFromModel(
                lasso, threshold=self.config.importance_threshold, prefit=True
            )

            selected_mask = selector.get_support()
            # selected_mask が None の場合の処理を追加
            if selected_mask is None:
                # 係数の絶対値に基づいて選択
                importances = np.abs(lasso.coef_)
                threshold = np.sort(importances)[-min(5, len(importances))]
                selected_mask = importances >= threshold

            selected_features = [
                feature_names[i] for i in range(len(feature_names)) if selected_mask[i]
            ]

            # 最小特徴量数の保証
            if len(selected_features) < 5:
                importances = np.abs(lasso.coef_)
                top_indices = np.argsort(importances)[-min(5, len(importances)) :]
                selected_features = [feature_names[i] for i in top_indices]

            results = {
                "method": "lasso",
                "coefficients": lasso.coef_.tolist(),
                "alpha": lasso.alpha_,
                "selected_features": selected_features,
            }

            return selected_features, results

        except Exception as e:
            logger.error(f"Lasso選択エラー: {e}")
            # エラー時は最初の5つの特徴量を返す
            fallback_features = (
                feature_names[: min(5, len(feature_names))] if feature_names else []
            )
            return fallback_features, {"error": str(e)}

    def _random_forest_selection(
        self, X: np.ndarray, y: pd.Series, feature_names: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """ランダムフォレストによる選択"""
        try:
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )
            rf.fit(X, y)  # 明示的にfitを実行

            selector = SelectFromModel(
                rf, threshold=self.config.importance_threshold, prefit=True
            )

            selected_mask = selector.get_support()
            # selected_mask が None の場合の処理を追加
            if selected_mask is None:
                # 重要度に基づいて選択
                importances = rf.feature_importances_
                threshold = np.sort(importances)[-min(5, len(importances))]
                selected_mask = importances >= threshold

            selected_features = [
                feature_names[i] for i in range(len(feature_names)) if selected_mask[i]
            ]

            # 最小特徴量数の保証
            if len(selected_features) < 5:
                importances = rf.feature_importances_
                top_indices = np.argsort(importances)[-min(5, len(importances)) :]
                selected_features = [feature_names[i] for i in top_indices]

            results = {
                "method": "random_forest",
                "feature_importances": rf.feature_importances_.tolist(),
                "selected_features": selected_features,
            }

            return selected_features, results

        except Exception as e:
            logger.error(f"ランダムフォレスト選択エラー: {e}")
            # エラー時は最初の5つの特徴量を返す
            fallback_features = (
                feature_names[: min(5, len(feature_names))] if feature_names else []
            )
            return fallback_features, {"error": str(e)}

    def _rfe_selection(
        self, X: np.ndarray, y: pd.Series, feature_names: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """再帰的特徴量除去"""
        try:
            estimator = RandomForestClassifier(
                n_estimators=50, random_state=self.config.random_state
            )
            n_features = self.config.k_features or max(5, int(len(feature_names) * 0.3))
            selector = RFE(estimator, n_features_to_select=n_features)
            selector.fit(X, y)

            selected_mask = selector.get_support()
            # selected_mask が None の場合の処理を追加
            if selected_mask is None:
                selected_mask = np.zeros(len(feature_names), dtype=bool)
                selected_mask[: min(n_features, len(feature_names))] = True

            selected_features = [
                feature_names[i] for i in range(len(feature_names)) if selected_mask[i]
            ]

            results = {
                "method": "rfe",
                "ranking": (
                    selector.ranking_.tolist() if hasattr(selector, "ranking_") else []
                ),
                "selected_features": selected_features,
                "n_features": n_features,
            }

            return selected_features, results

        except Exception as e:
            logger.error(f"RFE選択エラー: {e}")
            # エラー時は最初の5つの特徴量を返す
            fallback_features = (
                feature_names[: min(5, len(feature_names))] if feature_names else []
            )
            return fallback_features, {"error": str(e)}

    def _rfecv_selection(
        self, X: np.ndarray, y: pd.Series, feature_names: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """クロスバリデーション付き再帰的特徴量除去"""
        try:
            estimator = RandomForestClassifier(
                n_estimators=50, random_state=self.config.random_state
            )
            selector = RFECV(estimator, cv=self.config.cv_folds)
            selector.fit(X, y)

            selected_mask = selector.get_support()
            # selected_mask が None の場合の処理を追加
            if selected_mask is None:
                # cv_results_ から最適な特徴量数を取得
                if (
                    hasattr(selector, "cv_results_")
                    and "mean_test_score" in selector.cv_results_
                ):
                    n_features = np.argmax(selector.cv_results_["mean_test_score"]) + 1
                else:
                    n_features = min(5, len(feature_names))
                selected_mask = np.zeros(len(feature_names), dtype=bool)
                selected_mask[:n_features] = True

            selected_features = [
                feature_names[i] for i in range(len(feature_names)) if selected_mask[i]
            ]

            results = {
                "method": "rfecv",
                "ranking": (
                    selector.ranking_.tolist() if hasattr(selector, "ranking_") else []
                ),
                "cv_scores": (
                    selector.cv_results_["mean_test_score"].tolist()
                    if hasattr(selector, "cv_results_")
                    and "mean_test_score" in selector.cv_results_
                    else []
                ),
                "optimal_features": (
                    selector.n_features_
                    if hasattr(selector, "n_features_")
                    else len(selected_features)
                ),
                "selected_features": selected_features,
            }

            return selected_features, results

        except Exception as e:
            logger.error(f"RFECV選択エラー: {e}")
            # エラー時は最初の5つの特徴量を返す
            fallback_features = (
                feature_names[: min(5, len(feature_names))] if feature_names else []
            )
            return fallback_features, {"error": str(e)}

    def _permutation_selection(
        self, X: np.ndarray, y: pd.Series, feature_names: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """順列重要度による選択"""
        try:
            estimator = RandomForestClassifier(
                n_estimators=50, random_state=self.config.random_state
            )
            estimator.fit(X, y)

            perm_importance = permutation_importance(
                estimator, X, y, n_repeats=5, random_state=self.config.random_state
            )

            # 重要度の閾値以上の特徴量を選択
            # 安全に属性にアクセスする
            if isinstance(perm_importance, dict):
                importances_mean = perm_importance.get(
                    "importances_mean", np.zeros(len(feature_names))
                )
            else:
                importances_mean = getattr(
                    perm_importance, "importances_mean", np.zeros(len(feature_names))
                )

            # numpy配列に変換
            if not isinstance(importances_mean, np.ndarray):
                importances_mean = np.array(importances_mean)

            important_features = importances_mean > self.config.importance_threshold
            selected_features = [
                feature_names[i]
                for i in range(len(feature_names))
                if i < len(important_features) and important_features[i]
            ]

            # 最小特徴量数の保証
            if len(selected_features) < 5:
                top_indices = np.argsort(importances_mean)[
                    -min(5, len(importances_mean)) :
                ]
                selected_features = [feature_names[i] for i in top_indices]

            # importances_std の処理
            if isinstance(perm_importance, dict):
                importances_std = perm_importance.get("importances_std", [])
            else:
                importances_std = getattr(perm_importance, "importances_std", [])

            # numpy配列に変換
            if not isinstance(importances_std, np.ndarray):
                importances_std = (
                    np.array(importances_std)
                    if len(importances_std) > 0
                    else np.zeros(len(importances_mean))
                )

            results = {
                "method": "permutation",
                "importances_mean": importances_mean.tolist(),
                "importances_std": importances_std.tolist(),
                "selected_features": selected_features,
            }

            return selected_features, results

        except Exception as e:
            logger.error(f"順列重要度選択エラー: {e}")
            # エラー時は最初の5つの特徴量を返す
            fallback_features = (
                feature_names[: min(5, len(feature_names))] if feature_names else []
            )
            return fallback_features, {"error": str(e)}

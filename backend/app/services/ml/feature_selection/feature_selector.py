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

        # 選択された特徴量でDataFrameを作成 (インデックス操作を避ける)
        if selected_features:
            data_dict = {c: X[c].values for c in selected_features if c in X.columns}
            X_selected = pd.DataFrame(data_dict, index=X.index)
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
        # 欠損値・無限値の処理 (プリミティブな処理に限定)
        X_filled = X.replace([np.inf, -np.inf], np.nan)
        X_filled = X_filled.fillna(X_filled.median())

        # 定数特徴量の除去 (インデックス操作を避ける)
        cols = X_filled.columns.tolist()
        nunique = [X_filled[c].nunique() for c in cols]
        keep_cols = [cols[i] for i, n in enumerate(nunique) if n > 1]

        if len(keep_cols) < len(cols):
            logger.info(f"定数特徴量を除去: {len(cols) - len(keep_cols)}個")
            # 辞書から再構築
            data_dict = {c: X_filled[c].values for c in keep_cols}
            X_filled = pd.DataFrame(data_dict, index=X_filled.index)

        # 高相関特徴量の除去
        X_filled = self._remove_highly_correlated_features(X_filled)

        return X_filled.values, X_filled.columns.tolist()

    def _remove_highly_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """高相関特徴量を除去"""
        try:
            # 相関行列の取得
            corr_matrix = X.corr().abs()
            cols = corr_matrix.columns.tolist()
            drop_cols = []

            # 2重ループでの相関チェック (ilocを使わずvaluesで高速化)
            corr_values = corr_matrix.values
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    if corr_values[i, j] > self.config.correlation_threshold:
                        col_to_drop = cols[j]
                        if col_to_drop not in drop_cols:
                            drop_cols.append(col_to_drop)

            if drop_cols:
                logger.info(f"高相関特徴量を除去: {len(drop_cols)}個")
                # 必要なカラムのみで再構築
                keep_cols = [c for c in cols if c not in drop_cols]
                data_dict = {c: X[c].values for c in keep_cols}
                X = pd.DataFrame(data_dict, index=X.index)

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

    def _mask_to_features(
        self,
        mask: Optional[np.ndarray],
        scores: np.ndarray,
        feature_names: List[str],
        k: int = 5,
    ) -> List[str]:
        """マスクまたはスコアから特徴量を選択（共通処理）"""
        if mask is not None and mask.any():
            selected = [feature_names[i] for i, m in enumerate(mask) if m]
            if len(selected) >= k:
                return selected

        # マスクが無効または数が足りない場合はスコア上位を選択
        top_idx = np.argsort(np.abs(scores))[-min(k, len(scores)) :]
        return [feature_names[i] for i in top_idx]

    def _single_method_selection(
        self,
        X: np.ndarray,
        y: pd.Series,
        feature_names: List[str],
        method: SelectionMethod,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """単一手法による特徴量選択"""
        try:
            k_def = self.config.k_features or max(5, int(len(feature_names) * 0.3))

            if method == SelectionMethod.UNIVARIATE_F:
                sel = SelectKBest(f_classif, k=k_def).fit(X, y)
                feats = self._mask_to_features(
                    sel.get_support(), sel.scores_, feature_names, k=k_def
                )
                return feats, {
                    "method": "f_classif",
                    "scores": sel.scores_.tolist(),
                    "selected_features": feats,
                }

            if method == SelectionMethod.UNIVARIATE_CHI2:
                X_pos = X - X.min(axis=0) + 1e-8
                sel = SelectKBest(chi2, k=k_def).fit(X_pos, y)
                feats = self._mask_to_features(
                    sel.get_support(), sel.scores_, feature_names, k=k_def
                )
                return feats, {
                    "method": "chi2",
                    "scores": sel.scores_.tolist(),
                    "selected_features": feats,
                }

            if method == SelectionMethod.MUTUAL_INFO:
                sel = SelectKBest(mutual_info_classif, k=k_def).fit(X, y)
                feats = self._mask_to_features(
                    sel.get_support(), sel.scores_, feature_names, k=k_def
                )
                return feats, {
                    "method": "mutual_info",
                    "scores": sel.scores_.tolist(),
                    "selected_features": feats,
                }

            if method == SelectionMethod.LASSO:
                model = LassoCV(
                    cv=self.config.cv_folds, random_state=self.config.random_state
                ).fit(X, y)
                sel = SelectFromModel(
                    model, threshold=self.config.importance_threshold, prefit=True
                )
                feats = self._mask_to_features(
                    sel.get_support(), model.coef_, feature_names, k=k_def
                )
                return feats, {
                    "method": "lasso",
                    "coefficients": model.coef_.tolist(),
                    "selected_features": feats,
                }

            if method == SelectionMethod.RANDOM_FOREST:
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                ).fit(X, y)
                sel = SelectFromModel(
                    model, threshold=self.config.importance_threshold, prefit=True
                )
                feats = self._mask_to_features(
                    sel.get_support(),
                    model.feature_importances_,
                    feature_names,
                    k=k_def,
                )
                return feats, {
                    "method": "random_forest",
                    "importances": model.feature_importances_.tolist(),
                    "selected_features": feats,
                }

            if method == SelectionMethod.RFE:
                est = RandomForestClassifier(
                    n_estimators=50, random_state=self.config.random_state
                )
                sel = RFE(est, n_features_to_select=k_def).fit(X, y)
                feats = self._mask_to_features(
                    sel.get_support(), -sel.ranking_, feature_names, k=k_def
                )
                return feats, {
                    "method": "rfe",
                    "ranking": sel.ranking_.tolist(),
                    "selected_features": feats,
                }

            if method == SelectionMethod.RFECV:
                est = RandomForestClassifier(
                    n_estimators=50, random_state=self.config.random_state
                )
                sel = RFECV(est, cv=self.config.cv_folds).fit(X, y)
                feats = self._mask_to_features(
                    sel.get_support(),
                    sel.support_.astype(float),
                    feature_names,
                    k=k_def,
                )
                return feats, {
                    "method": "rfecv",
                    "n_features": int(sel.n_features_),
                    "selected_features": feats,
                }

            if method == SelectionMethod.PERMUTATION:
                est = RandomForestClassifier(
                    n_estimators=50, random_state=self.config.random_state
                ).fit(X, y)
                imp = permutation_importance(
                    est, X, y, n_repeats=5, random_state=self.config.random_state
                )
                feats = self._mask_to_features(
                    imp.importances_mean > self.config.importance_threshold,
                    imp.importances_mean,
                    feature_names,
                    k=k_def,
                )
                return feats, {
                    "method": "permutation",
                    "importances": imp.importances_mean.tolist(),
                    "selected_features": feats,
                }

            raise ValueError(f"未対応の手法: {method}")

        except Exception as e:
            method_name = method.value if hasattr(method, "value") else str(method)
            logger.error(f"{method_name}選択エラー: {e}")
            return feature_names[:5], {"error": str(e)}

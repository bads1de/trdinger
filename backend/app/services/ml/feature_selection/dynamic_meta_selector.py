import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Union
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from lightgbm import LGBMClassifier

logger = logging.getLogger(__name__)


class DynamicMetaSelector(BaseEstimator, SelectorMixin):
    """
    自律型動的特徴量選択器 (DynamicMetaSelector)

    1. 特徴量間の相関に基づく動的クラスタリング
    2. クラスタ内代表者の選定
    3. シャドウ特徴量を用いた有意性テスト
    """

    def __init__(
        self,
        clustering_threshold: float = 0.8,
        min_features: int = 5,
        n_shadow_iterations: int = 5,
        random_state: int = 42,
        **kwargs,  # 予期しない引数を無視するために追加
    ):
        self.clustering_threshold = clustering_threshold
        self.min_features = min_features
        self.n_shadow_iterations = n_shadow_iterations
        self.random_state = random_state
        self.support_mask_ = None
        self.feature_names_in_ = None
        self.selected_features_ = None

    def _cluster_features(self, X: pd.DataFrame) -> Dict[int, List[str]]:
        # ... (既存のコードと同じ)
        corr = X.corr().fillna(0)
        dist = 1 - np.abs(corr.values)
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        condensed_dist = squareform(dist)
        linkage_matrix = linkage(condensed_dist, method="ward")
        cluster_labels = fcluster(
            linkage_matrix, 1 - self.clustering_threshold, criterion="distance"
        )
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(X.columns[i])
        return clusters

    def _shadow_filtering(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """シャドウ特徴量を用いてノイズ特徴量を排除する"""
        n_features = X.shape[1]
        hit_counts = np.zeros(n_features)

        # モデル初期化 (importance_type='gain' を指定して、より本質的な寄与度を測る)
        model = LGBMClassifier(
            n_estimators=50,
            learning_rate=0.1,
            num_leaves=15,
            importance_type="gain",  # 'split' から 'gain' に変更
            random_state=self.random_state,
            verbosity=-1,
            force_col_wise=True,
        )

        rng = np.random.RandomState(self.random_state)

        for i in range(self.n_shadow_iterations):
            # シャドウ特徴量の生成（各列を独立にシャッフル）
            X_shadow = X.copy().values
            for col in range(n_features):
                rng.shuffle(X_shadow[:, col])

            # 本物とシャドウを結合
            X_combined = np.hstack([X.values, X_shadow])

            # 重要度の計算
            model.fit(X_combined, y)
            importances = model.feature_importances_

            real_imp = importances[:n_features]
            shadow_imp = importances[n_features:]

            # シャドウの最大重要度を閾値にする
            shadow_max = np.max(shadow_imp)
            hit_counts[real_imp > shadow_max] += 1

        # 半数以上の試行でシャドウに勝ったもののみ選択
        support_mask = hit_counts >= (self.n_shadow_iterations / 2)

        # 全て落とされた場合のセーフティ
        if not support_mask.any():
            logger.warning("All features failed shadow test. Falling back to top 5.")
            # 単純な重要度順でTop 5を返す
            model.fit(X.values, y)
            top_indices = np.argsort(model.feature_importances_)[-self.min_features :]
            support_mask = np.zeros(n_features, dtype=bool)
            support_mask[top_indices] = True

        return support_mask

    def _get_dynamic_k(self, n_samples: int) -> int:
        """サンプル数に応じて最大特徴量数を動的に決定する"""
        # ヒューリスティック: sqrt(N) * 1.5 〜 2
        k = int(np.sqrt(n_samples) * 1.5)
        # 上限と下限の制約
        return max(self.min_features, min(k, 30))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DynamicMetaSelector":
        """自律的に最適な特徴量セットを選択する"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = X.columns.tolist()
        n_samples = len(X)
        target_k = self._get_dynamic_k(n_samples)

        logger.info(
            f"DynamicMetaSelector starting: Samples={n_samples}, Max-K={target_k}"
        )

        # 1. 階層的クラスタリングで情報の重複を検知
        clusters = self._cluster_features(X)

        # 2. 各クラスタから代表者を選定 (ターゲットへの相互情報量が最大のもの)
        from sklearn.feature_selection import mutual_info_classif

        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        mi_series = pd.Series(mi_scores, index=X.columns)

        representative_features = []
        for cluster_id, features in clusters.items():
            # primary_proba が含まれる場合は、無条件でそれを代表にする
            if "primary_proba" in features:
                representative_features.append("primary_proba")
            else:
                # それ以外は相互情報量が最大のものを代表にする
                best_feature = mi_series.loc[features].idxmax()
                representative_features.append(best_feature)

        X_reduced = X[representative_features]

        # 3. シャドウ特徴量によるノイズ除去
        support_mask_reduced = self._shadow_filtering(X_reduced, y)
        final_candidates = X_reduced.columns[support_mask_reduced].tolist()

        # primary_proba が漏れていたら強制的に戻す
        if (
            "primary_proba" in self.feature_names_in_
            and "primary_proba" not in final_candidates
        ):
            final_candidates.append("primary_proba")

        # 4. 最終的な Top-K 選定
        # 残った候補の中で重要度（または相互情報量）が高いものを target_k 個選ぶ
        if len(final_candidates) > target_k:
            final_candidates = (
                mi_series.loc[final_candidates]
                .sort_values(ascending=False)
                .head(target_k)
                .index.tolist()
            )

        self.selected_features_ = final_candidates

        # support_mask_ の作成
        self.support_mask_ = np.array(
            [f in final_candidates for f in self.feature_names_in_]
        )

        logger.info(
            f"DynamicMetaSelector complete: Selected {len(final_candidates)} features."
        )
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """選択された特徴量を抽出する"""
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]

        # NumPy配列の場合はマスクを適用
        return X[:, self.support_mask_]

    def _get_support_mask(self):
        return self.support_mask_

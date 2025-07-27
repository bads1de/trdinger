"""
高度特徴量選択システム

統計的検定、相関分析、重要度ベース選択を組み合わせた
高度な特徴量選択を実装します。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.ensemble import RandomForestRegressor
import warnings

logger = logging.getLogger(__name__)


class AdvancedFeatureSelector:
    """高度特徴量選択クラス"""

    def __init__(self):
        """初期化"""
        self.selection_history = []
        self.feature_scores = {}
        self.correlation_matrix = None

    def select_features_comprehensive(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        max_features: int = 100,
        selection_methods: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        包括的特徴量選択

        Args:
            features: 特徴量DataFrame
            target: ターゲット変数
            max_features: 最大特徴量数
            selection_methods: 選択手法のリスト

        Returns:
            選択された特徴量とメタデータ
        """
        if selection_methods is None:
            selection_methods = [
                "statistical_test",
                "correlation_filter",
                "mutual_information",
                "importance_based",
            ]

        logger.info(
            f"包括的特徴量選択を開始: {len(features.columns)}個 → {max_features}個"
        )

        selection_info = {
            "original_count": len(features.columns),
            "target_count": max_features,
            "methods_used": selection_methods,
            "selection_steps": [],
        }

        current_features = features.copy()

        # 1. 統計的検定による選択
        if "statistical_test" in selection_methods:
            current_features, step_info = self._statistical_test_selection(
                current_features, target, max_features * 2
            )
            selection_info["selection_steps"].append(step_info)

        # 2. 相関フィルタリング
        if "correlation_filter" in selection_methods:
            current_features, step_info = self._correlation_filter_selection(
                current_features, target, correlation_threshold=0.95
            )
            selection_info["selection_steps"].append(step_info)

        # 3. 相互情報量による選択
        if "mutual_information" in selection_methods:
            current_features, step_info = self._mutual_information_selection(
                current_features, target, max_features * 1.5
            )
            selection_info["selection_steps"].append(step_info)

        # 4. 重要度ベース選択（最終選択）
        if "importance_based" in selection_methods:
            current_features, step_info = self._importance_based_selection(
                current_features, target, max_features
            )
            selection_info["selection_steps"].append(step_info)

        selection_info["final_count"] = len(current_features.columns)
        selection_info["reduction_ratio"] = (
            1 - selection_info["final_count"] / selection_info["original_count"]
        )

        # 選択履歴を保存
        self.selection_history.append(selection_info)

        logger.info(
            f"特徴量選択完了: {selection_info['original_count']}個 → "
            f"{selection_info['final_count']}個 "
            f"(削減率: {selection_info['reduction_ratio']:.2%})"
        )

        return current_features, selection_info

    def _statistical_test_selection(
        self, features: pd.DataFrame, target: pd.Series, max_features: int
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """統計的検定による特徴量選択"""
        try:
            logger.info("統計的検定による特徴量選択を実行中...")

            # インデックスを統一してから有効なデータのマスクを作成
            features_reset = features.reset_index(drop=True)
            target_reset = target.reset_index(drop=True)

            valid_mask = target_reset.notna() & features_reset.notna().all(axis=1)
            valid_features = features_reset.loc[valid_mask]
            valid_target = target_reset.loc[valid_mask]

            if len(valid_features) == 0:
                logger.warning("有効なデータが見つかりません")
                return features, {
                    "method": "statistical_test",
                    "selected": 0,
                    "error": "no_valid_data",
                }

            # F統計量とp値を計算
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f_scores, p_values = f_regression(valid_features, valid_target)

            # p値でソートして上位を選択
            feature_scores = pd.DataFrame(
                {
                    "feature": valid_features.columns,
                    "f_score": f_scores,
                    "p_value": p_values,
                }
            )

            # p値が有効な特徴量のみを選択
            valid_scores = feature_scores[
                (feature_scores["p_value"].notna())
                & (feature_scores["p_value"] < 0.05)  # 5%有意水準
            ].sort_values("f_score", ascending=False)

            # 上位特徴量を選択
            selected_count = min(len(valid_scores), max_features)
            selected_features = valid_scores.head(selected_count)["feature"].tolist()

            result_features = features[selected_features]

            step_info = {
                "method": "statistical_test",
                "input_count": len(features.columns),
                "selected": len(selected_features),
                "significance_threshold": 0.05,
                "mean_f_score": (
                    float(valid_scores["f_score"].mean())
                    if len(valid_scores) > 0
                    else 0
                ),
            }

            logger.info(f"統計的検定選択完了: {len(selected_features)}個選択")
            return result_features, step_info

        except Exception as e:
            logger.error(f"統計的検定選択エラー: {e}")
            return features, {
                "method": "statistical_test",
                "selected": 0,
                "error": str(e),
            }

    def _correlation_filter_selection(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        correlation_threshold: float = 0.95,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """相関フィルタリングによる特徴量選択"""
        try:
            logger.info(f"相関フィルタリングを実行中 (閾値: {correlation_threshold})")

            # 相関行列を計算
            correlation_matrix = features.corr().abs()
            self.correlation_matrix = correlation_matrix

            # 高相関ペアを特定
            high_corr_pairs = []
            removed_features = set()

            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > correlation_threshold:
                        feature1 = correlation_matrix.columns[i]
                        feature2 = correlation_matrix.columns[j]
                        high_corr_pairs.append(
                            (feature1, feature2, correlation_matrix.iloc[i, j])
                        )

            # ターゲットとの相関を考慮して除去する特徴量を決定
            target_corr = features.corrwith(target).abs()

            for feature1, feature2, corr_value in high_corr_pairs:
                if (
                    feature1 not in removed_features
                    and feature2 not in removed_features
                ):
                    # ターゲットとの相関が低い方を除去
                    if target_corr.get(feature1, 0) < target_corr.get(feature2, 0):
                        removed_features.add(feature1)
                    else:
                        removed_features.add(feature2)

            # 除去対象以外の特徴量を選択
            selected_features = [
                col for col in features.columns if col not in removed_features
            ]
            result_features = features[selected_features]

            step_info = {
                "method": "correlation_filter",
                "input_count": len(features.columns),
                "selected": len(selected_features),
                "removed": len(removed_features),
                "correlation_threshold": correlation_threshold,
                "high_corr_pairs": len(high_corr_pairs),
            }

            logger.info(f"相関フィルタリング完了: {len(removed_features)}個除去")
            return result_features, step_info

        except Exception as e:
            logger.error(f"相関フィルタリングエラー: {e}")
            return features, {
                "method": "correlation_filter",
                "selected": 0,
                "error": str(e),
            }

    def _mutual_information_selection(
        self, features: pd.DataFrame, target: pd.Series, max_features: int
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """相互情報量による特徴量選択"""
        try:
            logger.info("相互情報量による特徴量選択を実行中...")

            # インデックスを統一してから有効なデータのマスクを作成
            features_reset = features.reset_index(drop=True)
            target_reset = target.reset_index(drop=True)

            valid_mask = target_reset.notna() & features_reset.notna().all(axis=1)
            valid_features = features_reset.loc[valid_mask]
            valid_target = target_reset.loc[valid_mask]

            if len(valid_features) == 0:
                logger.warning("有効なデータが見つかりません")
                return features, {
                    "method": "mutual_information",
                    "selected": 0,
                    "error": "no_valid_data",
                }

            # 相互情報量を計算
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mi_scores = mutual_info_regression(
                    valid_features, valid_target, random_state=42
                )

            # スコアでソートして上位を選択
            feature_scores = pd.DataFrame(
                {"feature": valid_features.columns, "mi_score": mi_scores}
            ).sort_values("mi_score", ascending=False)

            # 上位特徴量を選択
            selected_count = min(len(feature_scores), int(max_features))
            selected_features = feature_scores.head(selected_count)["feature"].tolist()

            result_features = features[selected_features]

            step_info = {
                "method": "mutual_information",
                "input_count": len(features.columns),
                "selected": len(selected_features),
                "mean_mi_score": float(feature_scores["mi_score"].mean()),
            }

            logger.info(f"相互情報量選択完了: {len(selected_features)}個選択")
            return result_features, step_info

        except Exception as e:
            logger.error(f"相互情報量選択エラー: {e}")
            return features, {
                "method": "mutual_information",
                "selected": 0,
                "error": str(e),
            }

    def _importance_based_selection(
        self, features: pd.DataFrame, target: pd.Series, max_features: int
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """重要度ベース特徴量選択"""
        try:
            logger.info("重要度ベース特徴量選択を実行中...")

            # インデックスを統一してから有効なデータのマスクを作成
            features_reset = features.reset_index(drop=True)
            target_reset = target.reset_index(drop=True)

            valid_mask = target_reset.notna() & features_reset.notna().all(axis=1)
            valid_features = features_reset.loc[valid_mask]
            valid_target = target_reset.loc[valid_mask]

            if len(valid_features) == 0:
                logger.warning("有効なデータが見つかりません")
                return features, {
                    "method": "importance_based",
                    "selected": 0,
                    "error": "no_valid_data",
                }

            # RandomForestで重要度を計算
            rf = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=1,  # 安定性のため
                max_depth=10,  # 過学習防止
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rf.fit(valid_features, valid_target)

            # 重要度でソートして上位を選択
            feature_importance = pd.DataFrame(
                {
                    "feature": valid_features.columns,
                    "importance": rf.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            # 上位特徴量を選択
            selected_count = min(len(feature_importance), max_features)
            selected_features = feature_importance.head(selected_count)[
                "feature"
            ].tolist()

            result_features = features[selected_features]

            # 重要度スコアを保存
            self.feature_scores = dict(
                zip(feature_importance["feature"], feature_importance["importance"])
            )

            step_info = {
                "method": "importance_based",
                "input_count": len(features.columns),
                "selected": len(selected_features),
                "mean_importance": float(feature_importance["importance"].mean()),
                "rf_score": float(rf.score(valid_features, valid_target)),
            }

            logger.info(f"重要度ベース選択完了: {len(selected_features)}個選択")
            return result_features, step_info

        except Exception as e:
            logger.error(f"重要度ベース選択エラー: {e}")
            return features, {
                "method": "importance_based",
                "selected": 0,
                "error": str(e),
            }

    def get_selection_summary(self) -> Dict[str, Any]:
        """選択履歴のサマリーを取得"""
        if not self.selection_history:
            return {"message": "選択履歴がありません"}

        latest = self.selection_history[-1]
        return {
            "total_selections": len(self.selection_history),
            "latest_selection": latest,
            "average_reduction_ratio": np.mean(
                [h.get("reduction_ratio", 0) for h in self.selection_history]
            ),
        }

    def get_feature_scores(self) -> Dict[str, float]:
        """特徴量スコアを取得"""
        return self.feature_scores.copy()

    def clear_history(self):
        """選択履歴をクリア"""
        self.selection_history.clear()
        self.feature_scores.clear()
        self.correlation_matrix = None
        logger.debug("特徴量選択履歴をクリアしました")

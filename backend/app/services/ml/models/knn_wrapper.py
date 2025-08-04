"""
K-Nearest Neighborsモデルラッパー

アンサンブル学習で使用するKNNモデルのラッパークラスを提供します。
scikit-learnのKNeighborsClassifierを使用してアンサンブル専用に最適化されたモデルです。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class KNNModel:
    """
    アンサンブル内で使用するKNNモデルラッパー

    scikit-learnのKNeighborsClassifierを使用してアンサンブル専用に最適化されたモデル
    """

    # アルゴリズム名（AlgorithmRegistryから取得）
    ALGORITHM_NAME = "knn"

    def __init__(self, automl_config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            automl_config: AutoML設定（現在は未使用）
        """
        self.model = None
        self.is_trained = False
        self.feature_columns = None
        self.automl_config = automl_config

        # デフォルトパラメータ
        self.default_params = {
            "n_neighbors": 5,
            "weights": "distance",
            "algorithm": "auto",
            "metric": "minkowski",
            "p": 2,  # ユークリッド距離
            "n_jobs": -1,
        }

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **training_params,
    ) -> Dict[str, Any]:
        """
        KNNモデルを学習

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット

        Returns:
            学習結果の辞書
        """
        try:
            logger.info("🔍 KNNモデルの学習を開始")

            # 特徴量カラムを保存
            self.feature_columns = list(X_train.columns)

            # モデル初期化
            self.model = KNeighborsClassifier(**self.default_params)

            # 学習実行（KNNは遅延学習なので実際はデータを保存するだけ）
            self.model.fit(X_train, y_train)

            # 予測
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)

            # 確率予測（KNNは確率予測をサポート）
            y_pred_proba_train = self.model.predict_proba(X_train)
            y_pred_proba_test = self.model.predict_proba(X_test)

            # 統一された評価指標計算器を使用
            from ..evaluation.enhanced_metrics import (
                EnhancedMetricsCalculator,
                MetricsConfig,
            )

            config = MetricsConfig(
                include_balanced_accuracy=True,
                include_pr_auc=True,
                include_roc_auc=True,
                include_confusion_matrix=True,
                include_classification_report=True,
                average_method="weighted",
                zero_division=0,
            )

            metrics_calculator = EnhancedMetricsCalculator(config)

            # 包括的な評価指標を計算（テストデータ）
            test_metrics = metrics_calculator.calculate_comprehensive_metrics(
                y_test, y_pred_test, y_pred_proba_test
            )

            # 包括的な評価指標を計算（学習データ）
            train_metrics = metrics_calculator.calculate_comprehensive_metrics(
                y_train, y_pred_train, y_pred_proba_train
            )

            # 特徴量重要度（KNNでは直接的な重要度なし）
            # 各特徴量の分散を重要度として近似
            feature_variance = X_train.var()
            total_variance = feature_variance.sum()
            feature_importance = {}
            if total_variance > 0:
                feature_importance = dict(
                    zip(self.feature_columns, feature_variance / total_variance)
                )

            self.is_trained = True

            results = {
                "algorithm": self.ALGORITHM_NAME,
                "n_neighbors": self.model.n_neighbors,
                "weights": self.model.weights,
                "algorithm_type": self.model.algorithm,
                "metric": self.model.metric,
                "feature_count": len(self.feature_columns),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "num_classes": len(self.model.classes_) if hasattr(self.model, "classes_") else 0,
                "feature_importance": feature_importance,
            }

            # テストデータの評価指標を追加（プレフィックス付き）
            for key, value in test_metrics.items():
                if key not in ["error"]:  # エラー情報は除外
                    results[f"test_{key}"] = value
                    # フロントエンド用の統一キー（test_なしのキー）
                    if key in [
                        "accuracy",
                        "balanced_accuracy",
                        "f1_score",
                        "matthews_corrcoef",
                    ]:
                        results[key] = value
                    elif key == "roc_auc" or key == "roc_auc_ovr":
                        results["auc_roc"] = value
                    elif key == "pr_auc" or key == "pr_auc_macro":
                        results["auc_pr"] = value

            # 学習データの評価指標を追加（プレフィックス付き）
            for key, value in train_metrics.items():
                if key not in ["error"]:  # エラー情報は除外
                    results[f"train_{key}"] = value

            logger.info(
                f"✅ KNN学習完了 - テスト精度: {test_metrics.get('accuracy', 0.0):.4f}"
            )
            return results

        except Exception as e:
            logger.error(f"❌ KNN学習エラー: {e}")
            raise UnifiedModelError(f"KNN学習に失敗しました: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測結果
        """
        if not self.is_trained or self.model is None:
            raise UnifiedModelError("モデルが学習されていません")

        try:
            # 特徴量の順序を確認
            if self.feature_columns:
                X = X[self.feature_columns]

            predictions = self.model.predict(X)
            return predictions

        except Exception as e:
            logger.error(f"KNN予測エラー: {e}")
            raise UnifiedModelError(f"予測に失敗しました: {e}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測確率を取得

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率の配列
        """
        if not self.is_trained or self.model is None:
            raise UnifiedModelError("モデルが学習されていません")

        try:
            # 特徴量の順序を確認
            if self.feature_columns:
                X = X[self.feature_columns]

            probabilities = self.model.predict_proba(X)
            return probabilities

        except Exception as e:
            logger.error(f"KNN確率予測エラー: {e}")
            raise UnifiedModelError(f"確率予測に失敗しました: {e}")

    @property
    def feature_columns(self) -> List[str]:
        """特徴量カラム名のリストを取得"""
        return self._feature_columns

    @feature_columns.setter
    def feature_columns(self, columns: List[str]):
        """特徴量カラム名のリストを設定"""
        self._feature_columns = columns

    def get_feature_importance(self) -> Dict[str, float]:
        """
        特徴量重要度を取得（分散ベース）

        Returns:
            特徴量重要度の辞書
        """
        if not self.is_trained or self.model is None:
            raise UnifiedModelError("モデルが学習されていません")

        if not self.feature_columns:
            raise UnifiedModelError("特徴量カラムが設定されていません")

        # 学習データから分散を計算（近似的重要度）
        if hasattr(self.model, "_fit_X"):
            X_train = pd.DataFrame(self.model._fit_X, columns=self.feature_columns)
            feature_variance = X_train.var()
            total_variance = feature_variance.sum()
            if total_variance > 0:
                return dict(
                    zip(self.feature_columns, feature_variance / total_variance)
                )

        return {}

    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得

        Returns:
            モデル情報の辞書
        """
        if not self.is_trained or self.model is None:
            return {"status": "not_trained"}

        return {
            "algorithm": self.ALGORITHM_NAME,
            "n_neighbors": self.model.n_neighbors,
            "weights": self.model.weights,
            "algorithm_type": self.model.algorithm,
            "metric": self.model.metric,
            "p": self.model.p,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "status": "trained",
        }

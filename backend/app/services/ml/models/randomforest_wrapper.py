"""
RandomForestモデルラッパー

アンサンブル学習で使用するRandomForestモデルのラッパークラスを提供します。
scikit-learnのRandomForestClassifierを使用してアンサンブル専用に最適化されたモデルです。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ....utils.unified_error_handler import UnifiedModelError
from ...evaluation.enhanced_metrics import (
    EnhancedMetricsCalculator,
    MetricsConfig,
)

logger = logging.getLogger(__name__)


class RandomForestModel:
    """
    アンサンブル内で使用するRandomForestモデルラッパー

    scikit-learnのRandomForestClassifierを使用してアンサンブル専用に最適化されたモデル
    """
    
    # アルゴリズム名（AlgorithmRegistryから取得）
    ALGORITHM_NAME = "randomforest"

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
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42,
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
        RandomForestモデルを学習

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット

        Returns:
            学習結果の辞書
        """
        try:
            logger.info("🌳 RandomForestモデルの学習を開始")

            # 特徴量カラムを保存
            self.feature_columns = list(X_train.columns)

            # モデル初期化
            self.model = RandomForestClassifier(**self.default_params)

            # 学習実行
            self.model.fit(X_train, y_train)

            # 予測
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)

            # 確率予測（RandomForestは確率予測をサポート）
            y_pred_proba_train = self.model.predict_proba(X_train)
            y_pred_proba_test = self.model.predict_proba(X_test)

            # 評価指標計算
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
            
            train_metrics = metrics_calculator.calculate_comprehensive_metrics(
                y_train, y_pred_train, y_pred_proba_train
            )
            test_metrics = metrics_calculator.calculate_comprehensive_metrics(
                y_test, y_pred_test, y_pred_proba_test
            )

            # 特徴量重要度
            feature_importance = dict(
                zip(self.feature_columns, self.model.feature_importances_)
            )

            self.is_trained = True
            
            n_classes = len(np.unique(y_train))

            results = {
                "algorithm": self.ALGORITHM_NAME,
                "accuracy": test_metrics.get("accuracy", 0.0),
                "balanced_accuracy": test_metrics.get("balanced_accuracy", 0.0),
                "f1_score": test_metrics.get("f1_score", 0.0),
                "matthews_corrcoef": test_metrics.get("matthews_corrcoef", 0.0),
                "roc_auc": test_metrics.get("roc_auc", 0.0),
                "pr_auc": test_metrics.get("pr_auc", 0.0),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "feature_importance": feature_importance,
                "n_estimators": self.model.n_estimators,
                "max_depth": self.model.max_depth,
                "feature_count": len(self.feature_columns),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "num_classes": n_classes,
            }

            logger.info(f"✅ RandomForest学習完了 - テスト精度: {results['accuracy']:.4f}")
            return results

        except Exception as e:
            logger.error(f"❌ RandomForest学習エラー: {e}")
            raise UnifiedModelError(f"RandomForest学習に失敗しました: {e}")

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
            logger.error(f"RandomForest予測エラー: {e}")
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
            logger.error(f"RandomForest確率予測エラー: {e}")
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
        特徴量重要度を取得

        Returns:
            特徴量重要度の辞書
        """
        if not self.is_trained or self.model is None:
            raise UnifiedModelError("モデルが学習されていません")

        if not self.feature_columns:
            raise UnifiedModelError("特徴量カラムが設定されていません")

        return dict(zip(self.feature_columns, self.model.feature_importances_))

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
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "min_samples_split": self.model.min_samples_split,
            "min_samples_leaf": self.model.min_samples_leaf,
            "max_features": self.model.max_features,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "status": "trained",
        }

"""
GradientBoostingモデルラッパー

アンサンブル学習で使用するGradientBoostingモデルのラッパークラスを提供します。
scikit-learnのGradientBoostingClassifierを使用してアンサンブル専用に最適化されたモデルです。
"""

import logging
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from ....utils.error_handler import ModelError

logger = logging.getLogger(__name__)


class GradientBoostingModel:
    """
    アンサンブル内で使用するGradientBoostingモデルラッパー

    scikit-learnのGradientBoostingClassifierを使用してアンサンブル専用に最適化されたモデル
    """

    # アルゴリズム名（AlgorithmRegistryから取得）
    ALGORITHM_NAME = "gradientboosting"

    def __init__(self, automl_config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            automl_config: AutoML設定（現在は未使用）
        """
        self.model: Optional[GradientBoostingClassifier] = None
        self.is_trained = False
        self._feature_columns: Optional[List[str]] = None
        self.scaler = None
        self.automl_config = automl_config

        # デフォルトパラメータ
        self.default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
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
        GradientBoostingモデルを学習

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット

        Returns:
            学習結果の辞書
        """
        try:
            logger.info("📈 GradientBoostingモデルの学習を開始")

            # 特徴量カラムを保存
            self.feature_columns = list(X_train.columns)

            # モデル初期化
            self.model = GradientBoostingClassifier(**self.default_params)

            # 学習実行
            self.model.fit(X_train, y_train)

            # モデルが正常に設定されていることを確認
            assert self.model is not None, "Model should be initialized after fit"
            model: GradientBoostingClassifier = self.model

            # 予測
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # 確率予測（GradientBoostingは確率予測をサポート）
            y_pred_proba_train = model.predict_proba(X_train)
            y_pred_proba_test = model.predict_proba(X_test)

            # 共通の評価関数を使用
            from ..common.evaluation_utils import evaluate_model_predictions

            # 包括的な評価指標を計算（テストデータ）
            test_metrics = evaluate_model_predictions(
                y_test,
                cast(np.ndarray, y_pred_test),
                cast(np.ndarray, y_pred_proba_test),
            )

            # 包括的な評価指標を計算（学習データ）
            train_metrics = evaluate_model_predictions(
                y_train,
                cast(np.ndarray, y_pred_train),
                cast(np.ndarray, y_pred_proba_train),
            )

            # クラス数を取得
            n_classes = len(np.unique(y_train))

            # 特徴量重要度
            feature_importance = dict(
                zip(self.feature_columns, model.feature_importances_)
            )

            self.is_trained = True

            results = {
                "algorithm": self.ALGORITHM_NAME,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "feature_importance": feature_importance,
                "n_estimators": self.model.n_estimators,
                "max_depth": self.model.max_depth,
                "learning_rate": self.model.learning_rate,
                "best_iteration": getattr(
                    self.model, "n_estimators_", self.model.n_estimators
                ),
                "feature_count": len(self.feature_columns),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "num_classes": n_classes,
                # フロントエンド用の統一キー
                "accuracy": test_metrics.get("accuracy", 0.0),
                "balanced_accuracy": test_metrics.get("balanced_accuracy", 0.0),
                "f1_score": test_metrics.get("f1_score", 0.0),
                "matthews_corrcoef": test_metrics.get("matthews_corrcoef", 0.0),
                "auc_roc": test_metrics.get("roc_auc", 0.0),
                "auc_pr": test_metrics.get("pr_auc", 0.0),
            }

            logger.info(
                f"✅ GradientBoosting学習完了 - テスト精度: {test_metrics.get('accuracy', 0.0):.4f}"
            )
            return results

        except Exception as e:
            logger.error(f"❌ GradientBoosting学習エラー: {e}")
            raise ModelError(f"GradientBoosting学習に失敗しました: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測結果
        """
        if not self.is_trained or self.model is None:
            raise ModelError("モデルが学習されていません")

        try:
            # 特徴量の順序を確認
            if self.feature_columns:
                X = pd.DataFrame(X[self.feature_columns])

            predictions = self.model.predict(X)
            return predictions  # type: ignore

        except Exception as e:
            logger.error(f"GradientBoosting予測エラー: {e}")
            raise ModelError(f"予測に失敗しました: {e}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測確率を取得

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率の配列
        """
        if not self.is_trained or self.model is None:
            raise ModelError("モデルが学習されていません")

        try:
            # 特徴量の順序を確認
            if self.feature_columns:
                X = pd.DataFrame(X[self.feature_columns])

            probabilities = self.model.predict_proba(X)
            return probabilities  # type: ignore

        except Exception as e:
            logger.error(f"GradientBoosting確率予測エラー: {e}")
            raise ModelError(f"確率予測に失敗しました: {e}")

    @property
    def feature_columns(self) -> Optional[List[str]]:
        """特徴量カラム名のリストを取得"""
        return self._feature_columns

    @feature_columns.setter
    def feature_columns(self, columns: Optional[List[str]]):
        """特徴量カラム名のリストを設定"""
        self._feature_columns = columns

    def get_feature_importance(self) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Returns:
            特徴量重要度の辞書
        """
        if not self.is_trained or self.model is None:
            raise ModelError("モデルが学習されていません")

        if not self.feature_columns:
            raise ModelError("特徴量カラムが設定されていません")

        model: GradientBoostingClassifier = self.model
        return dict(zip(self.feature_columns, model.feature_importances_))

    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得

        Returns:
            モデル情報の辞書
        """
        if not self.is_trained or self.model is None:
            return {"status": "not_trained"}

        model: GradientBoostingClassifier = self.model
        return {
            "algorithm": self.ALGORITHM_NAME,
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "subsample": model.subsample,
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "status": "trained",
        }

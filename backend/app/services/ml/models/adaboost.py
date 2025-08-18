"""
AdaBoostモデルラッパー

アンサンブル学習で使用するAdaBoostモデルのラッパークラスを提供します。
scikit-learnのAdaBoostClassifierを使用してアンサンブル専用に最適化されたモデルです。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from ....utils.error_handler import ModelError

logger = logging.getLogger(__name__)


class AdaBoostModel:
    """
    アンサンブル内で使用するAdaBoostモデルラッパー

    scikit-learnのAdaBoostClassifierを使用してアンサンブル専用に最適化されたモデル
    """

    # アルゴリズム名（AlgorithmRegistryから取得）
    ALGORITHM_NAME = "adaboost"

    def __init__(self, automl_config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            automl_config: AutoML設定（現在は未使用）
        """
        self.model = None
        self.is_trained = False
        self._feature_columns = None
        self.automl_config = automl_config

        # ベース推定器
        self.base_estimator = DecisionTreeClassifier(
            max_depth=3, class_weight="balanced", random_state=42
        )

        # デフォルトパラメータ
        self.default_params = {
            "estimator": self.base_estimator,
            "n_estimators": 100,
            "learning_rate": 1.0,
            "algorithm": "SAMME",  # SAMME.R -> SAMME に変更（新しいscikit-learnバージョンでサポート）
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
        AdaBoostモデルを学習

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット

        Returns:
            学習結果の辞書
        """
        try:
            logger.info("🚀 AdaBoostモデルの学習を開始")

            # 特徴量カラムを保存
            self._feature_columns = list(X_train.columns)

            # モデル初期化
            self.model = AdaBoostClassifier(**self.default_params)

            # 学習実行
            self.model.fit(X_train, y_train)

            # 予測
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)

            # 確率予測（AdaBoostは確率予測をサポート）
            y_pred_proba_train = self.model.predict_proba(X_train)
            y_pred_proba_test = self.model.predict_proba(X_test)

            # 共通の評価関数を使用
            from ..common.evaluation_utils import evaluate_model_predictions

            # 包括的な評価指標を計算（テストデータ）
            test_metrics = evaluate_model_predictions(
                y_test, y_pred_test, y_pred_proba_test
            )

            # 包括的な評価指標を計算（学習データ）
            train_metrics = evaluate_model_predictions(
                y_train, y_pred_train, y_pred_proba_train
            )

            # クラス数を取得
            n_classes = len(np.unique(y_train))

            # 特徴量重要度
            feature_importance = dict(
                zip(self._feature_columns, self.model.feature_importances_)
            )

            self.is_trained = True

            results = {
                "algorithm": self.ALGORITHM_NAME,
                "n_estimators": self.model.n_estimators,
                "learning_rate": self.model.learning_rate,
                "adaboost_algorithm": self.model.algorithm,  # キー名を変更して重複を回避
                "feature_count": len(self._feature_columns),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "num_classes": n_classes,
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
                f"✅ AdaBoost学習完了 - テスト精度: {test_metrics.get('accuracy', 0.0):.4f}"
            )
            return results

        except Exception as e:
            logger.error(f"❌ AdaBoost学習エラー: {e}")
            raise ModelError(f"AdaBoost学習に失敗しました: {e}")

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
            if self._feature_columns:
                X = X[self._feature_columns]

            predictions = self.model.predict(X)
            return predictions

        except Exception as e:
            logger.error(f"AdaBoost予測エラー: {e}")
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
            if self._feature_columns:
                X = X[self._feature_columns]

            probabilities = self.model.predict_proba(X)
            return probabilities

        except Exception as e:
            logger.error(f"AdaBoost確率予測エラー: {e}")
            raise ModelError(f"確率予測に失敗しました: {e}")

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
            raise ModelError("モデルが学習されていません")

        if not self._feature_columns:
            raise ModelError("特徴量カラムが設定されていません")

        return dict(zip(self._feature_columns, self.model.feature_importances_))

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
            "learning_rate": self.model.learning_rate,
            "algorithm_type": self.model.algorithm,
            "base_estimator_max_depth": self.base_estimator.max_depth,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "status": "trained",
        }

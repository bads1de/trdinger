"""
NaiveBayesモデルラッパー

アンサンブル学習で使用するNaiveBayesモデルのラッパークラスを提供します。
scikit-learnのGaussianNBを使用してアンサンブル専用に最適化されたモデルです。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class NaiveBayesModel:
    """
    アンサンブル内で使用するNaiveBayesモデルラッパー

    scikit-learnのGaussianNBを使用してアンサンブル専用に最適化されたモデル
    """

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
        self.default_params = {"var_smoothing": 1e-9}

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **training_params,
    ) -> Dict[str, Any]:
        """
        NaiveBayesモデルを学習

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット

        Returns:
            学習結果の辞書
        """
        try:
            logger.info("🔮 NaiveBayesモデルの学習を開始")

            # 特徴量カラムを保存
            self.feature_columns = list(X_train.columns)

            # モデル初期化
            self.model = GaussianNB(**self.default_params)

            # 学習実行
            self.model.fit(X_train, y_train)

            # 予測
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)

            # 評価指標計算
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            train_balanced_acc = balanced_accuracy_score(y_train, y_pred_train)
            test_balanced_acc = balanced_accuracy_score(y_test, y_pred_test)
            train_f1 = f1_score(y_train, y_pred_train, average="weighted")
            test_f1 = f1_score(y_test, y_pred_test, average="weighted")

            # 特徴量重要度（NaiveBayesでは直接的な重要度なし）
            # 各クラスの平均値の差を重要度として近似
            feature_importance = {}
            if hasattr(self.model, "theta_"):
                # クラス間の平均値の分散を重要度として使用
                class_means = self.model.theta_
                if class_means.shape[0] > 1:
                    importance_scores = np.var(class_means, axis=0)
                    feature_importance = dict(
                        zip(
                            self.feature_columns,
                            importance_scores / np.sum(importance_scores),  # 正規化
                        )
                    )

            self.is_trained = True

            results = {
                "algorithm": "naivebayes",
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "train_balanced_accuracy": train_balanced_acc,
                "test_balanced_accuracy": test_balanced_acc,
                "train_f1_score": train_f1,
                "test_f1_score": test_f1,
                "feature_importance": feature_importance,
                "var_smoothing": self.model.var_smoothing,
                "n_classes": len(self.model.classes_),
                "feature_count": len(self.feature_columns),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }

            logger.info(f"✅ NaiveBayes学習完了 - テスト精度: {test_accuracy:.4f}")
            return results

        except Exception as e:
            logger.error(f"❌ NaiveBayes学習エラー: {e}")
            raise UnifiedModelError(f"NaiveBayes学習に失敗しました: {e}")

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
            logger.error(f"NaiveBayes予測エラー: {e}")
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
            logger.error(f"NaiveBayes確率予測エラー: {e}")
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
        特徴量重要度を取得（クラス間分散ベース）

        Returns:
            特徴量重要度の辞書
        """
        if not self.is_trained or self.model is None:
            raise UnifiedModelError("モデルが学習されていません")

        if not self.feature_columns:
            raise UnifiedModelError("特徴量カラムが設定されていません")

        if hasattr(self.model, "theta_"):
            class_means = self.model.theta_
            if class_means.shape[0] > 1:
                importance_scores = np.var(class_means, axis=0)
                normalized_scores = importance_scores / np.sum(importance_scores)
                return dict(zip(self.feature_columns, normalized_scores))

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
            "algorithm": "naivebayes",
            "var_smoothing": self.model.var_smoothing,
            "n_classes": len(self.model.classes_),
            "classes": self.model.classes_.tolist(),
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "status": "trained",
        }

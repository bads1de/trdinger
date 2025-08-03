"""
Ridgeモデルラッパー

アンサンブル学習で使用するRidgeClassifierモデルのラッパークラスを提供します。
scikit-learnのRidgeClassifierを使用してアンサンブル専用に最適化されたモデルです。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)

from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class RidgeModel:
    """
    アンサンブル内で使用するRidgeClassifierモデルラッパー

    scikit-learnのRidgeClassifierを使用してアンサンブル専用に最適化されたモデル
    注意: RidgeClassifierはpredict_probaメソッドを持たないため、アンサンブルでは制限あり
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
        self.default_params = {
            "alpha": 1.0,
            "class_weight": "balanced",
            "random_state": 42,
            "max_iter": 1000,
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
        Ridgeモデルを学習

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット

        Returns:
            学習結果の辞書
        """
        try:
            logger.info("📏 Ridgeモデルの学習を開始")

            # 特徴量カラムを保存
            self.feature_columns = list(X_train.columns)

            # モデル初期化
            self.model = RidgeClassifier(**self.default_params)

            # 学習実行
            self.model.fit(X_train, y_train)

            # 予測
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)

            # 基本評価指標計算
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            train_balanced_acc = balanced_accuracy_score(y_train, y_pred_train)
            test_balanced_acc = balanced_accuracy_score(y_test, y_pred_test)
            train_f1 = f1_score(y_train, y_pred_train, average="weighted")
            test_f1 = f1_score(y_test, y_pred_test, average="weighted")

            # 追加評価指標計算（AUC指標は除外：RidgeClassifierはpredict_probaをサポートしない）
            train_mcc = matthews_corrcoef(y_train, y_pred_train)
            test_mcc = matthews_corrcoef(y_test, y_pred_test)

            # クラス数を取得
            n_classes = len(np.unique(y_train))

            # 特徴量重要度（係数の絶対値）
            if hasattr(self.model, "coef_"):
                if len(self.model.coef_.shape) > 1:
                    # 多クラス分類の場合
                    feature_importance = dict(
                        zip(
                            self.feature_columns,
                            np.mean(np.abs(self.model.coef_), axis=0),
                        )
                    )
                else:
                    # 二値分類の場合
                    feature_importance = dict(
                        zip(self.feature_columns, np.abs(self.model.coef_))
                    )
            else:
                feature_importance = {}

            self.is_trained = True

            results = {
                "algorithm": "ridge",
                # 基本指標
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "accuracy": test_accuracy,  # フロントエンド用の統一キー
                "train_balanced_accuracy": train_balanced_acc,
                "test_balanced_accuracy": test_balanced_acc,
                "balanced_accuracy": test_balanced_acc,  # フロントエンド用の統一キー
                "train_f1_score": train_f1,
                "test_f1_score": test_f1,
                "f1_score": test_f1,  # フロントエンド用の統一キー
                # 追加指標（AUC指標は除外）
                "train_mcc": train_mcc,
                "test_mcc": test_mcc,
                "matthews_corrcoef": test_mcc,  # フロントエンド用の統一キー
                # モデル情報
                "feature_importance": feature_importance,
                "alpha": self.model.alpha,
                "max_iter": self.model.max_iter,
                "feature_count": len(self.feature_columns),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "num_classes": n_classes,
                "has_predict_proba": False,  # RidgeClassifierは確率予測なし
            }

            logger.info(f"✅ Ridge学習完了 - テスト精度: {test_accuracy:.4f}")
            return results

        except Exception as e:
            logger.error(f"❌ Ridge学習エラー: {e}")
            raise UnifiedModelError(f"Ridge学習に失敗しました: {e}")

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
            logger.error(f"Ridge予測エラー: {e}")
            raise UnifiedModelError(f"予測に失敗しました: {e}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測確率を取得（RidgeClassifierは対応していないため例外を発生）

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率の配列

        Raises:
            UnifiedModelError: RidgeClassifierは確率予測に対応していない
        """
        raise UnifiedModelError(
            "RidgeClassifierは確率予測（predict_proba）に対応していません"
        )

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
        特徴量重要度を取得（係数の絶対値）

        Returns:
            特徴量重要度の辞書
        """
        if not self.is_trained or self.model is None:
            raise UnifiedModelError("モデルが学習されていません")

        if not self.feature_columns:
            raise UnifiedModelError("特徴量カラムが設定されていません")

        if hasattr(self.model, "coef_"):
            if len(self.model.coef_.shape) > 1:
                # 多クラス分類の場合
                importance = np.mean(np.abs(self.model.coef_), axis=0)
            else:
                # 二値分類の場合
                importance = np.abs(self.model.coef_)

            return dict(zip(self.feature_columns, importance))
        else:
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
            "algorithm": "ridge",
            "alpha": self.model.alpha,
            "max_iter": self.model.max_iter,
            "class_weight": "balanced",
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "has_predict_proba": False,
            "status": "trained",
        }

"""
Ridgeモデルラッパー

アンサンブル学習で使用するRidgeClassifierモデルのラッパークラスを提供します。
scikit-learnのRidgeClassifierを使用してアンサンブル専用に最適化されたモデルです。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier

from ..evaluation.enhanced_metrics import (
    EnhancedMetricsCalculator,
    MetricsConfig,
)
from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class RidgeModel:
    """
    アンサンブル内で使用するRidgeClassifierモデルラッパー

    scikit-learnのRidgeClassifierを使用してアンサンブル専用に最適化されたモデル
    注意: RidgeClassifierはpredict_probaメソッドを持たないため、アンサンブルでは制限あり
    """

    ALGORITHM_NAME = "ridge"

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

            # 統一された評価指標計算器を使用
            config = MetricsConfig(
                include_balanced_accuracy=True,
                include_pr_auc=False,  # RidgeClassifierは確率予測をサポートしない
                include_roc_auc=False,  # RidgeClassifierは確率予測をサポートしない
                include_confusion_matrix=True,
                include_classification_report=True,
                average_method="weighted",
                zero_division=0,
            )

            metrics_calculator = EnhancedMetricsCalculator(config)

            # 包括的な評価指標を計算（テストデータ）
            # RidgeClassifierは確率予測をサポートしないため、y_probaはNone
            test_metrics = metrics_calculator.calculate_comprehensive_metrics(
                y_test, y_pred_test, y_proba=None
            )

            # 包括的な評価指標を計算（学習データ）
            train_metrics = metrics_calculator.calculate_comprehensive_metrics(
                y_train, y_pred_train, y_proba=None
            )

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

            # クラス数を取得
            n_classes = len(np.unique(y_train))

            results = {
                "algorithm": self.ALGORITHM_NAME,
                "alpha": self.model.alpha,
                "max_iter": self.model.max_iter,
                "feature_count": len(self.feature_columns),
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

            # 学習データの評価指標を追加（プレフィックス付き）
            for key, value in train_metrics.items():
                if key not in ["error"]:  # エラー情報は除外
                    results[f"train_{key}"] = value

            results["has_predict_proba"] = False  # RidgeClassifierは確率予測なし

            logger.info(
                f"✅ Ridge学習完了 - テスト精度: {test_metrics['accuracy']:.4f}"
            )
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
            "algorithm": self.ALGORITHM_NAME,
            "alpha": self.model.alpha,
            "max_iter": self.model.max_iter,
            "class_weight": "balanced",
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "has_predict_proba": False,
            "status": "trained",
        }

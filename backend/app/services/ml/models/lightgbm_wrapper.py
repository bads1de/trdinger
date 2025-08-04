"""
LightGBMモデルラッパー

アンサンブル学習で使用するLightGBMモデルのラッパークラスを提供します。
LightGBMTrainerの機能を簡略化してアンサンブル専用に最適化されています。
"""

import logging
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class LightGBMModel:
    """
    アンサンブル内で使用するLightGBMモデルラッパー

    LightGBMTrainerの機能を簡略化してアンサンブル専用に最適化
    """

    # アルゴリズム名（AlgorithmRegistryから取得）
    ALGORITHM_NAME = "lightgbm"

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

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        LightGBMモデルを学習

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット

        Returns:
            学習結果
        """
        try:
            # 特徴量カラムを保存
            self.feature_columns = X_train.columns.tolist()

            # LightGBMデータセットを作成
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            # クラス数を判定
            num_classes = len(np.unique(y_train))

            # LightGBMパラメータ
            params = {
                "objective": "multiclass" if num_classes > 2 else "binary",
                "num_class": num_classes if num_classes > 2 else None,
                "metric": "multi_logloss" if num_classes > 2 else "binary_logloss",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.1,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "random_state": 42,
            }

            # モデル学習
            self.model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, valid_data],
                valid_names=["train", "valid"],
                num_boost_round=100,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(0),  # ログを抑制
                ],
            )

            # 予測と評価
            y_pred_proba = self.model.predict(
                X_test, num_iteration=self.model.best_iteration
            )

            if num_classes > 2:
                y_pred_class = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred_class = (y_pred_proba > 0.5).astype(int)

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

            # 包括的な評価指標を計算
            detailed_metrics = metrics_calculator.calculate_comprehensive_metrics(
                y_test, y_pred_class, y_pred_proba
            )

            # 特徴量重要度を計算
            feature_importance = {}
            if self.model and hasattr(self.model, "feature_importance"):
                importance_scores = self.model.feature_importance(
                    importance_type="gain"
                )
                feature_importance = dict(zip(self.feature_columns, importance_scores))
                logger.info(f"特徴量重要度を計算: {len(feature_importance)}個の特徴量")

            self.is_trained = True

            logger.info(f"LightGBM学習開始: {num_classes}クラス分類")
            logger.info(f"クラス分布: {dict(y_train.value_counts())}")
            logger.info(
                f"LightGBMモデル学習完了: 精度={detailed_metrics.get('accuracy', 0.0):.4f}"
            )

            # 詳細な評価指標を含む結果を返す
            result = {
                "algorithm": self.ALGORITHM_NAME,  # アルゴリズム名を追加
                "num_classes": num_classes,
                "best_iteration": self.model.best_iteration,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": len(self.feature_columns),
                "feature_importance": feature_importance,  # 特徴量重要度を追加
                **detailed_metrics,  # 詳細な評価指標を追加
            }

            return result

        except Exception as e:
            logger.error(f"LightGBMモデル学習エラー: {e}")
            raise UnifiedModelError(f"LightGBMモデル学習に失敗しました: {e}")

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Args:
            top_n: 上位N個の特徴量

        Returns:
            特徴量重要度の辞書
        """
        if not self.is_trained or not self.model:
            logger.warning("学習済みモデルがありません")
            return {}

        try:
            # LightGBMの特徴量重要度を取得
            importance_scores = self.model.feature_importance(importance_type="gain")

            if not self.feature_columns or len(importance_scores) != len(
                self.feature_columns
            ):
                logger.warning("特徴量カラム情報が不正です")
                return {}

            # 特徴量名と重要度のペアを作成
            feature_importance = dict(zip(self.feature_columns, importance_scores))

            # 重要度でソートして上位N個を取得
            sorted_importance = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:top_n]

            return dict(sorted_importance)

        except Exception as e:
            logger.error(f"特徴量重要度取得エラー: {e}")
            return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率
        """
        if not self.is_trained or self.model is None:
            raise UnifiedModelError("学習済みモデルがありません")

        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測確率を取得

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率
        """
        return self.predict(X)

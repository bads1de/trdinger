"""
XGBoostモデルラッパー

アンサンブル学習で使用するXGBoostモデルのラッパークラスを提供します。
XGBoostを使用してアンサンブル専用に最適化されたモデルです。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    アンサンブル内で使用するXGBoostモデルラッパー

    XGBoostを使用してアンサンブル専用に最適化されたモデル
    """

    # アルゴリズム名（AlgorithmRegistryから取得）
    ALGORITHM_NAME = "xgboost"

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
        XGBoostモデルを学習

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

            # クラス数を判定
            num_classes = len(np.unique(y_train))

            # XGBoostパラメータ
            params = {
                "objective": "multi:softprob" if num_classes > 2 else "binary:logistic",
                "num_class": num_classes if num_classes > 2 else None,
                "eval_metric": "mlogloss" if num_classes > 2 else "logloss",
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbosity": 0,
            }

            # XGBoostデータセットを作成
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            # モデル学習
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, "train"), (dtest, "eval")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            # 予測と評価
            y_pred_proba = self.model.predict(dtest)

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
            if self.model and hasattr(self.model, "get_score"):
                try:
                    importance_scores = self.model.get_score(importance_type="gain")
                    # XGBoostの特徴量名をDataFrameのカラム名にマッピング
                    feature_importance = {}
                    for i, col in enumerate(self.feature_columns):
                        feature_key = f"f{i}"
                        if feature_key in importance_scores:
                            feature_importance[col] = importance_scores[feature_key]
                        else:
                            feature_importance[col] = 0.0
                except Exception as e:
                    logger.warning(f"特徴量重要度の計算に失敗: {e}")

            self.is_trained = True

            logger.info(f"XGBoost学習開始: {num_classes}クラス分類")
            logger.info(f"クラス分布: {dict(y_train.value_counts())}")
            logger.info(
                f"XGBoostモデル学習完了: 精度={detailed_metrics.get('accuracy', 0.0):.4f}"
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

        except ImportError:
            logger.error(
                "XGBoostがインストールされていません。pip install xgboostを実行してください。"
            )
            raise UnifiedModelError("XGBoostがインストールされていません")
        except Exception as e:
            logger.error(f"XGBoostモデル学習エラー: {e}")
            raise UnifiedModelError(f"XGBoostモデル学習に失敗しました: {e}")

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

        dtest = xgb.DMatrix(X)
        predictions = self.model.predict(dtest)
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
            # XGBoostの特徴量重要度を取得
            importance_scores = self.model.get_score(importance_type="gain")

            if not self.feature_columns:
                logger.warning("特徴量カラム情報がありません")
                return {}

            # 特徴量名と重要度のペアを作成（存在しない特徴量は0とする）
            feature_importance = {}
            for feature in self.feature_columns:
                feature_importance[feature] = importance_scores.get(feature, 0.0)

            # 重要度でソートして上位N個を取得
            sorted_importance = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:top_n]

            result = dict(sorted_importance)
            logger.info(f"XGBoost特徴量重要度を取得: {len(result)}個")
            return result

        except Exception as e:
            logger.error(f"XGBoost特徴量重要度取得エラー: {e}")
            return {}

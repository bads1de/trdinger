"""
CatBoostモデルラッパー

アンサンブル学習で使用するCatBoostモデルのラッパークラスを提供します。
CatBoostを使用してアンサンブル専用に最適化されたモデルです。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ....utils.error_handler import ModelError

logger = logging.getLogger(__name__)


class CatBoostModel:
    """
    アンサンブル内で使用するCatBoostモデルラッパー

    CatBoostを使用してアンサンブル専用に最適化されたモデル
    """

    # アルゴリズム名（AlgorithmRegistryから取得）
    ALGORITHM_NAME = "catboost"

    def __init__(self, automl_config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            automl_config: AutoML設定（現在は未使用）
        """
        self.model = None
        self.is_trained = False
        self.feature_columns = None
        self.scaler = None
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
        CatBoostモデルを学習

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット

        Returns:
            学習結果
        """
        try:
            from catboost import CatBoostClassifier  # type: ignore

            # 特徴量カラムを保存
            self.feature_columns = X_train.columns.tolist()

            # クラス数を判定
            num_classes = len(np.unique(y_train))

            # CatBoostパラメータ
            params = {
                "iterations": 100,
                "learning_rate": 0.1,
                "depth": 6,
                "l2_leaf_reg": 3,
                "bootstrap_type": "Bernoulli",  # Bayesian -> Bernoulli (subsampleと互換性あり)
                "subsample": 0.8,
                "random_seed": 42,
                "verbose": False,
                "allow_writing_files": False,  # ファイル出力を無効化
            }

            # 多クラス分類の場合の設定
            if num_classes > 2:
                params["loss_function"] = "MultiClass"
                params["classes_count"] = num_classes
            else:
                params["loss_function"] = "Logloss"

            # モデルを作成
            self.model = CatBoostClassifier(**params)

            # モデル学習
            self.model.fit(
                X_train,
                y_train,
                eval_set=(X_test, y_test),
                early_stopping_rounds=50,
                verbose=False,
            )

            # 予測と評価
            y_pred_proba = self.model.predict_proba(X_test)

            if num_classes > 2:
                y_pred_class = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred_class = (y_pred_proba[:, 1] > 0.5).astype(int)

            # 共通の評価関数を使用
            from ..common.evaluation_utils import evaluate_model_predictions

            # 学習データでも予測を実行
            y_pred_proba_train = self.model.predict_proba(X_train)
            if num_classes > 2:
                y_pred_class_train = np.argmax(y_pred_proba_train, axis=1)
            else:
                y_pred_class_train = (y_pred_proba_train[:, 1] > 0.5).astype(int)

            # 包括的な評価指標を計算（テストデータ）
            test_metrics = evaluate_model_predictions(
                y_test, y_pred_class, y_pred_proba
            )

            # 包括的な評価指標を計算（学習データ）
            train_metrics = evaluate_model_predictions(
                y_train, y_pred_class_train, y_pred_proba_train
            )

            # 特徴量重要度を計算
            feature_importance = {}
            if self.model and hasattr(self.model, "get_feature_importance"):
                try:
                    importance_scores = self.model.get_feature_importance()
                    feature_importance = dict(
                        zip(self.feature_columns, importance_scores)
                    )
                except Exception as e:
                    logger.warning(f"特徴量重要度の計算に失敗: {e}")

            self.is_trained = True

            logger.info(f"CatBoost学習開始: {num_classes}クラス分類")
            logger.info(f"クラス分布: {dict(y_train.value_counts())}")
            logger.info(
                f"CatBoostモデル学習完了: 精度={test_metrics.get('accuracy', 0.0):.4f}"
            )

            # 詳細な評価指標を含む結果を返す
            result = {
                "algorithm": self.ALGORITHM_NAME,  # アルゴリズム名を追加
                "num_classes": num_classes,
                "best_iteration": self.model.get_best_iteration(),
                "feature_count": len(self.feature_columns),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_importance": feature_importance,
            }

            # テストデータの評価指標を追加（プレフィックス付き）
            for key, value in test_metrics.items():
                if key not in ["error"]:  # エラー情報は除外
                    result[f"test_{key}"] = value

            # 学習データの評価指標を追加（プレフィックス付き）
            for key, value in train_metrics.items():
                if key not in ["error"]:  # エラー情報は除外
                    result[f"train_{key}"] = value

            return result

        except ImportError:
            logger.error(
                "CatBoostがインストールされていません。pip install catboostを実行してください。"
            )
            raise ModelError("CatBoostがインストールされていません")
        except Exception as e:
            logger.error(f"CatBoostモデル学習エラー: {e}")
            raise ModelError(f"CatBoostモデル学習に失敗しました: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率
        """
        if not self.is_trained or self.model is None:
            raise ModelError("学習済みモデルがありません")

        try:
            predictions = self.model.predict_proba(X)
            return predictions
        except Exception as e:
            raise ModelError(f"CatBoost予測エラー: {e}")

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
            # CatBoostの特徴量重要度を取得
            importance_scores = self.model.get_feature_importance()

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

            result = dict(sorted_importance)
            logger.info(f"CatBoost特徴量重要度を取得: {len(result)}個")
            return result

        except Exception as e:
            logger.error(f"CatBoost特徴量重要度取得エラー: {e}")
            return {}

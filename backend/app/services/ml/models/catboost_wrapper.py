"""
CatBoostモデルラッパー

アンサンブル学習で使用するCatBoostモデルのラッパークラスを提供します。
CatBoostを使用してアンサンブル専用に最適化されたモデルです。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class CatBoostModel:
    """
    アンサンブル内で使用するCatBoostモデルラッパー

    CatBoostを使用してアンサンブル専用に最適化されたモデル
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
            from catboost import CatBoostClassifier

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

            # 詳細な評価指標を計算
            from sklearn.metrics import (
                accuracy_score,
                balanced_accuracy_score,
                f1_score,
                matthews_corrcoef,
                roc_auc_score,
                average_precision_score,
            )

            # 学習データでも予測を実行
            y_pred_proba_train = self.model.predict_proba(X_train)
            if num_classes > 2:
                y_pred_class_train = np.argmax(y_pred_proba_train, axis=1)
            else:
                y_pred_class_train = (y_pred_proba_train[:, 1] > 0.5).astype(int)

            # 基本評価指標計算
            train_accuracy = accuracy_score(y_train, y_pred_class_train)
            test_accuracy = accuracy_score(y_test, y_pred_class)
            train_balanced_acc = balanced_accuracy_score(y_train, y_pred_class_train)
            test_balanced_acc = balanced_accuracy_score(y_test, y_pred_class)
            train_f1 = f1_score(y_train, y_pred_class_train, average="weighted")
            test_f1 = f1_score(y_test, y_pred_class, average="weighted")

            # 追加評価指標計算
            train_mcc = matthews_corrcoef(y_train, y_pred_class_train)
            test_mcc = matthews_corrcoef(y_test, y_pred_class)

            # AUC指標（多クラス対応）
            if num_classes == 2:
                # 二値分類
                train_roc_auc = roc_auc_score(y_train, y_pred_proba_train[:, 1])
                test_roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                train_pr_auc = average_precision_score(
                    y_train, y_pred_proba_train[:, 1]
                )
                test_pr_auc = average_precision_score(y_test, y_pred_proba[:, 1])
            else:
                # 多クラス分類
                train_roc_auc = roc_auc_score(
                    y_train, y_pred_proba_train, multi_class="ovr", average="weighted"
                )
                test_roc_auc = roc_auc_score(
                    y_test, y_pred_proba, multi_class="ovr", average="weighted"
                )
                # 多クラスPR-AUCは各クラスの平均
                train_pr_aucs = []
                test_pr_aucs = []
                for i in range(num_classes):
                    train_binary = (y_train == i).astype(int)
                    test_binary = (y_test == i).astype(int)
                    train_pr_aucs.append(
                        average_precision_score(train_binary, y_pred_proba_train[:, i])
                    )
                    test_pr_aucs.append(
                        average_precision_score(test_binary, y_pred_proba[:, i])
                    )
                train_pr_auc = np.mean(train_pr_aucs)
                test_pr_auc = np.mean(test_pr_aucs)

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
            logger.info(f"CatBoostモデル学習完了: 精度={test_accuracy:.4f}")

            # 詳細な評価指標を含む結果を返す
            return {
                "algorithm": "catboost",  # アルゴリズム名を追加
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
                # 追加指標
                "train_mcc": train_mcc,
                "test_mcc": test_mcc,
                "matthews_corrcoef": test_mcc,  # フロントエンド用の統一キー
                "train_roc_auc": train_roc_auc,
                "test_roc_auc": test_roc_auc,
                "auc_roc": test_roc_auc,  # フロントエンド用の統一キー
                "train_pr_auc": train_pr_auc,
                "test_pr_auc": test_pr_auc,
                "auc_pr": test_pr_auc,  # フロントエンド用の統一キー
                # モデル情報
                "feature_importance": feature_importance,
                "num_classes": num_classes,
                "best_iteration": self.model.get_best_iteration(),
                "feature_count": len(self.feature_columns),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }

        except ImportError:
            logger.error(
                "CatBoostがインストールされていません。pip install catboostを実行してください。"
            )
            raise UnifiedModelError("CatBoostがインストールされていません")
        except Exception as e:
            logger.error(f"CatBoostモデル学習エラー: {e}")
            raise UnifiedModelError(f"CatBoostモデル学習に失敗しました: {e}")

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

        try:
            predictions = self.model.predict_proba(X)
            return predictions
        except Exception as e:
            raise UnifiedModelError(f"CatBoost予測エラー: {e}")

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

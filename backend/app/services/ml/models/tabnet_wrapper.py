"""
TabNetモデルラッパー

アンサンブル学習で使用するTabNetモデルのラッパークラスを提供します。
TabNetを使用してアンサンブル専用に最適化されたモデルです。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class TabNetModel:
    """
    アンサンブル内で使用するTabNetモデルラッパー

    TabNetを使用してアンサンブル専用に最適化されたモデル
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
        TabNetモデルを学習

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット

        Returns:
            学習結果
        """
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            from sklearn.preprocessing import LabelEncoder

            # 特徴量カラムを保存
            self.feature_columns = X_train.columns.tolist()

            # クラス数を判定
            unique_classes = np.unique(y_train)
            num_classes = len(unique_classes)

            # ラベルエンコーディング（TabNetは0から始まる連続した整数を期待）
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)

            # TabNetパラメータ
            import torch.optim as optim
            from torch.optim.lr_scheduler import StepLR

            params = {
                "n_d": 8,  # 決定層の次元
                "n_a": 8,  # 注意層の次元
                "n_steps": 3,  # ステップ数
                "gamma": 1.3,  # 特徴量再利用係数
                "lambda_sparse": 1e-3,  # スパース正則化
                "optimizer_fn": optim.Adam,
                "optimizer_params": {"lr": 2e-2},
                "mask_type": "sparsemax",
                "scheduler_params": {"step_size": 10, "gamma": 0.9},
                "scheduler_fn": StepLR,
                "verbose": 0,
                "seed": 42,
            }

            # モデルを作成
            self.model = TabNetClassifier(**params)

            # データを numpy 配列に変換
            X_train_np = X_train.values.astype(np.float32)
            X_test_np = X_test.values.astype(np.float32)

            # モデル学習
            self.model.fit(
                X_train_np,
                y_train_encoded,
                eval_set=[(X_test_np, y_test_encoded)],
                eval_name=["test"],
                eval_metric=["accuracy"],
                max_epochs=50,
                patience=10,
                batch_size=256,
                virtual_batch_size=128,
                drop_last=False,
            )

            # 予測と評価
            y_pred_proba = self.model.predict_proba(X_test_np)

            if num_classes > 2:
                y_pred_class = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred_class = (y_pred_proba[:, 1] > 0.5).astype(int)

            # 元のラベルに戻す
            y_pred_class_original = self.label_encoder.inverse_transform(y_pred_class)

            # 学習データでも予測を実行
            y_pred_proba_train = self.model.predict_proba(X_train_np)
            if num_classes > 2:
                y_pred_class_train = np.argmax(y_pred_proba_train, axis=1)
            else:
                y_pred_class_train = (y_pred_proba_train[:, 1] > 0.5).astype(int)

            y_pred_class_train_original = self.label_encoder.inverse_transform(
                y_pred_class_train
            )

            # 詳細な評価指標を計算
            from sklearn.metrics import (
                accuracy_score,
                average_precision_score,
                balanced_accuracy_score,
                f1_score,
                matthews_corrcoef,
                roc_auc_score,
            )

            # 基本評価指標計算
            train_accuracy = accuracy_score(y_train, y_pred_class_train_original)
            test_accuracy = accuracy_score(y_test, y_pred_class_original)
            train_balanced_acc = balanced_accuracy_score(
                y_train, y_pred_class_train_original
            )
            test_balanced_acc = balanced_accuracy_score(y_test, y_pred_class_original)
            train_f1 = f1_score(
                y_train, y_pred_class_train_original, average="weighted"
            )
            test_f1 = f1_score(y_test, y_pred_class_original, average="weighted")

            # 追加評価指標計算
            train_mcc = matthews_corrcoef(y_train, y_pred_class_train_original)
            test_mcc = matthews_corrcoef(y_test, y_pred_class_original)

            # AUC指標（多クラス対応）
            # TabNetの場合、エンコードされたラベルでAUCを計算
            if num_classes == 2:
                # 二値分類
                train_roc_auc = roc_auc_score(y_train_encoded, y_pred_proba_train[:, 1])
                test_roc_auc = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
                train_pr_auc = average_precision_score(
                    y_train_encoded, y_pred_proba_train[:, 1]
                )
                test_pr_auc = average_precision_score(
                    y_test_encoded, y_pred_proba[:, 1]
                )
            else:
                # 多クラス分類
                train_roc_auc = roc_auc_score(
                    y_train_encoded,
                    y_pred_proba_train,
                    multi_class="ovr",
                    average="weighted",
                )
                test_roc_auc = roc_auc_score(
                    y_test_encoded, y_pred_proba, multi_class="ovr", average="weighted"
                )
                # 多クラスPR-AUCは各クラスの平均
                train_pr_aucs = []
                test_pr_aucs = []
                for i in range(num_classes):
                    train_binary = (y_train_encoded == i).astype(int)
                    test_binary = (y_test_encoded == i).astype(int)
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
            if self.model and hasattr(self.model, "feature_importances_"):
                try:
                    importance_scores = self.model.feature_importances_
                    feature_importance = dict(
                        zip(self.feature_columns, importance_scores)
                    )
                except Exception as e:
                    logger.warning(f"特徴量重要度の計算に失敗: {e}")

            self.is_trained = True

            logger.info(f"TabNet学習開始: {num_classes}クラス分類")
            logger.info(f"クラス分布: {dict(y_train.value_counts())}")
            logger.info(f"TabNetモデル学習完了: 精度={test_accuracy:.4f}")

            # 詳細な評価指標を含む結果を返す
            return {
                "algorithm": "tabnet",  # アルゴリズム名を追加
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
                "best_epoch": len(self.model.history["loss"]),
                "feature_count": len(self.feature_columns),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }

        except ImportError:
            logger.error(
                "pytorch-tabnetがインストールされていません。pip install pytorch-tabnetを実行してください。"
            )
            raise UnifiedModelError("pytorch-tabnetがインストールされていません")
        except Exception as e:
            logger.error(f"TabNetモデル学習エラー: {e}")
            raise UnifiedModelError(f"TabNetモデル学習に失敗しました: {e}")

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
            # データを numpy 配列に変換
            X_np = X.values.astype(np.float32)
            predictions = self.model.predict_proba(X_np)
            return predictions
        except Exception as e:
            raise UnifiedModelError(f"TabNet予測エラー: {e}")

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
            # TabNetの特徴量重要度を取得
            if hasattr(self.model, "feature_importances_"):
                importance_scores = self.model.feature_importances_
            else:
                logger.warning("TabNetモデルに特徴量重要度がありません")
                return {}

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
            logger.info(f"TabNet特徴量重要度を取得: {len(result)}個")
            return result

        except Exception as e:
            logger.error(f"TabNet特徴量重要度取得エラー: {e}")
            return {}

"""
TabNetモデルラッパー

アンサンブル学習で使用するTabNetモデルのラッパークラスを提供します。
TabNetを使用してアンサンブル専用に最適化されたモデルです。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

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

            # 評価指標を計算
            from sklearn.metrics import accuracy_score

            accuracy = accuracy_score(y_test, y_pred_class_original)

            self.is_trained = True

            logger.info(f"TabNet学習開始: {num_classes}クラス分類")
            logger.info(f"クラス分布: {dict(y_train.value_counts())}")
            logger.info(f"TabNetモデル学習完了: 精度={accuracy:.4f}")

            return {
                "accuracy": accuracy,
                "num_classes": num_classes,
                "best_epoch": len(self.model.history["loss"]),
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

"""
XGBoostモデルラッパー

アンサンブル学習で使用するXGBoostモデルのラッパークラスを提供します。
XGBoostを使用してアンサンブル専用に最適化されたモデルです。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    アンサンブル内で使用するXGBoostモデルラッパー

    XGBoostを使用してアンサンブル専用に最適化されたモデル
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
            import xgboost as xgb

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

            # 評価指標を計算
            from sklearn.metrics import accuracy_score

            accuracy = accuracy_score(y_test, y_pred_class)

            self.is_trained = True

            logger.info(f"XGBoost学習開始: {num_classes}クラス分類")
            logger.info(f"クラス分布: {dict(y_train.value_counts())}")
            logger.info(f"XGBoostモデル学習完了: 精度={accuracy:.4f}")

            return {
                "accuracy": accuracy,
                "num_classes": num_classes,
                "best_iteration": self.model.best_iteration,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }

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

        try:
            import xgboost as xgb

            dtest = xgb.DMatrix(X)
            predictions = self.model.predict(dtest)
            return predictions
        except ImportError:
            raise UnifiedModelError("XGBoostがインストールされていません")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測確率を取得

        Args:
            X: 特徴量DataFrame

        Returns:
            予測確率
        """
        return self.predict(X)

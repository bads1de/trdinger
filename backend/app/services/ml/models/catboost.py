"""
CatBoostモデルラッパー

アンサンブル学習で使用するCatBoostモデルのラッパークラスを提供します。
時系列データに特化したOrdered Boostingなど、CatBoost特有の機能を活用します。
"""

import logging
from typing import Any, Dict, List, Optional, Union

import catboost as cb
import numpy as np
import pandas as pd

from ....utils.error_handler import ModelError

logger = logging.getLogger(__name__)


class CatBoostModel:
    """
    アンサンブル内で使用するCatBoostモデルラッパー

    CatBoostの強みを活かした時系列データ向けの実装
    sklearn互換のインターフェースを提供
    """

    def __init__(
        self,
        random_state: int = 42,
        iterations: int = 100,
        learning_rate: float = 0.1,
        **kwargs,
    ):
        """
        初期化

        Args:
            random_state: ランダムシード
            iterations: イテレーション数
            learning_rate: 学習率
            **kwargs: その他のパラメータ
        """
        self.random_state = random_state
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.kwargs = kwargs

        # モデルのデフォルトパラメータ
        self.default_params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": random_state,
            "verbose": 0,
            "allow_writing_files": False,  # 一時ファイル作成を無効化
        }

        # kwargsでデフォルトを上書き
        self.default_params.update(kwargs)

        self.model: Optional[cb.CatBoostClassifier] = None
        self.feature_columns: Optional[List[str]] = None
        self.is_fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs,
    ):
        """
        sklearn互換のfitメソッド

        Args:
            X: 学習用特徴量（DataFrame or numpy array）
            y: 学習用ターゲット（Series or numpy array）
            **kwargs: その他のパラメータ（class_weightなど）

        Returns:
            self: 学習済みモデル
        """
        try:
            logger.info("CatBoostモデルの学習を開始")

            # データ型の変換
            if isinstance(X, pd.DataFrame):
                self.feature_columns = X.columns.tolist()
                X_train = X.values
            else:
                X_train = X

            if isinstance(y, pd.Series):
                y_train = y.values
            else:
                y_train = y

            # パラメータの準備
            params = self.default_params.copy()

            # class_weight処理
            if "class_weight" in kwargs:
                class_weight = kwargs["class_weight"]
                if class_weight == "balanced" or class_weight == "Balanced":
                    params["auto_class_weights"] = "Balanced"
                    logger.info("auto_class_weights='Balanced'を適用")
                elif isinstance(class_weight, dict):
                    # カスタムクラスウェイトの設定
                    params["class_weights"] = list(class_weight.values())
                    logger.info(f"カスタムクラスウェイト: {class_weight}")

            # モデルの作成と学習
            self.model = cb.CatBoostClassifier(**params)
            self.model.fit(X_train, y_train)

            self.is_fitted = True
            logger.info("CatBoostモデルの学習が完了")

            return self

        except Exception as e:
            logger.error(f"CatBoostモデルの学習でエラー: {e}")
            raise ModelError(f"CatBoostモデルの学習に失敗しました: {e}")

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
            logger.info("CatBoostモデル学習開始")

            # 特徴量カラムを保存
            self.feature_columns = X_train.columns.tolist()

            # パラメータの準備
            params = self.default_params.copy()

            # class_weight処理
            if "class_weight" in kwargs:
                class_weight = kwargs["class_weight"]
                if class_weight == "balanced":
                    params["auto_class_weights"] = "Balanced"

            # モデルの作成
            self.model = cb.CatBoostClassifier(**params)

            # 学習
            self.model.fit(
                X_train.values,
                y_train.values,
                eval_set=(X_test.values, y_test.values),
                verbose=False,
            )

            self.is_fitted = True

            # 予測
            y_pred = self.model.predict(X_test.values)

            # 評価結果（統一された評価関数を使用）
            from ..common.evaluation_utils import evaluate_model_predictions

            eval_metrics = evaluate_model_predictions(y_test, y_pred)

            accuracy = eval_metrics.get("accuracy", 0.0)
            f1 = eval_metrics.get("f1_score", 0.0)

            result = {
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "model_type": "CatBoost",
                "n_features": len(self.feature_columns),
                **eval_metrics,  # その他の指標も念のため含める
            }

            logger.info(f"CatBoost学習完了: Accuracy={accuracy:.4f}, F1={f1:.4f}")

            return result

        except Exception as e:
            logger.error(f"CatBoostモデル学習エラー: {e}")
            raise ModelError(f"CatBoostモデルの学習に失敗しました: {e}")

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Args:
            top_n: 上位N個の特徴量

        Returns:
            特徴量重要度の辞書
        """
        if not self.is_fitted or self.model is None:
            logger.warning("モデルが学習されていません")
            return {}

        try:
            # CatBoostの特徴量重要度を取得
            importance = self.model.get_feature_importance()

            if self.feature_columns:
                feature_importance = dict(zip(self.feature_columns, importance))
            else:
                feature_importance = {
                    f"feature_{i}": imp for i, imp in enumerate(importance)
                }

            # 上位N個を抽出
            sorted_importance = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:top_n]

            return dict(sorted_importance)

        except Exception as e:
            logger.error(f"特徴量重要度の取得でエラー: {e}")
            return {}

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        sklearn互換の予測メソッド（クラス予測）

        Args:
            X: 特徴量（DataFrame or numpy array）

        Returns:
            予測クラス
        """
        if not self.is_fitted or self.model is None:
            raise ModelError("モデルが学習されていません")

        try:
            # データ型の変換
            if isinstance(X, pd.DataFrame):
                X_pred = X.values
            else:
                X_pred = X

            # 予測
            predictions = self.model.predict(X_pred)

            return predictions

        except Exception as e:
            logger.error(f"予測でエラー: {e}")
            raise ModelError(f"予測に失敗しました: {e}")

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        予測確率を取得

        Args:
            X: 特徴量（DataFrame or numpy array）

        Returns:
            予測確率
        """
        if not self.is_fitted or self.model is None:
            raise ModelError("モデルが学習されていません")

        try:
            # データ型の変換
            if isinstance(X, pd.DataFrame):
                X_pred = X.values
            else:
                X_pred = X

            # 予測確率
            predictions_proba = self.model.predict_proba(X_pred)

            return predictions_proba

        except Exception as e:
            logger.error(f"予測確率の取得でエラー: {e}")
            raise ModelError(f"予測確率の取得に失敗しました: {e}")

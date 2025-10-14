"""
LightGBMモデルラッパー

アンサンブル学習で使用するLightGBMモデルのラッパークラスを提供します。
LightGBMTrainerの機能を簡略化してアンサンブル専用に最適化されています。
"""

import logging
from typing import Any, Dict, List, Optional, Union, cast

import lightgbm as lgb
import numpy as np
import pandas as pd

from ....utils.error_handler import ModelError

logger = logging.getLogger(__name__)


class LightGBMModel:
    """
    アンサンブル内で使用するLightGBMモデルラッパー

    LightGBMTrainerの機能を簡略化してアンサンブル専用に最適化
    sklearn互換のインターフェースを提供
    """

    # アルゴリズム名（AlgorithmRegistryから取得）
    ALGORITHM_NAME = "lightgbm"

    def __init__(
        self,
        automl_config: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        **kwargs,
    ):
        """
        初期化

        Args:
            automl_config: AutoML設定（現在は未使用）
            random_state: ランダムシード
            n_estimators: エスティメータ数
            learning_rate: 学習率
            **kwargs: その他のパラメータ
        """
        self.model = None
        self.is_trained = False
        self.feature_columns: Optional[List[str]] = None
        self.scaler = None
        self.automl_config = automl_config
        self.classes_ = None  # sklearn互換性のため

        # sklearn互換性のためのパラメータ
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        # その他のパラメータを設定
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> "LightGBMModel":
        """
        sklearn互換のfitメソッド

        Args:
            X: 学習用特徴量（DataFrame or numpy array）
            y: 学習用ターゲット（Series or numpy array）

        Returns:
            self: 学習済みモデル
        """
        try:
            # numpy配列をDataFrameに変換
            if not isinstance(X, pd.DataFrame):
                if hasattr(self, "feature_columns") and self.feature_columns:
                    X = pd.DataFrame(X, columns=cast(Any, self.feature_columns))
                else:
                    columns = [f"feature_{i}" for i in range(X.shape[1])]
                    X = pd.DataFrame(X, columns=cast(Any, columns))

            if not isinstance(y, pd.Series):
                y = pd.Series(y)

            # データを80:20で分割（バリデーション用）
            from sklearn.model_selection import train_test_split

            # stratifyパラメータは分類タスクでのみ使用可能
            stratify_param = y if len(np.unique(y)) < 20 else None

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_param
            )

            # 明示的な型キャストを追加
            X_train = cast(pd.DataFrame, X_train)
            X_val = cast(pd.DataFrame, X_val)
            y_train = cast(pd.Series, y_train)
            y_val = cast(pd.Series, y_val)

            # 内部の学習メソッドを呼び出し
            self._train_model_impl(X_train, X_val, y_train, y_val)

            # classes_属性を設定（sklearn互換性のため）
            self.classes_ = np.unique(y)

            return self

        except Exception as e:
            logger.error(f"sklearn互換fit実行エラー: {e}")
            raise ModelError(f"LightGBMモデルのfit実行に失敗しました: {e}")

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
            y_pred_proba = cast(
                np.ndarray,
                self.model.predict(X_test, num_iteration=self.model.best_iteration),
            )

            if num_classes > 2:
                y_pred_class = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred_class = (y_pred_proba > 0.5).astype(int)

            # 共通の評価関数を使用
            from ..common.evaluation_utils import evaluate_model_predictions

            detailed_metrics = evaluate_model_predictions(
                y_test, y_pred_class, y_pred_proba
            )

            # 特徴量重要度を計算
            feature_importance = {}
            if (
                self.model
                and hasattr(self.model, "feature_importance")
                and self.feature_columns
            ):
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
                "feature_count": (
                    len(self.feature_columns) if self.feature_columns else 0
                ),
                "feature_importance": feature_importance,  # 特徴量重要度を追加
                **detailed_metrics,  # 詳細な評価指標を追加
            }

            return result

        except Exception as e:
            logger.error(f"LightGBMモデル学習エラー: {e}")
            raise ModelError(f"LightGBMモデル学習に失敗しました: {e}")

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

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        sklearn互換の予測メソッド（クラス予測）

        Args:
            X: 特徴量（DataFrame or numpy array）

        Returns:
            予測クラス
        """
        if not self.is_trained or self.model is None:
            raise ModelError("学習済みモデルがありません")

        try:
            # numpy配列をDataFrameに変換
            if not isinstance(X, pd.DataFrame):
                if hasattr(self, "feature_columns") and self.feature_columns:
                    X = pd.DataFrame(X, columns=cast(Any, self.feature_columns))
                else:
                    X = pd.DataFrame(
                        X,
                        columns=cast(Any, [f"feature_{i}" for i in range(X.shape[1])]),
                    )

            # 予測確率を取得
            predictions_proba = cast(
                np.ndarray,
                self.model.predict(X, num_iteration=self.model.best_iteration),
            )

            # クラス数を判定
            if predictions_proba.ndim == 1:
                # 二値分類の場合
                predictions = (predictions_proba > 0.5).astype(int)
            else:
                # 多クラス分類の場合
                predictions = np.argmax(predictions_proba, axis=1)

            return predictions

        except Exception as e:
            logger.error(f"予測実行エラー: {e}")
            raise ModelError(f"予測実行に失敗しました: {e}")

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        予測確率を取得

        Args:
            X: 特徴量（DataFrame or numpy array）

        Returns:
            予測確率
        """
        if not self.is_trained or self.model is None:
            raise ModelError("学習済みモデルがありません")

        try:
            # numpy配列をDataFrameに変換
            if not isinstance(X, pd.DataFrame):
                if hasattr(self, "feature_columns") and self.feature_columns:
                    X = pd.DataFrame(X, columns=cast(Any, self.feature_columns))
                else:
                    X = pd.DataFrame(
                        X,
                        columns=cast(Any, [f"feature_{i}" for i in range(X.shape[1])]),
                    )

            predictions = cast(
                np.ndarray,
                self.model.predict(X, num_iteration=self.model.best_iteration),
            )

            # 二値分類の場合、確率を[1-p, p]の形式に変換
            if predictions.ndim == 1:
                predictions_proba = np.column_stack([1 - predictions, predictions])
                return cast(np.ndarray, predictions_proba)
            else:
                # 多クラス分類の場合はそのまま返す
                return predictions

        except Exception as e:
            logger.error(f"予測確率取得エラー: {e}")
            raise ModelError(f"予測確率取得に失敗しました: {e}")

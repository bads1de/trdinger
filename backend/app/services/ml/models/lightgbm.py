import logging
from typing import Any, Dict, List, Optional, Union, cast

import lightgbm as lgb
import numpy as np
import pandas as pd

from app.utils.error_handler import ModelError
from .base_gradient_boosting_model import BaseGradientBoostingModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseGradientBoostingModel):
    """
    LightGBMモデルラッパー

    BaseGradientBoostingModelを継承し、LightGBM固有の実装を提供します。
    """

    ALGORITHM_NAME = "lightgbm"

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        **kwargs,
    ):
        """
        初期化

        Args:
            random_state: ランダムシード
            n_estimators: エスティメータ数
            learning_rate: 学習率
            **kwargs: その他のパラメータ
        """
        super().__init__(random_state=random_state, **kwargs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def _create_dataset(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> lgb.Dataset:
        """
        LightGBM固有のデータセットオブジェクトを作成します。
        """
        return lgb.Dataset(X, label=y, weight=sample_weight, free_raw_data=False)

    def _get_model_params(self, num_classes: int, **kwargs) -> Dict[str, Any]:
        """
        LightGBM固有のパラメータディクショナリを生成します。
        """
        params = {
            "objective": "multiclass" if num_classes > 2 else "binary",
            "num_class": num_classes if num_classes > 2 else None,
            "metric": "multi_logloss" if num_classes > 2 else "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": kwargs.get("num_leaves", 31),
            "learning_rate": kwargs.get("learning_rate", self.learning_rate),
            "feature_fraction": kwargs.get("feature_fraction", 0.9),
            "bagging_fraction": kwargs.get("bagging_fraction", 0.8),
            "bagging_freq": kwargs.get("bagging_freq", 5),
            "verbose": -1,
            "random_state": self.random_state,
        }
        # 渡された kwargs でデフォルトを上書き
        params.update(kwargs)
        return params

    def _train_internal(
        self,
        train_data: lgb.Dataset,
        valid_data: lgb.Dataset,
        params: Dict[str, Any],
        early_stopping_rounds: Optional[int] = None,
        **kwargs,
    ) -> lgb.Booster:
        """
        LightGBM固有の学習プロセスを実行します。
        """
        callbacks = [lgb.log_evaluation(0)]
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))

        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            num_boost_round=kwargs.get("num_boost_round", self.n_estimators),
            callbacks=callbacks,
        )
        return model

    def _get_prediction_proba(self, data: lgb.Dataset) -> np.ndarray:
        """
        LightGBM固有の予測確率を取得します。
        """
        if self.model is None:
            raise ModelError("学習済みモデルがありません")
        
        # lgb.Datasetから特徴量データを取り出す
        # construct()で内部表現を構築し、get_data()でデータを取得
        X_data = data.construct().get_data()

        return cast(
            np.ndarray,
            self.model.predict(X_data, num_iteration=self.model.best_iteration),
        )

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        sklearn互換の予測メソッド（クラス予測）
        """
        if not self.is_trained or self.model is None:
            raise ModelError("学習済みモデルがありません")

        # feature_columnsをfit時に保存しているので、それを使用してDataFrameを整形
        if self.feature_columns and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=cast(Any, self.feature_columns))

        predictions_proba = cast(
            np.ndarray,
            self.model.predict(X, num_iteration=self.model.best_iteration),
        )

        if predictions_proba.ndim == 1:
            return (predictions_proba > 0.5).astype(int)
        else:
            return np.argmax(predictions_proba, axis=1)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        予測確率を取得
        """
        if not self.is_trained or self.model is None:
            raise ModelError("学習済みモデルがありません")

        # feature_columnsをfit時に保存しているので、それを使用してDataFrameを整形
        if self.feature_columns and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=cast(Any, self.feature_columns))

        predictions = cast(
            np.ndarray,
            self.model.predict(X, num_iteration=self.model.best_iteration),
        )

        if predictions.ndim == 1:
            return np.column_stack([1 - predictions, predictions])
        else:
            return predictions

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        特徴量重要度を取得
        """
        if not self.is_trained or self.model is None:
            logger.warning("学習済みモデルがありません")
            return {}

        try:
            importance_scores = self.model.feature_importance(importance_type="gain")

            if not self.feature_columns or len(importance_scores) != len(
                self.feature_columns
            ):
                logger.warning("特徴量カラム情報が不正です")
                return {}

            feature_importance = dict(zip(self.feature_columns, importance_scores))
            sorted_importance = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:top_n]

            return dict(sorted_importance)

        except Exception as e:
            logger.error(f"特徴量重要度取得エラー: {e}")
            return {}

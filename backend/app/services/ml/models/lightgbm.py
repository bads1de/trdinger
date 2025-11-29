import logging
from typing import Any, Dict, Optional, Union, cast

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
        valid_data: Optional[lgb.Dataset],
        params: Dict[str, Any],
        early_stopping_rounds: Optional[int] = None,
        **kwargs,
    ) -> lgb.Booster:
        """
        LightGBM固有の学習プロセスを実行します。
        """
        callbacks = [lgb.log_evaluation(0)]
        
        valid_sets = [train_data]
        valid_names = ["train"]
        
        if valid_data:
            valid_sets.append(valid_data)
            valid_names.append("valid")
            if early_stopping_rounds:
                callbacks.append(lgb.early_stopping(early_stopping_rounds))

        model = lgb.train(
            params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
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

    def _prepare_input_for_prediction(self, X: pd.DataFrame) -> Any:
        """
        予測用の入力データを準備します。
        LightGBMはDataFrameを直接受け取れます。
        """
        return X

    def _predict_raw(self, data: Any) -> np.ndarray:
        """
        モデルから生の予測値（確率）を取得します。
        """
        return cast(
            np.ndarray,
            self.model.predict(data, num_iteration=self.model.best_iteration),
        )
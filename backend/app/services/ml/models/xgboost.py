import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb

from app.utils.error_handler import ModelError

from .base_gradient_boosting_model import BaseGradientBoostingModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseGradientBoostingModel):
    """
    XGBoostモデルラッパー

    BaseGradientBoostingModelを継承し、XGBoost固有の実装を提供します。
    """

    ALGORITHM_NAME = "xgboost"

    def __init__(
        self,
        random_state: int = 42,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        **kwargs,
    ):
        """
        初期化
        """
        super().__init__(random_state=random_state, **kwargs)
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.feature_names: Optional[List[str]] = None
        self.best_iteration: Optional[int] = None  # 早期停止の最適イテレーション

    def _create_dataset(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> xgb.DMatrix:
        """XGBoost固有のデータセットを作成"""
        # 基底クラスで既にDataFrame化されているはずだが、念のため
        if not isinstance(X, pd.DataFrame):
            X = self._coerce_feature_frame(X, self.feature_columns)

        self.feature_names = X.columns.tolist()
        return xgb.DMatrix(
            X, label=y, feature_names=self.feature_names, weight=sample_weight
        )

    def _get_model_params(self, num_classes: int, **kwargs) -> Dict[str, Any]:
        """
        XGBoost固有のパラメータディクショナリを生成します。
        """
        is_regression = self._is_regression_task()
        params = {
            "objective": (
                "reg:squarederror"
                if is_regression
                else ("multi:softprob" if num_classes > 2 else "binary:logistic")
            ),
            "num_class": (
                None if is_regression else (num_classes if num_classes > 2 else None)
            ),
            "eval_metric": (
                "rmse"
                if is_regression
                else ("mlogloss" if num_classes > 2 else "logloss")
            ),
            "max_depth": kwargs.get("max_depth", self.max_depth),
            "learning_rate": kwargs.get("learning_rate", self.learning_rate),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
            "random_state": self.random_state,
            "verbosity": 0,
        }
        # 渡された kwargs でデフォルトを上書き
        params.update(kwargs)
        return params

    def _train_internal(
        self,
        train_data: xgb.DMatrix,
        valid_data: Optional[xgb.DMatrix],
        params: Dict[str, Any],
        early_stopping_rounds: Optional[int] = None,
        **kwargs,
    ) -> xgb.Booster:
        """
        XGBoost固有の学習プロセスを実行します。
        """
        evals = [(train_data, "train")]
        if valid_data:
            evals.append((valid_data, "eval"))

        actual_early_stopping_rounds = early_stopping_rounds if valid_data else None

        model = xgb.train(
            params,
            train_data,
            num_boost_round=kwargs.get("num_boost_round", self.n_estimators),
            evals=evals,
            early_stopping_rounds=actual_early_stopping_rounds,
            verbose_eval=False,
        )

        # 早期停止の最適イテレーションを保存
        if hasattr(model, "best_iteration"):
            self.best_iteration = model.best_iteration
        elif hasattr(model, "best_ntree_limit"):
            self.best_iteration = model.best_ntree_limit

        return model

    def _get_prediction_proba(self, data: xgb.DMatrix) -> np.ndarray:
        """
        XGBoost固有の予測確率を取得します。
        早期停止の最適イテレーションを使用します。
        """
        if self.model is None:
            raise ModelError("学習済みモデルがありません")
        return self.model.predict(data)

    def _prepare_input_for_prediction(self, X: pd.DataFrame) -> xgb.DMatrix:
        """
        予測用の入力データを準備します。
        特徴量の順序が訓練時と一致することを保証します。
        """
        # 特徴量名の順序検証と修正
        if self.feature_names:
            # 予測時に入力 DataFrame の列順序を訓練時に合わせる
            missing_cols = set(self.feature_names) - set(X.columns)
            extra_cols = set(X.columns) - set(self.feature_names)

            if missing_cols:
                logger.warning(
                    f"予測時に特徴量が不足しています: {missing_cols}。"
                    f"0で補完します。"
                )
                for col in missing_cols:
                    X[col] = 0.0

            if extra_cols:
                logger.debug(
                    f"予測時に不要な特徴量が含まれています: {extra_cols}。"
                    f"削除します。"
                )

            # 訓練時と同じ順序に列を並べ替える
            X = X[self.feature_names]

        return xgb.DMatrix(X, feature_names=self.feature_names)

    def _predict_raw(self, data: Any) -> np.ndarray:
        """
        モデルから生の予測値（確率）を取得します。
        早期停止の最適イテレーションを使用します。
        """
        if self.model is None:
            raise ModelError("学習済みモデルがありません")

        # 早期停止の最適イテレーションを使用
        if self.best_iteration is not None:
            # XGBoost >= 1.6.0 では iteration_range を使用
            return self.model.predict(
                data, iteration_range=(0, self.best_iteration + 1)
            )

        return self.model.predict(data)

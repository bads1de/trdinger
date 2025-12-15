import logging
from typing import Any, Dict, List, Optional, Union, cast

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
        n_estimators: int = 100,  # LightGBMと合わせて追加
        **kwargs,
    ):
        """
        初期化
        """
        super().__init__(random_state=random_state, **kwargs)
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators  # LightGBMと合わせて追加
        self.feature_names: Optional[List[str]] = None

    def _create_dataset(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> xgb.DMatrix:
        """
        XGBoost固有のデータセットオブジェクトを作成します。
        """
        if not isinstance(X, pd.DataFrame):
            # feature_columnsが設定されていない場合は仮の列名を使用
            columns = (
                self.feature_columns
                if self.feature_columns
                else [f"feature_{i}" for i in range(X.shape[1])]
            )
            X = pd.DataFrame(X, columns=cast(Any, columns))

        # 特徴量名を保存 (XGBoostのDMatrixで必要になるため)
        self.feature_names = X.columns.tolist()

        return xgb.DMatrix(
            X, label=y, feature_names=self.feature_names, weight=sample_weight
        )

    def _get_model_params(self, num_classes: int, **kwargs) -> Dict[str, Any]:
        """
        XGBoost固有のパラメータディクショナリを生成します。
        """
        params = {
            "objective": "multi:softprob" if num_classes > 2 else "binary:logistic",
            "num_class": num_classes if num_classes > 2 else None,
            "eval_metric": "mlogloss" if num_classes > 2 else "logloss",
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
        return model

    def _get_prediction_proba(self, data: xgb.DMatrix) -> np.ndarray:
        """
        XGBoost固有の予測確率を取得します。
        """
        if self.model is None:
            raise ModelError("学習済みモデルがありません")
        return self.model.predict(data)

    def _prepare_input_for_prediction(self, X: pd.DataFrame) -> Any:
        """
        予測用の入力データを準備します。
        XGBoostはDMatrixを使用します。
        """
        return xgb.DMatrix(X, feature_names=self.feature_names)

    def _predict_raw(self, data: Any) -> np.ndarray:
        """
        モデルから生の予測値（確率）を取得します。
        """
        return self.model.predict(data)



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

        return xgb.DMatrix(X, label=y, feature_names=self.feature_names, weight=sample_weight)

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
        valid_data: xgb.DMatrix,
        params: Dict[str, Any],
        early_stopping_rounds: Optional[int] = None,
        **kwargs,
    ) -> xgb.Booster:
        """
        XGBoost固有の学習プロセスを実行します。
        """
        model = xgb.train(
            params,
            train_data,
            num_boost_round=kwargs.get("num_boost_round", self.n_estimators),
            evals=[(train_data, "train"), (valid_data, "eval")],
            early_stopping_rounds=early_stopping_rounds,
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

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        予測を実行
        """
        if not self.is_trained or self.model is None:
            raise ModelError("学習済みモデルがありません")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=cast(Any, self.feature_columns))

        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        predictions = self.model.predict(dtest)

        if predictions.ndim > 1 and predictions.shape[1] > 1:
            return np.argmax(predictions, axis=1)
        else:
            return (predictions > 0.5).astype(int)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        予測確率を取得
        """
        if not self.is_trained or self.model is None:
            raise ModelError("学習済みモデルがありません")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=cast(Any, self.feature_columns))

        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        probabilities = self.model.predict(dtest)

        if probabilities.ndim == 1:
            probabilities = np.column_stack([1 - probabilities, probabilities])

        return probabilities

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        特徴量重要度を取得
        """
        if not self.is_trained or self.model is None:
            logger.warning("学習済みモデルがありません")
            return {}

        try:
            if not hasattr(self.model, "get_score"):
                logger.warning("モデルまたはget_score()メソッドがありません")
                return {col: 0.0 for col in self.feature_columns}

            importance_scores = self.model.get_score(importance_type="gain")
            feature_importance = {}

            if self.feature_names:
                for feature_name in self.feature_names:
                    feature_importance[feature_name] = importance_scores.get(
                        feature_name, 0.0
                    )
            else:
                for i, col in enumerate(self.feature_columns):
                    feature_key = f"f{i}"
                    feature_importance[col] = importance_scores.get(feature_key, 0.0)

            sorted_importance = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:top_n]

            return dict(sorted_importance)

        except Exception as e:
            logger.error(f"特徴量重要度取得エラー: {e}")
            return {}

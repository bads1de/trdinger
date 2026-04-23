"""future_log_realized_vol 回帰専用トレーナー。"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

import numpy as np
import pandas as pd

from app.utils.error_handler import ModelError

from ..models.base_gradient_boosting_model import BaseGradientBoostingModel
from ..models.lightgbm import LightGBMModel
from ..models.xgboost import XGBoostModel
from .base_ml_trainer import BaseMLTrainer


class VolatilityRegressionTrainer(BaseMLTrainer):
    """
    将来のボラティリティ（実現ボラティリティの対数）を予測する回帰モデルの学習を担当します。

    このトレーナーの主な目的は、市場の「嵐」を予測することです。
    予測されたボラティリティが一定の閾値を超える場合、MLフィルター（ボラティリティゲート）が
    新規エントリーを抑制し、不安定な相場での損失を回避します。
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        トレーナーを初期化します。

        Args:
            model_type (str): 使用するアルゴリズム。"lightgbm" または "xgboost"。デフォルトは "lightgbm"。
            model_params (Optional[Dict]): モデルに渡すハイパーパラメータ。
        """
        super().__init__(trainer_config=model_params or {})
        self.model_type = model_type
        self.model_params = model_params or {}

    def _build_model(self) -> BaseGradientBoostingModel:
        """
        指定されたアルゴリズムに基づいてモデルインスタンスを構築します。

        Returns:
            BaseGradientBoostingModel: LightGBMModel または XGBoostModel のインスタンス。

        Raises:
            ModelError: 未知の `model_type` が指定された場合。
        """
        params = {
            **self.model_params,
            "task_type": "volatility_regression",
        }
        if self.model_type == "lightgbm":
            return LightGBMModel(**params)
        if self.model_type == "xgboost":
            return XGBoostModel(**params)
        raise ModelError(
            f"サポートされていない回帰モデルです: {self.model_type}"
        )

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        与えられた特徴量から将来のボラティリティを予測します。

        Args:
            features_df (pd.DataFrame): 予測に使用する特徴量。

        Returns:
            np.ndarray: 予測されたボラティリティ（対数スケール）の配列。

        Raises:
            ModelError: モデルが学習されていない場合。
        """
        if self._model is None:
            raise ModelError("学習済みモデルがありません")
        return np.asarray(self._model.predict(features_df))

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **training_params,
    ) -> Dict[str, Any]:
        """
        モデルの学習を実際に実行します。

        Args:
            X_train (pd.DataFrame): 学習用特徴量。
            X_test (pd.DataFrame): 検証用特徴量。
            y_train (pd.Series): 学習用ターゲット（将来のボラティリティ）。
            y_test (pd.Series): 検証用ターゲット。
            **training_params: 早期終了回数、学習率、最大深度等のパラメータ。

        Returns:
            Dict[str, Any]: 学習結果のメトリクス。
                - "algorithm" (str): 使用したアルゴリズム名。
                - "train_samples" (int): 学習に使用したサンプル数。
                - "rmse" / "mae" (float): 検証データでの誤差指標（モデルクラスにより提供）。
                - "feature_count" (int): 使用された特徴量数。
        """
        model = cast(Any, self._build_model())
        self._model = cast(Any, model)
        eval_set = (
            [(X_test, y_test)]
            if X_test is not None and len(X_test) > 0
            else None
        )
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=training_params.get("early_stopping_rounds"),
            n_estimators=training_params.get("n_estimators"),
            learning_rate=training_params.get("learning_rate"),
            max_depth=training_params.get("max_depth"),
        )
        self.feature_columns = list(
            model.feature_columns or X_train.columns.tolist()
        )
        self.is_trained = True

        result = dict(getattr(self._model, "last_training_result", {}) or {})
        result.setdefault("algorithm", self.model_type)
        result.setdefault("train_samples", len(X_train))
        result.setdefault(
            "test_samples", len(X_test) if X_test is not None else 0
        )
        result.setdefault("feature_count", len(self.feature_columns))
        result.setdefault(
            "gate_cutoff_log_rv",
            self._coerce_float_param(
                training_params.get("gate_cutoff_log_rv", 0.0),
                0.0,
            ),
        )
        result.setdefault(
            "gate_cutoff_vol",
            self._coerce_float_param(
                training_params.get("gate_cutoff_vol", 1.0),
                1.0,
            ),
        )
        return result

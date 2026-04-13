"""
CatBoostモデルラッパー

BaseGradientBoostingModelを継承し、CatBoost固有の実装を提供します。
時系列データに特化したOrdered Boostingなど、CatBoost特有の機能を活用します。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import catboost as cb
import numpy as np
import pandas as pd

from app.utils.error_handler import ModelError

from .base_gradient_boosting_model import BaseGradientBoostingModel

logger = logging.getLogger(__name__)


class CatBoostModel(BaseGradientBoostingModel):
    """
    CatBoostモデルラッパー

    BaseGradientBoostingModelを継承し、CatBoost固有の実装を提供します。
    """

    ALGORITHM_NAME = "catboost"

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
        super().__init__(random_state=random_state, **kwargs)
        self.iterations = iterations
        self.learning_rate = learning_rate

    def _handle_class_weight_for_catboost(
        self, class_weight: Any, kwargs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        CatBoost用のclass_weight処理をオーバーライド

        Args:
            class_weight: class_weightパラメータ
            kwargs: その他のパラメータ

        Returns:
            CatBoost固有のパラメータ辞書
        """
        if not class_weight:
            return None

        catboost_params = {}
        if class_weight == "balanced" or class_weight == "Balanced":
            catboost_params["auto_class_weights"] = "Balanced"
            logger.info("auto_class_weights='Balanced'を適用")
        elif isinstance(class_weight, dict):
            # カスタムクラスウェイトの設定
            catboost_params["class_weights"] = list(class_weight.values())  # type: ignore[assignment]
            logger.info(f"カスタムクラスウェイト: {class_weight}")

        return catboost_params if catboost_params else None

    def _create_dataset(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        データセットをタプル形式で作成します。
        テスト互換性のため (X_values, y_values, sample_weight) のタプルを返します。
        """
        X_data = self._coerce_feature_frame(X, self.feature_columns).values

        if y is not None:
            y_data = self._coerce_target_series(y).values
            return (X_data, y_data, sample_weight)
        return (X_data, None, sample_weight)

    def _get_model_params(self, num_classes: int, **kwargs) -> Dict[str, Any]:
        """CatBoost固有のパラメータを生成"""
        params = {
            "iterations": kwargs.get("iterations", self.iterations),
            "learning_rate": kwargs.get("learning_rate", self.learning_rate),
            "depth": kwargs.get("depth", 6),
            "l2_leaf_reg": kwargs.get("l2_leaf_reg", 3.0),
            "random_seed": self.random_state,
            "verbose": 0,
            "allow_writing_files": False,
        }
        # class_weight関連の追加
        for k in ["auto_class_weights", "class_weights"]:
            if k in kwargs:
                params[k] = kwargs[k]

        # 残りのパラメータをマージ
        exclude = {
            "class_weight",
            "early_stopping_rounds",
            "num_boost_round",
            *params.keys(),
        }
        params.update({k: v for k, v in kwargs.items() if k not in exclude})
        return params

    def _train_internal(
        self,
        train_data: Any,
        valid_data: Any,
        params: Dict[str, Any],
        early_stopping_rounds: Optional[int] = None,
        **kwargs,
    ) -> cb.CatBoostRegressor:
        """
        CatBoost固有の学習プロセスを実行します。
        回帰タスクのみをサポートします。
        train_dataとvalid_dataは (X, y, sample_weight) のタプル形式を受け取ります。
        """
        # タプルからデータを展開
        X_train, y_train = train_data[0], train_data[1]
        sample_weight_train = train_data[2] if len(train_data) > 2 else None

        # CatBoostのPoolオブジェクトに変換（sample_weightを含む）
        train_pool = cb.Pool(data=X_train, label=y_train, weight=sample_weight_train)

        # eval_setの準備
        eval_set = None
        if valid_data is not None:
            X_val, y_val = valid_data[0], valid_data[1]
            sample_weight_val = valid_data[2] if len(valid_data) > 2 else None
            eval_pool = cb.Pool(data=X_val, label=y_val, weight=sample_weight_val)
            eval_set = [eval_pool]

        # CatBoostRegressorを作成（回帰タスク専用）
        model = cb.CatBoostRegressor(**params)

        # 学習
        model.fit(
            train_pool,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
            verbose=False,
        )

        return model

    def _get_prediction_proba(self, data: Any) -> np.ndarray:
        """
        CatBoost固有の予測値を取得します。
        回帰タスクのため、predict() を直接呼び出します。
        dataは (X, y) のタプル、cb.Poolオブジェクト、またはnumpy配列を受け取ります。
        """
        if self.model is None:
            raise ModelError("学習済みモデルがありません")

        # タプルの場合はXデータを抽出
        if isinstance(data, tuple):
            X_data = data[0]
        else:
            X_data = data

        # 回帰タスク: predict() を使用し、形状を (n_samples, 1) に整える
        predictions = self.model.predict(X_data)
        return np.asarray(predictions).reshape(-1, 1)

    def _prepare_input_for_prediction(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測用の入力データを準備します。
        CatBoostはnumpy配列を直接受け取ります。
        """
        return X.values

    def _predict_raw(self, data: Any) -> np.ndarray:
        """
        モデルから生の予測値を取得します。
        回帰タスクのため predict() を使用します。
        """
        if self.model is None:
            raise ModelError("学習済みモデルがありません")
        return self.model.predict(data)

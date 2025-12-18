"""
CatBoostモデルラッパー

BaseGradientBoostingModelを継承し、CatBoost固有の実装を提供します。
時系列データに特化したOrdered Boostingなど、CatBoost特有の機能を活用します。
"""

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
            catboost_params["class_weights"] = list(class_weight.values())
            logger.info(f"カスタムクラスウェイト: {class_weight}")

        return catboost_params if catboost_params else None

    def _create_dataset(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Any:
        """
        CatBoost固有のデータセットオブジェクトを作成します。
        CatBoostはnumpy配列を直接受け取ります。
        """
        # CatBoostはnumpy配列を直接受け取る
        if isinstance(X, pd.DataFrame):
            X_data = X.values
        else:
            X_data = X

        if y is not None:
            y_data = y.values if isinstance(y, pd.Series) else y
            return (X_data, y_data)
        return X_data

    def _get_model_params(self, num_classes: int, **kwargs) -> Dict[str, Any]:
        """CatBoost固有のパラメータを生成"""
        params = {
            "iterations": kwargs.get("iterations", self.iterations),
            "learning_rate": kwargs.get("learning_rate", self.learning_rate),
            "depth": kwargs.get("depth", 6), "l2_leaf_reg": kwargs.get("l2_leaf_reg", 3.0),
            "random_seed": self.random_state, "verbose": 0, "allow_writing_files": False
        }
        # class_weight関連の追加
        for k in ["auto_class_weights", "class_weights"]:
            if k in kwargs:
                params[k] = kwargs[k]

        # 残りのパラメータをマージ
        exclude = {"class_weight", "early_stopping_rounds", "num_boost_round", *params.keys()}
        params.update({k: v for k, v in kwargs.items() if k not in exclude})
        return params

    def _train_internal(
        self,
        train_data: Any,
        valid_data: Any,
        params: Dict[str, Any],
        early_stopping_rounds: Optional[int] = None,
        **kwargs,
    ) -> cb.CatBoostClassifier:
        """
        CatBoost固有の学習プロセスを実行します。
        """
        X_train, y_train = train_data
        X_val, y_val = valid_data

        # CatBoostClassifierを作成
        model = cb.CatBoostClassifier(**params)

        # 学習
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            verbose=False,
        )

        return model

    def _get_prediction_proba(self, data: Any) -> np.ndarray:
        """
        CatBoost固有の予測確率を取得します。
        """
        if self.model is None:
            raise ModelError("学習済みモデルがありません")

        # dataはタプル (X, y) の形式
        X_data = data[0] if isinstance(data, tuple) else data

        return self.model.predict_proba(X_data)

    def _prepare_input_for_prediction(self, X: pd.DataFrame) -> Any:
        """
        予測用の入力データを準備します。
        CatBoostはnumpy配列を直接受け取ります。
        """
        return X.values

    def _predict_raw(self, data: Any) -> np.ndarray:
        """
        モデルから生の予測値（確率）を取得します。
        """
        return self.model.predict_proba(data)




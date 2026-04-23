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
    LightGBM アルゴリズムを使用した勾配ブースティング決定木（GBDT）モデルのラッパーです。

    主な特徴:
    - 自動タスク判定: `task_type` 設定に基づいて、分類（バイナリ/マルチクラス）と回帰を自動的に切り替えます。
    - 効率的な学習: ヒストグラムベースのアルゴリズムにより、大規模データでも高速かつメモリ効率良く学習します。
    - 高度な制御: 早期終了（Early Stopping）や各種正則化パラメータをサポートしています。
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
        LightGBMモデルを初期化します。

        Args:
            random_state (int): 再現性のための乱数シード。
            n_estimators (int): 構築する決定木の最大数（ブースティングラウンド数）。
            learning_rate (float): 各ステップの更新の重み。小さいほど慎重な学習になります。
            **kwargs: その他の LightGBM パラメータ（`num_leaves`, `feature_fraction` 等）。
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
        LightGBM 専用のデータセットオブジェクトを作成します。

        Args:
            X: 特徴量行列。
            y: ターゲットラベル（推論時はNone）。
            sample_weight: 各サンプルの重み。

        Returns:
            lgb.Dataset: LightGBM内部で使用される最適化されたデータ構造。
        """
        return lgb.Dataset(
            X, label=y, weight=sample_weight, free_raw_data=False
        )

    def _get_model_params(self, num_classes: int, **kwargs) -> Dict[str, Any]:
        """
        タスクタイプとクラス数に基づいて、LightGBM 固有のパラメータセットを生成します。

        このメソッドは、以下の自動設定を行います：
        - `objective`:
            - "regression" (回帰タスク)
            - "binary" (2クラス分類)
            - "multiclass" (3クラス以上の分類)
        - `metric`:
            - "rmse" (回帰)
            - "binary_logloss" / "multi_logloss" (分類)

        Args:
            num_classes (int): 分類対象のクラス数。
            **kwargs: ユーザー定義の上書きパラメータ。

        Returns:
            Dict[str, Any]: 構築されたパラメータ辞書。
        """
        is_regression = self._is_regression_task()
        is_multi = num_classes > 2 and not is_regression
        params = {
            "objective": (
                "regression"
                if is_regression
                else ("multiclass" if is_multi else "binary")
            ),
            "num_class": (
                None if is_regression else (num_classes if is_multi else None)
            ),
            "metric": (
                "rmse"
                if is_regression
                else ("multi_logloss" if is_multi else "binary_logloss")
            ),
            "boosting_type": "gbdt",
            "verbose": -1,
            "random_state": self.random_state,
            "num_leaves": kwargs.get("num_leaves", 31),
            "learning_rate": kwargs.get("learning_rate", self.learning_rate),
            "feature_fraction": kwargs.get("feature_fraction", 0.9),
            "bagging_fraction": kwargs.get("bagging_fraction", 0.8),
            "bagging_freq": kwargs.get("bagging_freq", 5),
        }
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
        LightGBM の学習エンジンを呼び出し、モデル（Booster）を構築します。

        Args:
            train_data (lgb.Dataset): 学習用データ。
            valid_data (Optional[lgb.Dataset]): 検証用データ。早期終了に使用されます。
            params (Dict[str, Any]): 学習パラメータ。
            early_stopping_rounds (Optional[int]): 指定回数スコアが改善しない場合に学習を打ち切る設定。
            **kwargs: その他の学習設定（`num_boost_round` 等）。

        Returns:
            lgb.Booster: 学習済みのLightGBMブースターオブジェクト。
        """
        callbacks: list[Any] = [lgb.log_evaluation(0)]

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
            self.model.predict(
                X_data, num_iteration=self.model.best_iteration
            ),
        )

    def _prepare_input_for_prediction(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        予測用の入力データを準備します。
        LightGBMはDataFrameを直接受け取れます。
        """
        return X

    def _predict_raw(self, data: Any) -> np.ndarray:
        """
        モデルから生の予測値（確率）を取得します。
        """
        if self.model is None:
            raise ModelError("学習済みモデルがありません")

        return cast(
            np.ndarray,
            self.model.predict(data, num_iteration=self.model.best_iteration),
        )

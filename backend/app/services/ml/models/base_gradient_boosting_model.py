from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from app.utils.error_handler import ModelError

from ..common.utils import get_feature_importance_unified, predict_class_from_proba
from ..evaluation.metrics import metrics_collector

logger = logging.getLogger(__name__)


class BaseGradientBoostingModel(ABC):
    """
    勾配ブースティングモデルの抽象基底クラス

    LightGBM、XGBoost、CatBoostの共通ロジックをカプセル化します。
    sklearn互換のfit/predict_probaインターフェースを提供します。
    """

    ALGORITHM_NAME: str = "base_gradient_boosting"

    def __init__(self, random_state: int = 42, **kwargs):
        self.model = None
        self.is_trained = False
        self.feature_columns: Optional[List[str]] = None
        self.classes_ = None  # sklearn互換性のため
        self.random_state = random_state
        self.task_type = kwargs.pop("task_type", "classification")
        self.last_training_result: Dict[str, Any] = {}

        # その他のパラメータを設定
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def _coerce_feature_frame(
        X: Union[pd.DataFrame, np.ndarray],
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """特徴量入力をDataFrameに正規化する。"""
        if isinstance(X, pd.DataFrame):
            return X
        cols = feature_columns or [f"feature_{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cast(Any, cols))

    @staticmethod
    def _coerce_target_series(y: Union[pd.Series, np.ndarray]) -> pd.Series:
        """ターゲット入力をSeriesに正規化する。"""
        if isinstance(y, pd.Series):
            return y
        return pd.Series(y)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs,
    ) -> "BaseGradientBoostingModel":
        """
        sklearn互換のfitメソッド
        """
        try:
            # 入力整形
            X = self._coerce_feature_frame(X, self.feature_columns)
            y = self._coerce_target_series(y)

            # 検証データ準備
            eval_set = kwargs.get("eval_set")
            early_stop = kwargs.get("early_stopping_rounds")
            X_train, y_train, X_val, y_val = X, y, None, None

            if eval_set and isinstance(eval_set, list) and len(eval_set) > 0:
                X_val, y_val = eval_set[0]
            elif early_stop:
                logger.info(
                    f"Early Stopping有効(rounds={early_stop}): データを分割します"
                )
                idx = int(len(X) * 0.8)
                X_train, X_val, y_train, y_val = (
                    X.iloc[:idx],
                    X.iloc[idx:],
                    y.iloc[:idx],
                    y.iloc[idx:],
                )

            self._train_model_impl(
                cast(pd.DataFrame, X_train),
                cast(Optional[pd.DataFrame], X_val),
                cast(pd.Series, y_train),
                cast(Optional[pd.Series], y_val),
                **kwargs,
            )

            self.classes_ = None if self._is_regression_task() else np.unique(y)
            return self

        except Exception as e:
            logger.error(f"fit実行エラー: {e}")
            raise ModelError(f"{self.ALGORITHM_NAME}モデルのfit失敗: {e}")

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame],
        y_train: pd.Series,
        y_test: Optional[pd.Series],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        モデル学習の共通実装（テンプレートメソッド）
        """
        try:
            self.feature_columns = X_train.columns.tolist()
            num_classes = len(np.unique(y_train))
            is_regression = self._is_regression_task()

            # class_weightの処理
            sample_weight = None
            class_weight = kwargs.get("class_weight")

            # CatBoost用のclass_weight処理（サブクラスでオーバーライド可能）
            if not is_regression:
                catboost_params = self._handle_class_weight_for_catboost(
                    class_weight, kwargs
                )
                if catboost_params:
                    kwargs.update(catboost_params)
                elif class_weight:
                    try:
                        sample_weight = compute_sample_weight(
                            class_weight=class_weight, y=y_train
                        )
                        logger.info(
                            f"class_weight={class_weight} を適用してsample_weightを計算しました"
                        )
                    except Exception as e:
                        logger.warning(f"sample_weightの計算に失敗しました: {e}")

            # モデル固有のデータセット作成
            train_data = self._create_dataset(X_train, y_train, sample_weight)
            valid_data = None
            if X_test is not None and y_test is not None:
                valid_data = self._create_dataset(X_test, y_test)

            # モデル固有の学習パラメータ取得
            params = self._get_model_params(num_classes, **kwargs)
            # sklearn特有のパラメータをモデル用paramsから除去（不具合防止）
            params.pop("class_weight", None)

            # 共通パラメータを除去してモデル固有の学習実行
            train_kwargs = kwargs.copy()
            train_kwargs.pop("class_weight", None)
            es_rounds = train_kwargs.pop("early_stopping_rounds", 50)

            self.model = self._train_internal(
                train_data,
                valid_data,
                params,
                early_stopping_rounds=es_rounds,
                **train_kwargs,
            )

            # 予測と評価 (検証データがある場合のみ)
            detailed_metrics = {}
            if valid_data is not None and y_test is not None and X_test is not None:
                if is_regression:
                    y_pred = np.asarray(self.predict(cast(pd.DataFrame, X_test)))
                    detailed_metrics = (
                        metrics_collector.calculate_volatility_regression_metrics(
                            np.asarray(y_test),
                            y_pred,
                        )
                    )
                    logger.info(
                        f"{self.ALGORITHM_NAME}回帰モデル学習完了: "
                        f"QLIKE={detailed_metrics.get('qlike', 0.0):.4f}, "
                        f"RMSE(log_rv)={detailed_metrics.get('rmse_log_rv', 0.0):.4f}"
                    )
                else:
                    y_pred_proba = self._get_prediction_proba(valid_data)
                    y_pred_class = (
                        np.argmax(y_pred_proba, axis=1)
                        if num_classes > 2
                        else (y_pred_proba > 0.5).astype(int)
                    )

                    detailed_metrics = (
                        metrics_collector.calculate_comprehensive_metrics(
                            y_test, y_pred_class, y_pred_proba
                        )
                    )
                    logger.info(
                        f"{self.ALGORITHM_NAME}モデル学習完了: 精度={detailed_metrics.get('accuracy', 0.0):.4f}"
                    )
            else:
                logger.info(f"{self.ALGORITHM_NAME}モデル学習完了 (検証データなし)")

            # 特徴量重要度
            feature_importance = self.get_feature_importance()

            self.is_trained = True

            result = {
                "algorithm": self.ALGORITHM_NAME,
                "num_classes": 1 if is_regression else num_classes,
                "train_samples": len(X_train),
                "test_samples": len(X_test) if X_test is not None else 0,
                "feature_count": (
                    len(self.feature_columns) if self.feature_columns else 0
                ),
                "feature_importance": feature_importance,
                **detailed_metrics,
            }

            # best_iterationなど、モデル固有の属性を結果に追加
            if self.model is not None and hasattr(self.model, "best_iteration"):
                result["best_iteration"] = self.model.best_iteration

            self.last_training_result = result
            return result

        except Exception as e:
            logger.error(f"{self.ALGORITHM_NAME}モデル学習エラー: {e}")
            raise ModelError(f"{self.ALGORITHM_NAME}モデル学習に失敗しました: {e}")

    def _handle_class_weight_for_catboost(
        self, class_weight: Any, kwargs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        CatBoost用のclass_weight処理（サブクラスでオーバーライド可能）

        Args:
            class_weight: class_weightパラメータ
            kwargs: その他のパラメータ

        Returns:
            CatBoost固有のパラメータ辞書、またはNone（非CatBoostモデルの場合）
        """
        # デフォルトではNoneを返す（LightGBM/XGBoostの場合）
        return None

    @abstractmethod
    def _create_dataset(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> object:
        """
        モデル固有のデータセットオブジェクトを作成します。
        例: lgb.Dataset, xgb.DMatrix
        """

    @abstractmethod
    def _get_model_params(self, num_classes: int, **kwargs) -> Dict[str, object]:
        """
        モデル固有のパラメータディクショナリを生成します。
        """

    @abstractmethod
    def _train_internal(
        self,
        train_data: object,
        valid_data: Optional[object],
        params: Dict[str, object],
        early_stopping_rounds: Optional[int] = None,
        **kwargs,
    ) -> object:
        """
        モデル固有の学習プロセスを実行します。
        """

    @abstractmethod
    def _get_prediction_proba(self, data: Any) -> np.ndarray:
        """
        モデル固有の予測確率を取得します。
        学習時の検証用に使用されます。
        """

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        sklearn互換の予測メソッド（クラス予測）
        """
        if not self.is_trained or self.model is None:
            raise ModelError("学習済みモデルがありません")

        if self.feature_columns and not isinstance(X, pd.DataFrame):
            X = self._coerce_feature_frame(X, self.feature_columns)

        if self._is_regression_task():
            data = self._prepare_input_for_prediction(cast(pd.DataFrame, X))
            return self._predict_raw(data)
        return predict_class_from_proba(self.predict_proba(X))

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        予測確率を取得。
        """
        if not self.is_trained or self.model is None:
            raise ModelError("学習済みモデルがありません")
        if self._is_regression_task():
            raise ModelError("回帰タスクでは predict_proba は利用できません")

        # feature_columnsを使用してDataFrameを整形（常にDataFrameに変換する）
        X_df = self._coerce_feature_frame(X, self.feature_columns or None)

        # モデル固有の入力データ準備
        data = self._prepare_input_for_prediction(X_df)

        # モデル固有の予測実行
        predictions = self._predict_raw(data)

        # 形状の正規化 (1D -> 2D for binary)
        if predictions.ndim == 1:
            return np.column_stack([1 - predictions, predictions])

        return predictions

    @abstractmethod
    def _prepare_input_for_prediction(self, X: pd.DataFrame) -> object:
        """
        予測用の入力データを準備します。
        """

    @abstractmethod
    def _predict_raw(self, data: object) -> np.ndarray:
        """
        モデルから生の予測値（確率）を取得します。
        """

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        特徴量重要度を取得（共通ユーティリティを使用）
        """
        return get_feature_importance_unified(
            self.model, self.feature_columns or [], top_n=top_n
        )

    def _is_regression_task(self) -> bool:
        return str(getattr(self, "task_type", "classification")).lower() in {
            "volatility_regression",
            "regression",
        }

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from app.utils.error_handler import ModelError

from ..common.evaluation_utils import evaluate_model_predictions
from ..common.ml_utils import get_feature_importance_unified

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

        # その他のパラメータを設定
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs,
    ) -> "BaseGradientBoostingModel":
        """
        sklearn互換のfitメソッド。
        内部で_train_model_implを呼び出します。
        """
        try:
            # numpy配列をDataFrameに変換
            if not isinstance(X, pd.DataFrame):
                # feature_columnsが設定されていない場合は仮の列名を使用
                columns = (
                    self.feature_columns
                    if self.feature_columns
                    else [f"feature_{i}" for i in range(X.shape[1])]
                )
                X = pd.DataFrame(X, columns=cast(Any, columns))

            if not isinstance(y, pd.Series):
                y = pd.Series(y)

            # 検証用データの準備
            # 1. eval_setが指定されている場合 (sklearn非標準だがxgboost等はサポート)
            eval_set = kwargs.get("eval_set")
            early_stopping_rounds = kwargs.get("early_stopping_rounds")

            if eval_set:
                # eval_setが提供されている場合はそれを使用
                X_train, y_train = X, y
                # eval_setは [(X_val, y_val)] の形式を想定
                if isinstance(eval_set, list) and len(eval_set) > 0:
                    X_val, y_val = eval_set[0]
                else:
                    logger.warning(
                        "無効なeval_set形式です。検証データなしで学習します。"
                    )
                    X_val, y_val = None, None
            elif early_stopping_rounds:
                # Early Stoppingが有効で、eval_setがない場合は分割が必要
                # 時系列データのため、シャッフルせずに分割（最後の20%を検証用）
                logger.info(
                    f"Early Stopping有効(rounds={early_stopping_rounds}): データを分割して学習します"
                )
                split_index = int(len(X) * 0.8)
                X_train = X.iloc[:split_index]
                X_val = X.iloc[split_index:]
                y_train = y.iloc[:split_index]
                y_val = y.iloc[split_index:]
            else:
                # Early Stoppingが無効、かつeval_setもない場合
                # 全データを学習に使用し、検証データはなし
                X_train, y_train = X, y
                X_val, y_val = None, None

            # 内部の学習メソッドを呼び出し
            self._train_model_impl(
                cast(pd.DataFrame, X_train),
                cast(Optional[pd.DataFrame], X_val),
                cast(pd.Series, y_train),
                cast(Optional[pd.Series], y_val),
                **kwargs,
            )

            self.classes_ = np.unique(y)
            return self

        except Exception as e:
            logger.error(f"sklearn互換fit実行エラー: {e}")
            raise ModelError(f"{self.ALGORITHM_NAME}モデルのfit実行に失敗しました: {e}")

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

            # class_weightの処理
            sample_weight = None
            class_weight = kwargs.get("class_weight")

            # CatBoost用のclass_weight処理（サブクラスでオーバーライド可能）
            catboost_params = self._handle_class_weight_for_catboost(
                class_weight, kwargs
            )
            if catboost_params:
                # CatBoost固有のパラメータがある場合は、kwargsに追加
                kwargs.update(catboost_params)
            elif class_weight:
                # LightGBM/XGBoost用のsample_weight処理
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

            # モデル固有の学習実行
            self.model = self._train_internal(
                train_data,
                valid_data,
                params,
                early_stopping_rounds=kwargs.get("early_stopping_rounds", 50),
                **kwargs,  # ここに **kwargs を追加
            )

            # 予測と評価 (検証データがある場合のみ)
            detailed_metrics = {}
            if valid_data is not None and y_test is not None:
                y_pred_proba = self._get_prediction_proba(valid_data)
                y_pred_class = (
                    np.argmax(y_pred_proba, axis=1)
                    if num_classes > 2
                    else (y_pred_proba > 0.5).astype(int)
                )

                detailed_metrics = evaluate_model_predictions(
                    y_test, y_pred_class, y_pred_proba
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
                "num_classes": num_classes,
                "train_samples": len(X_train),
                "test_samples": len(X_test) if X_test is not None else 0,
                "feature_count": (
                    len(self.feature_columns) if self.feature_columns else 0
                ),
                "feature_importance": feature_importance,
                **detailed_metrics,
            }

            # best_iterationなど、モデル固有の属性を結果に追加
            if hasattr(self.model, "best_iteration"):
                result["best_iteration"] = self.model.best_iteration

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
    ) -> Any:
        """
        モデル固有のデータセットオブジェクトを作成します。
        例: lgb.Dataset, xgb.DMatrix
        """

    @abstractmethod
    def _get_model_params(self, num_classes: int, **kwargs) -> Dict[str, Any]:
        """
        モデル固有のパラメータディクショナリを生成します。
        """

    @abstractmethod
    def _train_internal(
        self,
        train_data: Any,
        valid_data: Optional[Any],
        params: Dict[str, Any],
        early_stopping_rounds: Optional[int] = None,
        **kwargs,
    ) -> Any:
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
        sklearn互換の予測メソッド（クラス予測）。
        """
        if not self.is_trained or self.model is None:
            raise ModelError("学習済みモデルがありません")

        # feature_columnsを使用してDataFrameを整形
        if self.feature_columns and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=cast(Any, self.feature_columns))

        # 予測確率を取得
        predictions_proba = self.predict_proba(X)

        # クラス予測に変換
        if predictions_proba.ndim == 1:
            return (predictions_proba > 0.5).astype(int)
        elif predictions_proba.shape[1] == 2:
            # 2クラス分類の場合、確率が高い方のクラスを返す（0.5閾値と同じ）
            return np.argmax(predictions_proba, axis=1)
        else:
            # 多クラス分類
            return np.argmax(predictions_proba, axis=1)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        予測確率を取得。
        """
        if not self.is_trained or self.model is None:
            raise ModelError("学習済みモデルがありません")

        # feature_columnsを使用してDataFrameを整形
        if self.feature_columns and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=cast(Any, self.feature_columns))

        # モデル固有の入力データ準備
        data = self._prepare_input_for_prediction(X)

        # モデル固有の予測実行
        predictions = self._predict_raw(data)

        # 形状の正規化 (1D -> 2D for binary)
        if predictions.ndim == 1:
            return np.column_stack([1 - predictions, predictions])

        return predictions

    @abstractmethod
    def _prepare_input_for_prediction(self, X: pd.DataFrame) -> Any:
        """
        予測用の入力データを準備します。
        """

    @abstractmethod
    def _predict_raw(self, data: Any) -> np.ndarray:
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

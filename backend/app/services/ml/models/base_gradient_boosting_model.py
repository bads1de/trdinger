
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from app.utils.error_handler import ModelError
from ..common.evaluation_utils import evaluate_model_predictions

logger = logging.getLogger(__name__)


class BaseGradientBoostingModel(ABC):
    """
    勾配ブースティングモデルの抽象基底クラス

    LightGBMModelとXGBoostModelの共通ロジックをカプセル化します。
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

            # 時系列データのため、シャッフルせずに分割（最後の20%を検証用）
            split_index = int(len(X) * 0.8)

            X_train = X.iloc[:split_index]
            X_val = X.iloc[split_index:]
            y_train = y.iloc[:split_index]
            y_val = y.iloc[split_index:]

            # 内部の学習メソッドを呼び出し
            self._train_model_impl(
                cast(pd.DataFrame, X_train),
                cast(pd.DataFrame, X_val),
                cast(pd.Series, y_train),
                cast(pd.Series, y_val),
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
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
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
            if class_weight:
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
            valid_data = self._create_dataset(X_test, y_test)

            # モデル固有の学習パラメータ取得
            params = self._get_model_params(num_classes, **kwargs)

            # モデル固有の学習実行
            self.model = self._train_internal(
                train_data,
                valid_data,
                params,
                early_stopping_rounds=kwargs.get("early_stopping_rounds", 50),
                **kwargs, # ここに **kwargs を追加
            )

            # 予測と評価
            y_pred_proba = self._get_prediction_proba(valid_data)
            y_pred_class = (
                np.argmax(y_pred_proba, axis=1)
                if num_classes > 2
                else (y_pred_proba > 0.5).astype(int)
            )

            detailed_metrics = evaluate_model_predictions(
                y_test, y_pred_class, y_pred_proba
            )

            # 特徴量重要度
            feature_importance = self.get_feature_importance()

            self.is_trained = True

            logger.info(
                f"{self.ALGORITHM_NAME}モデル学習完了: 精度={detailed_metrics.get('accuracy', 0.0):.4f}"
            )

            result = {
                "algorithm": self.ALGORITHM_NAME,
                "num_classes": num_classes,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
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
        pass

    @abstractmethod
    def _get_model_params(self, num_classes: int, **kwargs) -> Dict[str, Any]:
        """
        モデル固有のパラメータディクショナリを生成します。
        """
        pass

    @abstractmethod
    def _train_internal(
        self,
        train_data: Any,
        valid_data: Any,
        params: Dict[str, Any],
        early_stopping_rounds: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """
        モデル固有の学習プロセスを実行します。
        """
        pass


    @abstractmethod
    def _get_prediction_proba(self, data: Any) -> np.ndarray:
        """
        モデル固有の予測確率を取得します。
        """
        pass

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        sklearn互換の予測メソッド（クラス予測）。
        """
        pass

    @abstractmethod
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        予測確率を取得。
        """
        pass

    @abstractmethod
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        特徴量重要度を取得。
        """
        pass


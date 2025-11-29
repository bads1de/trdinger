"""
単一モデルトレーナー

BaseMLTrainerを継承し、単一の機械学習モデルでのトレーニングを提供します。
LightGBM、XGBoostをサポートします。
"""

import logging
from typing import Any, Dict, cast

import numpy as np
import pandas as pd

from ..base_ml_trainer import BaseMLTrainer
from ..exceptions import MLModelError

logger = logging.getLogger(__name__)


class SingleModelTrainer(BaseMLTrainer):
    """
    単一モデル学習トレーナー

    BaseMLTrainerを継承し、単一の機械学習モデルでのトレーニング機能を提供します。
    アンサンブルを使用せず、指定された単一モデルで学習を行います。
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
    ):
        """
        初期化

        Args:
            model_type: 使用するモデルタイプ（lightgbm, xgboost）
        """
        super().__init__()

        self.model_type = model_type.lower()
        self.single_model = None
        self.last_training_results = None  # 最後の学習結果を保持

        # サポートされているモデルタイプを確認
        supported_models = ["lightgbm", "xgboost"]
        if self.model_type not in supported_models:
            raise MLModelError(
                f"サポートされていないモデルタイプ: {self.model_type}. "
                f"サポートされているモデル: {supported_models}"
            )

        logger.info(f"SingleModelTrainer初期化: model_type={self.model_type}")

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **training_params,
    ) -> Dict[str, Any]:
        """
        単一モデルの学習を実行

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット
            **training_params: 追加の学習パラメータ

        Returns:
            学習結果の辞書
        """
        try:
            logger.info(f"🤖 {self.model_type.upper()}モデルでの学習を開始します")

            # モデルインスタンスを作成
            self.single_model = self._create_model_instance()

            # モデルを学習
            training_result = self.single_model.fit(
                X_train, y_train, eval_set=[(X_test, y_test)], **training_params
            )

            # 学習完了フラグを設定
            self.is_trained = True
            self.feature_columns = list(X_train.columns)

            # BaseMLTrainer用のモデル参照を設定
            self._model = self.single_model.model

            # 結果を整形
            # training_resultはモデル内部のfitメソッドからは返されない（selfを返す）
            # そのため、評価結果を取得する必要がある
            # evaluate_model_predictionsはBaseGradientBoostingModel.fit内部でログ出力されているが
            # ここでも取得して返すのが望ましい
            
            y_pred_proba = self.predict_proba(X_test)
            y_pred = self.predict(X_test)
            
            from ..common.evaluation_utils import evaluate_model_predictions
            
            detailed_metrics = evaluate_model_predictions(
                y_test, y_pred, y_pred_proba
            )

            # 特徴量重要度
            feature_importance = self.single_model.get_feature_importance()

            result = {
                "model_type": self.model_type,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": len(X_train.columns),
                "feature_importance": feature_importance,
                **detailed_metrics,
            }

            # 学習結果を保存（メタデータ用）
            self.last_training_results = result

            logger.info(f"✅ {self.model_type.upper()}モデルの学習が完了しました")
            return result

        except Exception as e:
            logger.error(f"❌ {self.model_type.upper()}モデルの学習に失敗: {e}")
            raise MLModelError(
                f"{self.model_type.upper()}モデルの学習に失敗しました: {e}"
            )

    def _create_model_instance(self):
        """指定されたモデルタイプのインスタンスを作成"""
        try:
            if self.model_type == "lightgbm":
                from ..models.lightgbm import LightGBMModel

                return LightGBMModel()

            elif self.model_type == "xgboost":
                from ..models.xgboost import XGBoostModel

                return XGBoostModel()

            else:
                raise MLModelError(f"未対応のモデルタイプ: {self.model_type}")

        except ImportError as e:
            logger.error(f"{self.model_type.upper()}の依存関係が不足しています: {e}")
            raise MLModelError(
                f"{self.model_type.upper()}の依存関係がインストールされていません。"
                f"必要なライブラリをインストールしてください。"
            )

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        単一モデルで予測を実行（クラス予測）

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測クラスの配列
        """
        if self.single_model is None or not self.single_model.is_trained:
            raise MLModelError("学習済み単一モデルがありません")

        try:
            # 特徴量の順序を学習時と合わせる
            if self.feature_columns:
                features_df = cast(
                    pd.DataFrame, features_df.loc[:, self.feature_columns]
                )

            return self.single_model.predict(features_df)

        except Exception as e:
            logger.error(f"{self.model_type.upper()}モデルの予測エラー: {e}")
            raise MLModelError(
                f"{self.model_type.upper()}モデルの予測に失敗しました: {e}"
            )

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        単一モデルで予測確率を取得

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測確率の配列
        """
        if self.single_model is None or not self.single_model.is_trained:
            raise MLModelError("学習済み単一モデルがありません")

        try:
            # 特徴量の順序を学習時と合わせる
            if self.feature_columns:
                features_df = cast(
                    pd.DataFrame, features_df.loc[:, self.feature_columns]
                )

            return self.single_model.predict_proba(features_df)

        except Exception as e:
            logger.error(f"{self.model_type.upper()}モデルの予測確率取得エラー: {e}")
            raise MLModelError(
                f"{self.model_type.upper()}モデルの予測確率取得に失敗しました: {e}"
            )

    @staticmethod
    def get_available_models() -> list:
        """
        利用可能な単一モデルのリストを取得（Essential Modelsのみ）

        Returns:
            利用可能なモデルタイプのリスト
        """
        available = []
        import importlib.util

        # Essential 2 Models（依存ライブラリベースのモデル）
        essential_models = ["lightgbm", "xgboost"]
        for model in essential_models:
            if importlib.util.find_spec(model):
                available.append(model)

        return available

    def _get_model_to_save(self) -> Any:
        """保存対象のモデルオブジェクトを取得"""
        if self.single_model is None:
            return None
        return self.single_model.model

    def _get_model_specific_metadata(self, model_name: str) -> Dict[str, Any]:
        """モデル固有のメタデータを取得"""
        metadata = {
            "model_type": self.model_type,
            "trainer_type": "single_model",
            "feature_count": (len(self.feature_columns) if self.feature_columns else 0),
        }

        # 学習結果の評価指標をメタデータに追加
        if self.last_training_results:
            # 主要な評価指標を抽出
            performance_metrics = {}
            for key in [
                "accuracy",
                "balanced_accuracy",
                "f1_score",
                "matthews_corrcoef",
                "auc_roc",
                "auc_pr",
                "test_accuracy",
                "test_balanced_accuracy",
                "test_f1_score",
                "test_mcc",
                "test_roc_auc",
                "test_pr_auc",
            ]:
                if key in self.last_training_results:
                    performance_metrics[key] = self.last_training_results[key]

            metadata["performance_metrics"] = performance_metrics

        return metadata

    def load_model(self, model_path: str) -> bool:
        """モデルを読み込み（オーバーライド）"""
        # 親クラスのload_modelを呼び出してモデルをロード
        if not super().load_model(model_path):
            return False

        # メタデータからモデルタイプを復元
        if hasattr(self, "metadata"):
            self.model_type = self.metadata.get("model_type", self.model_type)

        # single_modelを再構築
        try:
            self.single_model = self._create_model_instance()
            # ロードしたモデルをセット
            self.single_model.model = self._model
            self.single_model.is_trained = True
            # 特徴量カラムもセット
            self.single_model.feature_columns = self.feature_columns

            return True

        except Exception as e:
            logger.error(f"モデル読み込み後の再構築に失敗: {e}")
            return False
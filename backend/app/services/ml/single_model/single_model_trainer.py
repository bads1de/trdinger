"""
単一モデルトレーナー

BaseMLTrainerを継承し、単一の機械学習モデルでのトレーニングを提供します。
LightGBM、XGBoost、CatBoost、TabNetをサポートします。
"""

import logging
from typing import Any, Dict, Optional, cast

import numpy as np
import pandas as pd

from ..exceptions import MLModelError
from ..base_ml_trainer import BaseMLTrainer

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
        automl_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初期化

        Args:
            model_type: 使用するモデルタイプ（lightgbm, xgboost, catboost, tabnet）
            automl_config: AutoML設定（オプション）
        """
        super().__init__(automl_config=automl_config)

        self.model_type = model_type.lower()
        self.single_model = None
        self.last_training_results = None  # 最後の学習結果を保持

        # サポートされているモデルタイプを確認
        supported_models = [
            "lightgbm",
            "xgboost",
            "catboost",
            "tabnet",
            "randomforest",
            "extratrees",
            "gradientboosting",
            "adaboost",
            "ridge",
            "naivebayes",
            "knn",
        ]
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
            training_result = self.single_model._train_model_impl(
                X_train, X_test, y_train, y_test, **training_params
            )

            # 学習完了フラグを設定
            self.is_trained = True
            self.feature_columns = list(X_train.columns)

            # 結果を整形
            result = {
                "model_type": self.model_type,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": len(X_train.columns),
                **training_result,
            }

            # 学習結果を保存（save_modelで使用）
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

                return LightGBMModel(automl_config=self.automl_config)

            elif self.model_type == "xgboost":
                from ..models.xgboost import XGBoostModel

                return XGBoostModel(automl_config=self.automl_config)

            elif self.model_type == "catboost":
                from ..models.catboost import CatBoostModel

                return CatBoostModel(automl_config=self.automl_config)

            elif self.model_type == "tabnet":
                from ..models.tabnet import TabNetModel

                return TabNetModel(automl_config=self.automl_config)

            elif self.model_type == "randomforest":
                from ..models.randomforest import RandomForestModel

                return RandomForestModel(automl_config=self.automl_config)

            elif self.model_type == "extratrees":
                from ..models.extratrees import ExtraTreesModel

                return ExtraTreesModel(automl_config=self.automl_config)

            elif self.model_type == "gradientboosting":
                from ..models.gradientboosting import GradientBoostingModel

                return GradientBoostingModel(automl_config=self.automl_config)

            elif self.model_type == "adaboost":
                from ..models.adaboost import AdaBoostModel

                return AdaBoostModel(automl_config=self.automl_config)

            elif self.model_type == "ridge":
                from ..models.ridge import RidgeModel

                return RidgeModel(automl_config=self.automl_config)

            elif self.model_type == "naivebayes":
                from ..models.naivebayes import NaiveBayesModel

                return NaiveBayesModel(automl_config=self.automl_config)

            elif self.model_type == "knn":
                from ..models.knn import KNNModel

                return KNNModel(automl_config=self.automl_config)

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
        単一モデルで予測を実行

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測確率の配列 [下落確率, レンジ確率, 上昇確率]
        """
        if self.single_model is None or not self.single_model.is_trained:
            raise MLModelError("学習済み単一モデルがありません")

        try:
            # 特徴量の順序を学習時と合わせる
            if self.feature_columns:
                features_df = cast(
                    pd.DataFrame, features_df.loc[:, self.feature_columns]
                )

            # 単一モデルで予測確率を取得
            predictions = self.single_model.predict_proba(features_df)

            # 予測確率が3クラス分類であることを確認
            if predictions.ndim == 2 and predictions.shape[1] == 3:
                return predictions
            else:
                raise MLModelError(
                    f"予期しない予測確率の形状: {predictions.shape}. "
                    f"3クラス分類 (down, range, up) の確率が期待されます。"
                )

        except Exception as e:
            logger.error(f"{self.model_type.upper()}モデルの予測エラー: {e}")
            raise MLModelError(
                f"{self.model_type.upper()}モデルの予測に失敗しました: {e}"
            )

    def save_model(
        self, model_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        単一モデルを保存

        Args:
            model_name: モデル名
            metadata: メタデータ（オプション）

        Returns:
            保存されたモデルのパス
        """
        if self.single_model is None or not self.single_model.is_trained:
            raise MLModelError("保存する学習済みモデルがありません")

        try:
            from ..model_manager import model_manager

            # メタデータに単一モデル情報を追加
            final_metadata = metadata or {}
            final_metadata.update(
                {
                    "model_type": self.model_type,
                    "trainer_type": "single_model",
                    "feature_count": (
                        len(self.feature_columns) if self.feature_columns else 0
                    ),
                }
            )

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

                # メタデータに追加
                final_metadata.update(performance_metrics)
                final_metadata["training_samples"] = self.last_training_results.get(
                    "training_samples", 0
                )
                final_metadata["test_samples"] = self.last_training_results.get(
                    "test_samples", 0
                )

                logger.info(
                    f"評価指標をメタデータに追加: {len(performance_metrics)}個の指標"
                )

            # 特徴量重要度をメタデータに追加
            try:
                feature_importance = self.get_feature_importance(top_n=100)
                if feature_importance:
                    final_metadata["feature_importance"] = feature_importance
                    logger.info(
                        f"特徴量重要度をメタデータに追加: {len(feature_importance)}個"
                    )
            except Exception as e:
                logger.warning(f"特徴量重要度の取得に失敗: {e}")

            # 単一モデルを保存
            model_path = model_manager.save_model(
                model=self.single_model.model,
                model_name=model_name,
                metadata=final_metadata,
                scaler=getattr(self.single_model, "scaler", None),
                feature_columns=self.feature_columns,
            )

            if model_path is None:
                raise MLModelError("モデルの保存に失敗しました")

            logger.info(f"単一モデル保存完了: {model_path}")
            return cast(str, model_path)

        except Exception as e:
            logger.error(f"単一モデル保存エラー: {e}")
            raise MLModelError(f"単一モデルの保存に失敗しました: {e}")

    def load_model(self, model_path: str) -> bool:
        """
        単一モデルを読み込み

        Args:
            model_path: モデルファイルパス

        Returns:
            読み込み成功フラグ
        """
        try:
            # モデルインスタンスを作成
            self.single_model = self._create_model_instance()

            # モデルを読み込み
            from ..model_manager import model_manager

            model_data = model_manager.load_model(model_path)

            if model_data is not None:
                # モデルデータをsingle_modelに設定
                self.single_model.model = model_data.get("model")
                self.single_model.scaler = model_data.get("scaler")
                self.feature_columns = model_data.get("feature_columns")

                self.is_trained = True
                logger.info(f"単一モデル読み込み完了: model_type={self.model_type}")
                return True
            else:
                logger.error("単一モデルの読み込みに失敗")
                return False

        except Exception as e:
            logger.error(f"単一モデル読み込みエラー: {e}")
            return False

    @property
    def model(self):
        """学習済みモデルを取得（互換性のため）"""
        return self.single_model.model if self.single_model else None

    @model.setter
    def model(self, value):
        """学習済みモデルを設定（互換性のため）"""
        # BaseMLTrainerとの互換性のため、setterを提供
        # 実際の設定は_train_model_implで行われる

    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得

        Returns:
            モデル情報の辞書
        """
        if self.single_model is None:
            return {
                "model_type": self.model_type,
                "is_trained": False,
                "trainer_type": "single_model",
            }

        return {
            "model_type": self.model_type,
            "is_trained": self.single_model.is_trained,
            "trainer_type": "single_model",
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
        }

    @staticmethod
    def get_available_models() -> list:
        """
        利用可能な単一モデルのリストを取得

        Returns:
            利用可能なモデルタイプのリスト
        """
        available = []
        import importlib.util

        # 依存ライブラリベースのモデル
        libs = ["lightgbm", "xgboost", "catboost", "tabnet"]
        for lib in libs:
            if importlib.util.find_spec(lib):
                available.append(lib)

        # scikit-learnベースのモデル
        sklearn_models = [
            "randomforest",
            "extratrees",
            "gradientboosting",
            "adaboost",
            "ridge",
            "naivebayes",
            "knn",
        ]

        # scikit-learn自体がインストールされているかチェック
        if importlib.util.find_spec("sklearn"):
            available.extend(sklearn_models)

        return available

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Args:
            top_n: 上位N個の特徴量

        Returns:
            特徴量重要度の辞書
        """
        if not self.is_trained or self.single_model is None:
            logger.warning("学習済み単一モデルがありません")
            return {}

        try:
            # 単一モデルから特徴量重要度を取得
            if hasattr(self.single_model, "get_feature_importance"):
                try:
                    return self.single_model.get_feature_importance()
                except TypeError:
                    return self.single_model.get_feature_importance()
            else:
                logger.warning(
                    f"{self.model_type}モデルは特徴量重要度をサポートしていません"
                )
                return {}

        except Exception as e:
            logger.error(f"単一モデル特徴量重要度取得エラー: {e}")
            return {}

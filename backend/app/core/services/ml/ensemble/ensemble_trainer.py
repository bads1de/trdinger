"""
アンサンブル学習トレーナー

BaseMLTrainerを継承し、アンサンブル学習のオーケストレーションを行います。
バギングとスタッキングの両方をサポートし、設定に応じて適切な手法を選択します。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from ..base_ml_trainer import BaseMLTrainer
from .bagging import BaggingEnsemble
from .stacking import StackingEnsemble
from ....utils.unified_error_handler import UnifiedModelError

logger = logging.getLogger(__name__)


class EnsembleTrainer(BaseMLTrainer):
    """
    アンサンブル学習トレーナー

    BaseMLTrainerを継承し、アンサンブル学習機能を提供します。
    設定に応じてバギングまたはスタッキングを実行します。
    """

    def __init__(
        self,
        ensemble_config: Dict[str, Any],
        automl_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初期化

        Args:
            ensemble_config: アンサンブル設定
            automl_config: AutoML設定（オプション）
        """
        super().__init__(automl_config=automl_config)

        self.ensemble_config = ensemble_config
        self.model_type = "EnsembleModel"
        self.ensemble_method = ensemble_config.get("method", "bagging")
        self.ensemble_model = None

        logger.info(f"EnsembleTrainer初期化: method={self.ensemble_method}")

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        アンサンブルモデルで予測を実行

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測確率の配列 [下落確率, レンジ確率, 上昇確率]
        """
        if self.ensemble_model is None or not self.ensemble_model.is_fitted:
            raise UnifiedModelError("学習済みアンサンブルモデルがありません")

        try:
            # アンサンブルモデルは主にLightGBMベースなのでスケーリング不要
            features_scaled = features_df

            # アンサンブルモデルで予測確率を取得
            predictions = self.ensemble_model.predict_proba(features_scaled)

            # 予測確率が3クラス分類であることを確認
            if predictions.ndim == 2 and predictions.shape[1] == 3:
                # 3クラス分類の場合、そのまま返す
                return predictions
            else:
                # 予期しない形状の場合はエラー
                raise UnifiedModelError(
                    f"予期しない予測確率の形状: {predictions.shape}. "
                    f"3クラス分類 (down, range, up) の確率が期待されます。"
                )

            return predictions

        except Exception as e:
            logger.error(f"アンサンブル予測エラー: {e}")
            raise UnifiedModelError(f"アンサンブル予測に失敗しました: {e}")

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **training_params,
    ) -> Dict[str, Any]:
        """
        アンサンブルモデルを学習（BaseMLTrainerの抽象メソッド実装）

        Args:
            X_train: 学習用特徴量
            X_test: テスト用特徴量
            y_train: 学習用ターゲット
            y_test: テスト用ターゲット
            **training_params: 追加の学習パラメータ

        Returns:
            学習結果
        """
        try:
            logger.info(f"アンサンブル学習開始: method={self.ensemble_method}")

            # アンサンブル手法に応じてモデルを作成
            if self.ensemble_method.lower() == "bagging":
                # バギング設定を準備
                bagging_config = self.ensemble_config.get("bagging_params", {})
                # base_model_typeが指定されていない場合のみデフォルトを設定
                if "base_model_type" not in bagging_config:
                    bagging_config["base_model_type"] = "lightgbm"
                bagging_config.update(
                    {
                        "random_state": training_params.get("random_state", 42),
                    }
                )

                self.ensemble_model = BaggingEnsemble(
                    config=bagging_config, automl_config=self.automl_config
                )

            elif self.ensemble_method.lower() == "stacking":
                # スタッキング設定を準備
                stacking_config = self.ensemble_config.get("stacking_params", {})
                stacking_config.update(
                    {"random_state": training_params.get("random_state", 42)}
                )

                self.ensemble_model = StackingEnsemble(
                    config=stacking_config, automl_config=self.automl_config
                )

            else:
                raise UnifiedModelError(
                    f"サポートされていないアンサンブル手法: {self.ensemble_method}"
                )

            # アンサンブルモデルを学習
            training_result = self.ensemble_model.fit(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
            )

            # 予測と評価
            y_pred_proba = self.ensemble_model.predict_proba(X_test)
            y_pred = self.ensemble_model.predict(X_test)

            # 予測確率が3クラス分類であることを確認
            if y_pred_proba.ndim != 2 or y_pred_proba.shape[1] != 3:
                logger.warning(
                    f"予測確率の形状が期待と異なります: {y_pred_proba.shape}"
                )
                # 3クラス分類でない場合は評価をスキップ

            # 詳細な評価指標を計算
            from ....utils.metrics_calculator import calculate_detailed_metrics

            detailed_metrics = calculate_detailed_metrics(y_test, y_pred, y_pred_proba)

            # 分類レポート
            from sklearn.metrics import classification_report

            class_report = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )

            # 特徴量重要度
            feature_importance = self.ensemble_model.get_feature_importance()

            # AutoML特徴量分析
            automl_feature_analysis = None
            if self.use_automl and hasattr(self, "automl_analyzer"):
                try:
                    automl_feature_analysis = self.automl_analyzer.analyze_features(
                        X_train, y_train
                    )
                except Exception as e:
                    logger.warning(f"AutoML特徴量分析でエラー: {e}")

            # 結果をまとめ
            result = {
                **detailed_metrics,
                **training_result,
                "classification_report": class_report,
                "feature_importance": feature_importance,
                "automl_feature_analysis": automl_feature_analysis,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "model_type": self.model_type,
                "ensemble_method": self.ensemble_method,
                "automl_enabled": self.use_automl,
            }

            # 学習完了フラグを設定
            self.is_trained = True

            # 精度を取得
            accuracy = detailed_metrics.get("accuracy", 0.0)
            logger.info(
                f"アンサンブルモデル学習完了: method={self.ensemble_method}, 精度={accuracy:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"アンサンブルモデル学習エラー: {e}")
            raise UnifiedModelError(f"アンサンブルモデル学習に失敗しました: {e}")

    def save_model(
        self, model_path: str, model_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        アンサンブルモデルを保存

        Args:
            model_path: 保存先パス
            model_metadata: モデルメタデータ（オプション）

        Returns:
            保存成功フラグ
        """
        try:
            if self.ensemble_model is None:
                logger.warning("保存するアンサンブルモデルがありません")
                return False

            # アンサンブルモデルを保存
            saved_paths = self.ensemble_model.save_models(model_path)

            # 追加のメタデータを保存
            import joblib

            metadata = {
                "ensemble_config": self.ensemble_config,
                "automl_config": self.automl_config,
                "model_type": self.model_type,
                "ensemble_method": self.ensemble_method,
                "feature_columns": self.feature_columns,
                "scaler": self.scaler,
                "is_trained": self.is_trained,
            }

            # 追加のモデルメタデータがある場合は統合
            if model_metadata:
                metadata.update(model_metadata)

            metadata_path = f"{model_path}_ensemble_metadata.pkl"
            joblib.dump(metadata, metadata_path)

            logger.info(f"アンサンブルモデル保存完了: {len(saved_paths)} ファイル")
            return True

        except Exception as e:
            logger.error(f"アンサンブルモデル保存エラー: {e}")
            return False

    def load_model(self, model_path: str) -> bool:
        """
        アンサンブルモデルを読み込み

        Args:
            model_path: 読み込み元パス

        Returns:
            読み込み成功フラグ
        """
        try:
            import joblib
            import os

            # メタデータを読み込み
            metadata_path = f"{model_path}_ensemble_metadata.pkl"
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.ensemble_config = metadata["ensemble_config"]
                self.automl_config = metadata["automl_config"]
                self.model_type = metadata["model_type"]
                self.ensemble_method = metadata["ensemble_method"]
                self.feature_columns = metadata["feature_columns"]
                self.scaler = metadata["scaler"]
                self.is_trained = metadata["is_trained"]

            # アンサンブルモデルを作成
            if self.ensemble_method.lower() == "bagging":
                bagging_config = self.ensemble_config.get("bagging_params", {})
                self.ensemble_model = BaggingEnsemble(
                    config=bagging_config, automl_config=self.automl_config
                )
            elif self.ensemble_method.lower() == "stacking":
                stacking_config = self.ensemble_config.get("stacking_params", {})
                self.ensemble_model = StackingEnsemble(
                    config=stacking_config, automl_config=self.automl_config
                )
            else:
                raise UnifiedModelError(
                    f"サポートされていないアンサンブル手法: {self.ensemble_method}"
                )

            # アンサンブルモデルを読み込み
            success = self.ensemble_model.load_models(model_path)

            if success:
                logger.info(
                    f"アンサンブルモデル読み込み完了: method={self.ensemble_method}"
                )
            else:
                logger.error("アンサンブルモデルの読み込みに失敗")

            return success

        except Exception as e:
            logger.error(f"アンサンブルモデル読み込みエラー: {e}")
            return False

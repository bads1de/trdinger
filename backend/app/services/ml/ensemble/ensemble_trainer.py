"""
アンサンブル学習トレーナー

BaseMLTrainerを継承し、アンサンブル学習のオーケストレーションを行います。
バギングとスタッキングの両方をサポートし、設定に応じて適切な手法を選択します。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ....utils.unified_error_handler import UnifiedModelError
from ..base_ml_trainer import BaseMLTrainer
from .bagging import BaggingEnsemble
from .stacking import StackingEnsemble

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

    def _extract_optimized_parameters(
        self, training_params: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        最適化されたパラメータを各モデル・手法別に分離

        Args:
            training_params: 最適化されたパラメータを含む学習パラメータ

        Returns:
            分離されたパラメータ辞書
        """
        optimized_params = {
            "base_models": {
                "lightgbm": {},
                "xgboost": {},
                "randomforest": {},
                "catboost": {},
                "tabnet": {},
            },
            "bagging": {},
            "stacking": {},
        }

        for param_name, param_value in training_params.items():
            # LightGBMパラメータ
            if param_name.startswith("lgb_"):
                clean_name = param_name.replace("lgb_", "")
                optimized_params["base_models"]["lightgbm"][clean_name] = param_value

            # XGBoostパラメータ
            elif param_name.startswith("xgb_"):
                clean_name = param_name.replace("xgb_", "")
                optimized_params["base_models"]["xgboost"][clean_name] = param_value

            # RandomForestパラメータ
            elif param_name.startswith("rf_"):
                clean_name = param_name.replace("rf_", "")
                optimized_params["base_models"]["randomforest"][
                    clean_name
                ] = param_value

            # CatBoostパラメータ
            elif param_name.startswith("cat_"):
                clean_name = param_name.replace("cat_", "")
                optimized_params["base_models"]["catboost"][clean_name] = param_value

            # TabNetパラメータ
            elif param_name.startswith("tab_"):
                clean_name = param_name.replace("tab_", "")
                optimized_params["base_models"]["tabnet"][clean_name] = param_value

            # バギングパラメータ
            elif param_name.startswith("bagging_"):
                clean_name = param_name.replace("bagging_", "")
                optimized_params["bagging"][clean_name] = param_value

            # スタッキングパラメータ
            elif param_name.startswith("stacking_"):
                clean_name = param_name.replace("stacking_", "")
                if clean_name.startswith("meta_"):
                    # メタモデルパラメータ
                    meta_param = clean_name.replace("meta_", "")
                    if "meta_model_params" not in optimized_params["stacking"]:
                        optimized_params["stacking"]["meta_model_params"] = {}
                    optimized_params["stacking"]["meta_model_params"][
                        meta_param
                    ] = param_value
                else:
                    optimized_params["stacking"][clean_name] = param_value

        return optimized_params

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

            # ハイパーパラメータ最適化からのパラメータを分離
            optimized_params = self._extract_optimized_parameters(training_params)

            # アンサンブル手法に応じてモデルを作成
            if self.ensemble_method.lower() == "bagging":
                # バギング設定を準備
                bagging_config = self.ensemble_config.get("bagging_params", {})
                # base_model_typeが指定されていない場合のみデフォルトを設定
                if "base_model_type" not in bagging_config:
                    bagging_config["base_model_type"] = "lightgbm"

                # 最適化されたバギングパラメータを適用
                if "bagging" in optimized_params:
                    bagging_config.update(optimized_params["bagging"])

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

                # 最適化されたスタッキングパラメータを適用
                if "stacking" in optimized_params:
                    stacking_config.update(optimized_params["stacking"])

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

            # 最適化されたベースモデルパラメータをアンサンブルモデルに渡す
            base_model_params = optimized_params.get("base_models", {})

            # アンサンブルモデルを学習
            training_result = self.ensemble_model.fit(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                base_model_params=base_model_params,
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

            # 統一された評価指標計算器を使用
            from ..evaluation.enhanced_metrics import (
                EnhancedMetricsCalculator,
                MetricsConfig,
            )

            config = MetricsConfig(
                include_balanced_accuracy=True,
                include_pr_auc=True,
                include_roc_auc=True,
                include_confusion_matrix=True,
                include_classification_report=True,
                average_method="weighted",
                zero_division=0,
            )

            metrics_calculator = EnhancedMetricsCalculator(config)
            detailed_metrics = metrics_calculator.calculate_comprehensive_metrics(
                y_test, y_pred, y_pred_proba
            )

            # 分類レポート
            from sklearn.metrics import classification_report

            class_report = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )

            # 特徴量重要度
            feature_importance = self.ensemble_model.get_feature_importance()
            if not feature_importance:
                logger.warning(
                    "アンサンブルモデルから特徴量重要度を取得できませんでした"
                )
                feature_importance = {}
            else:
                logger.info(f"特徴量重要度を取得: {len(feature_importance)}個")

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
            from datetime import datetime

            import joblib

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            algorithm_name = getattr(self.ensemble_model, "best_algorithm", "unknown")

            metadata = {
                "ensemble_config": self.ensemble_config,
                "automl_config": self.automl_config,
                "model_type": self.model_type,
                "ensemble_method": self.ensemble_method,
                "feature_columns": self.feature_columns,
                "scaler": self.scaler,
                "is_trained": self.is_trained,
                "best_algorithm": algorithm_name,
                "best_model_score": getattr(
                    self.ensemble_model, "best_model_score", None
                ),
                "timestamp": timestamp,
            }

            # 追加のモデルメタデータがある場合は統合
            if model_metadata:
                metadata.update(model_metadata)

            metadata_path = f"{model_path}_ensemble_metadata_{timestamp}.pkl"
            joblib.dump(metadata, metadata_path)

            logger.info(
                f"アンサンブルモデル保存完了: {len(saved_paths)} ファイル, 最高性能アルゴリズム: {algorithm_name}"
            )
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
            # メタデータを読み込み（タイムスタンプ付きファイルに対応）
            import glob
            import os

            import joblib

            metadata_patterns = [
                f"{model_path}_ensemble_metadata_*.pkl",  # 新形式
                f"{model_path}_ensemble_metadata.pkl",  # 旧形式
            ]

            metadata_path = None
            for pattern in metadata_patterns:
                files = glob.glob(pattern)
                if files:
                    metadata_path = sorted(files)[-1]  # 最新のファイルを選択
                    break

            if metadata_path and os.path.exists(metadata_path):
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

    def _cleanup_models(self, level):
        """
        EnsembleTrainer固有のモデルクリーンアップ

        Args:
            level: クリーンアップレベル
        """
        try:
            # アンサンブルモデルのクリーンアップ
            if self.ensemble_model is not None:
                try:
                    if hasattr(self.ensemble_model, "cleanup_resources"):
                        self.ensemble_model.cleanup_resources()
                    elif hasattr(self.ensemble_model, "cleanup"):
                        self.ensemble_model.cleanup()

                    # アンサンブルモデル自体をクリア
                    self.ensemble_model = None
                    logger.debug("アンサンブルモデルをクリアしました")

                except Exception as ensemble_error:
                    logger.warning(
                        f"アンサンブルモデルクリーンアップ警告: {ensemble_error}"
                    )

            # 親クラスのモデルクリーンアップを呼び出し
            super()._cleanup_models(level)

        except Exception as e:
            logger.warning(f"EnsembleTrainerモデルクリーンアップエラー: {e}")
            # エラーが発生してもクリーンアップは続行
            self.ensemble_model = None

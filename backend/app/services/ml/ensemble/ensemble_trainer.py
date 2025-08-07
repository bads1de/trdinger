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
        self, model_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        アンサンブルモデルを統一されたモデル管理システムで保存

        Args:
            model_name: モデル名
            metadata: モデルメタデータ（オプション）

        Returns:
            保存されたモデルのパス
        """
        if self.ensemble_model is None or not self.is_trained:
            raise UnifiedModelError("保存する学習済みアンサンブルモデルがありません")

        try:
            from ..model_manager import model_manager

            # アンサンブル用メタデータを準備
            algorithm_name = getattr(self.ensemble_model, "best_algorithm", "unknown")

            final_metadata = metadata or {}
            final_metadata.update(
                {
                    "model_type": algorithm_name,  # 最高性能アルゴリズム名を使用
                    "trainer_type": "ensemble",
                    "ensemble_method": self.ensemble_method,
                    "feature_count": (
                        len(self.feature_columns) if self.feature_columns else 0
                    ),
                    "best_algorithm": algorithm_name,
                    "best_model_score": getattr(
                        self.ensemble_model, "best_model_score", None
                    ),
                    "selected_model_only": True,
                    "ensemble_config": self.ensemble_config,
                    "automl_config": self.automl_config,
                }
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

            # 最高性能モデル1つのみを保存
            if len(self.ensemble_model.base_models) == 1:
                best_model = self.ensemble_model.base_models[0]

                # 統一されたモデル保存を使用
                model_path = model_manager.save_model(
                    model=best_model,
                    model_name=model_name,
                    metadata=final_metadata,
                    scaler=self.scaler,
                    feature_columns=self.feature_columns,
                )

                logger.info(
                    f"アンサンブル最高性能モデル保存完了: {model_path} (アルゴリズム: {algorithm_name})"
                )
                return model_path
            else:
                raise UnifiedModelError(
                    "アンサンブルモデルが最高性能モデル1つのみを保持していません"
                )

        except Exception as e:
            logger.error(f"アンサンブルモデル保存エラー: {e}")
            raise UnifiedModelError(f"アンサンブルモデルの保存に失敗しました: {e}")

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
            import warnings

            import joblib
            from sklearn.exceptions import InconsistentVersionWarning

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
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InconsistentVersionWarning)
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

    def get_feature_importance(self, top_n: int = 100) -> Dict[str, float]:
        """
        アンサンブルモデルから特徴量重要度を取得

        Args:
            top_n: 上位N個の特徴量

        Returns:
            特徴量重要度の辞書
        """
        if not self.is_trained or not self.ensemble_model:
            logger.warning("学習済みアンサンブルモデルがありません")
            return {}

        try:
            # アンサンブルモデルから特徴量重要度を取得
            feature_importance = self.ensemble_model.get_feature_importance()
            if not feature_importance:
                logger.warning(
                    "アンサンブルモデルから特徴量重要度を取得できませんでした"
                )
                return {}

            # 上位N個を取得
            sorted_importance = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:top_n]

            logger.info(
                f"アンサンブルから特徴量重要度を取得: {len(sorted_importance)}個"
            )
            return dict(sorted_importance)

        except Exception as e:
            logger.error(f"アンサンブル特徴量重要度取得エラー: {e}")
            return {}

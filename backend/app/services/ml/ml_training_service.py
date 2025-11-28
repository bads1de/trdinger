"""
ML学習サービス

MLモデルの学習・評価・保存を取り扱うサービス層です。
内部実装の詳細や特定の最適化手法の説明はDocstringに含めず、
サービスの役割（学習ワークフローの調整と結果の提供）に限定して記述します。
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ...utils.error_handler import safe_ml_operation
from ..optimization.optimization_service import (
    OptimizationService,
    OptimizationSettings,
)
from .base_ml_trainer import BaseMLTrainer
from .common.base_resource_manager import BaseResourceManager, CleanupLevel
from .config import ml_config
from .ensemble.ensemble_trainer import EnsembleTrainer
from .single_model.single_model_trainer import SingleModelTrainer

logger = logging.getLogger(__name__)


class MLTrainingService(BaseResourceManager):
    """
    ML学習サービス

    BaseMLTrainerを使用してMLモデルの学習、評価、保存を専門的に行うサービス。
    コードの重複を解消し、保守性を向上させます。
    """

    def __init__(
        self,
        trainer_type: str = "ensemble",
        ensemble_config: Optional[Dict[str, Any]] = None,
        single_model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初期化

        Args:
            trainer_type: 使用するトレーナーのタイプ（'ensemble' または 'single'）
            ensemble_config: アンサンブル設定（辞書形式）
            single_model_config: 単一モデル設定（辞書形式）
        """
        # BaseResourceManagerの初期化
        super().__init__()

        self.config = ml_config
        self.ensemble_config = ensemble_config
        self.single_model_config = single_model_config
        self.optimization_service = OptimizationService()

        # 統合されたトレーナー設定を作成
        trainer_config = self._create_trainer_config(
            trainer_type, ensemble_config, single_model_config
        )

        # トレーナーを選択して初期化
        if trainer_type.lower() == "single":
            model_type = trainer_config.get("model_type", "lightgbm")
            # 明示的に SingleModelTrainer を使用
            self.trainer = SingleModelTrainer(model_type=model_type)
        elif trainer_type.lower() == "ensemble":
            # アンサンブル設定を取得
            ens_config = trainer_config.get("ensemble_config", ensemble_config or {})
            self.trainer = EnsembleTrainer(ensemble_config=ens_config)
        else:
            raise ValueError(f"未対応のトレーナータイプ: {trainer_type}")

        self.trainer_type = trainer_type

        if trainer_type == "single" and single_model_config:
            logger.info(f"単一モデル設定: {single_model_config}")

    def _create_trainer_config(
        self,
        trainer_type: str,
        ensemble_config: Optional[Dict[str, Any]],
        single_model_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        統合されたトレーナー設定を作成

        Args:
            trainer_type: トレーナータイプ
            ensemble_config: アンサンブル設定
            single_model_config: 単一モデル設定

        Returns:
            トレーナー設定辞書
        """
        if trainer_type.lower() == "ensemble":
            # アンサンブル設定のデフォルト値（スタッキング）
            default_ensemble_config = {
                "method": "stacking",
                "stacking_params": {
                    "base_models": ["lightgbm", "xgboost"],
                    "meta_model": "lightgbm",
                    "cv_folds": 5,
                    "use_probas": True,
                    "random_state": 42,
                },
            }

            # 設定をマージ
            final_ensemble_config = default_ensemble_config.copy()
            if ensemble_config:
                final_ensemble_config.update(ensemble_config)

            return {
                "type": "ensemble",
                "model_type": final_ensemble_config.get("method", "stacking"),
                "ensemble_config": final_ensemble_config,
            }

        elif trainer_type.lower() == "single":
            # 単一モデル設定のデフォルト値
            model_type = "lightgbm"
            if single_model_config and "model_type" in single_model_config:
                model_type = single_model_config["model_type"]

            return {
                "type": "single",
                "model_type": model_type,
                "model_params": single_model_config,
            }

        else:
            raise ValueError(
                f"サポートされていないトレーナータイプ: {trainer_type}。"
                f"サポートされているタイプ: 'ensemble', 'single'"
            )

    @staticmethod
    def get_available_single_models() -> List[str]:
        """利用可能な単一モデルのリストを取得"""
        return SingleModelTrainer.get_available_models()

    @staticmethod
    def determine_trainer_type(ensemble_config: Optional[Dict[str, Any]]) -> str:
        """
        アンサンブル設定に基づいてトレーナータイプを決定

        Args:
            ensemble_config: アンサンブル設定

        Returns:
            トレーナータイプ（'ensemble' または 'single'）
        """
        if ensemble_config and ensemble_config.get("enabled", True) is False:
            return "single"
        return "ensemble"

    def train_model(
        self,
        training_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        save_model: bool = True,
        model_name: Optional[str] = None,
        optimization_settings: Optional[OptimizationSettings] = None,
        **training_params,
    ) -> Dict[str, Any]:
        """
        MLモデルを学習（最適化機能付き）

        Args:
            training_data: 学習用OHLCVデータ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            save_model: モデルを保存するか
            model_name: モデル名（オプション）
            optimization_settings: 最適化設定（オプション）
            **training_params: 追加の学習パラメータ
                - use_cross_validation: クロスバリデーションを使用（デフォルト: False）
                - cv_splits: CV分割数（デフォルト: ml_config.training.CROSS_VALIDATION_FOLDS）

        Returns:
            学習結果の辞書

        Raises:
            MLDataError: データが無効な場合
            MLModelError: 学習に失敗した場合
            ValueError: 無効なパラメータ組み合わせの場合
        """
        # TimeSeriesSplit関連パラメータをml_configから設定（未指定の場合）
        training_params = self._prepare_training_params(training_params)

        trainer = self.trainer

        # 最適化が有効な場合は最適化ワークフローを実行
        if optimization_settings and optimization_settings.enabled:
            # 最適化を実行
            optimization_result = self.optimization_service.optimize_parameters(
                trainer=trainer,
                training_data=training_data,
                optimization_settings=optimization_settings,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                model_name=model_name,
                **training_params,
            )

            # 最適化されたパラメータで最終モデルを学習
            final_training_params = {
                **training_params,
                **optimization_result["best_params"],
            }

            final_result = trainer.train_model(
                training_data=training_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                save_model=save_model,
                model_name=model_name,
                **final_training_params,
            )

            # 最適化情報を結果に追加
            final_result["optimization_result"] = optimization_result
            return final_result

        else:
            # 通常のトレーニング
            return trainer.train_model(
                training_data=training_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                save_model=save_model,
                model_name=model_name,
                **training_params,
            )

    def _prepare_training_params(
        self, training_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        training_paramsを準備

        Args:
            training_params: 元のtraining_params

        Returns:
            準備されたtraining_params

        Raises:
            ValueError: 無効なパラメータ組み合わせの場合
        """
        # パラメータのコピーを作成
        params = training_params.copy()

        # クロスバリデーション利用時のパラメータ
        use_cross_validation = params.get("use_cross_validation", False)
        if use_cross_validation:
            # CV分割数（ml_configから取得、training_paramsで上書き可能）
            if "cv_splits" not in params:
                params["cv_splits"] = self.config.training.CROSS_VALIDATION_FOLDS

            # パラメータバリデーション
            cv_splits = params.get("cv_splits")
            if cv_splits is not None and cv_splits < 2:
                raise ValueError(f"cv_splitsは2以上である必要があります: {cv_splits}")

            logger.info(f"CV設定: " f"cv_splits={cv_splits}")

        return params

    def evaluate_model(
        self,
        test_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        学習済みモデルを評価

        Args:
            test_data: テスト用OHLCVデータ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）

        Returns:
            評価結果の辞書
        """
        # BaseMLTrainerに委譲
        return self.trainer.evaluate_model(
            test_data=test_data,
            funding_rate_data=funding_rate_data,
            open_interest_data=open_interest_data,
        )

    def get_training_status(self) -> Dict[str, Any]:
        """
        学習状態を取得

        Returns:
            学習状態の辞書
        """
        # トレーナーから基本情報を取得
        if hasattr(self.trainer, "get_model_info"):
            model_info = self.trainer.get_model_info()
            model_info["trainer_type"] = self.trainer_type
            return model_info
        else:
            return {
                "is_trained": self.trainer.is_trained,
                "feature_columns": self.trainer.feature_columns,
                "feature_count": (
                    len(self.trainer.feature_columns)
                    if self.trainer.feature_columns
                    else 0
                ),
                "model_type": (
                    type(self.trainer.model).__name__ if self.trainer.model else None
                ),
                "trainer_type": self.trainer_type,
            }

    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        予測を実行

        Args:
            features_df: 特徴量DataFrame

        Returns:
            予測結果
        """
        predictions = self.trainer.predict(features_df)
        return {
            "predictions": predictions,
            "model_type": self.trainer_type,
            "feature_count": (
                len(self.trainer.feature_columns) if self.trainer.feature_columns else 0
            ),
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Returns:
            特徴量重要度の辞書
        """
        if hasattr(self.trainer, "get_feature_importance"):
            return self.trainer.get_feature_importance()
        else:
            return {}

    @safe_ml_operation(
        default_return=False, context="モデル読み込みでエラーが発生しました"
    )
    def load_model(self, model_path: str) -> bool:
        """
        学習済みモデルを読み込み

        Args:
            model_path: モデルファイルパス

        Returns:
            読み込み成功フラグ
        """
        return self.trainer.load_model(model_path)

    def generate_signals(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        予測信号を生成

        Args:
            features: 特徴量DataFrame

        Returns:
            予測確率の辞書 {"up": float, "down": float, "range": float}
        """
        if not self.trainer:
            logger.warning("トレーナーが初期化されていません")
            return self.config.prediction.get_default_predictions()

        # BaseMLTrainer.predict_signal に委譲
        return self.trainer.predict_signal(features)

    def _cleanup_temporary_files(self, level: CleanupLevel):
        """一時ファイルのクリーンアップ"""
        # MLTrainingServiceでは特に一時ファイルは作成しないため、パス
        pass

    def _cleanup_cache(self, level: CleanupLevel):
        """キャッシュのクリーンアップ"""
        # MLTrainingServiceでは特にキャッシュは管理しないため、パス
        pass

    def _cleanup_models(self, level: CleanupLevel):
        """モデルオブジェクトのクリーンアップ"""
        try:
            # トレーナーのクリーンアップ
            if hasattr(self, "trainer") and self.trainer:
                if hasattr(self.trainer, "cleanup_resources"):
                    self.trainer.cleanup_resources(level)
                    logger.debug("トレーナーをクリーンアップしました")

            # 最適化サービスのクリーンアップ
            if hasattr(self, "optimization_service"):
                self.optimization_service.cleanup()

        except Exception as e:
            logger.warning(f"MLTrainingServiceモデルクリーンアップエラー: {e}")


# グローバルインスタンス（デフォルトはアンサンブル）
ml_training_service = MLTrainingService(trainer_type="ensemble")

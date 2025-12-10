"""
ML学習サービス

MLモデルの学習・評価・保存を取り扱うサービス層です。
内部実装の詳細や特定の最適化手法の説明はDocstringに含めず、
サービスの役割（学習ワークフローの調整と結果の提供）に限定して記述します。
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from ...utils.error_handler import safe_ml_operation
from app.services.ml.optimization.optimization_service import (
    OptimizationService,
    OptimizationSettings,
)

from .common.base_resource_manager import BaseResourceManager, CleanupLevel
from ...config.unified_config import unified_config
from .ensemble.ensemble_trainer import EnsembleTrainer

logger = logging.getLogger(__name__)


class MLTrainingService(BaseResourceManager):
    """
    ML学習サービス（統一トレーナー対応）

    EnsembleTrainerを使用してMLモデル（単一・アンサンブル両対応）の学習、評価、保存を行うサービス。
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
            trainer_type: 使用するトレーナーのタイプ（'ensemble' または 'single'）。
                この引数は後方互換のために残されていますが、内部では常にEnsembleTrainerが使用されます。
            ensemble_config: アンサンブル設定（辞書形式）
            single_model_config: 単一モデル設定（辞書形式）
                ※後方互換のため残しているが、ensemble_configに変換される
        """
        # BaseResourceManagerの初期化
        super().__init__()

        self.optimization_service = OptimizationService()

        # 統合されたトレーナー設定を作成
        final_config = self._create_unified_config(
            trainer_type, ensemble_config, single_model_config
        )

        # 常にEnsembleTrainerを使用（単一モデルもサポート）
        self.trainer = EnsembleTrainer(ensemble_config=final_config)
        self.trainer_type = trainer_type  # 後方互換のため保持

        mode = "単一モデル" if self.trainer.is_single_model else "アンサンブル"
        logger.info(f"MLTrainingService初期化: mode={mode}, config={final_config}")

    @property
    def config(self):
        """現在の統一ML設定を取得"""
        return unified_config.ml

    def _create_unified_config(
        self,
        trainer_type: str,
        ensemble_config: Optional[Dict[str, Any]],
        single_model_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        統一されたトレーナー設定を作成（EnsembleTrainer用）

        Args:
            trainer_type: トレーナータイプ
            ensemble_config: アンサンブル設定
            single_model_config: 単一モデル設定

        Returns:
            EnsembleTrainer用の設定辞書
        """
        if trainer_type.lower() == "single":
            # 単一モデルをEnsembleTrainer形式に変換
            model_type = "lightgbm"
            config_copy = {}

            if single_model_config:
                config_copy = single_model_config.copy()
                if "model_type" in config_copy:
                    model_type = config_copy.pop("model_type")

            # 単一モデルもStackingEnsemble（モデル数1）として扱う
            # これにより、統一されたインターフェースで学習・評価が可能
            unified_conf = {
                "method": "stacking",
                "models": [model_type],
            }
            # 残りの設定をマージ（必要に応じて）
            unified_conf.update(config_copy)

            return unified_conf

        elif trainer_type.lower() == "ensemble":
            # アンサンブル設定のデフォルト値 (unified_configから取得)
            ensemble_settings = self.config.ensemble

            default_config = {
                "method": ensemble_settings.default_method,
                "models": ensemble_settings.algorithms,
                "stacking_params": {
                    "meta_model": ensemble_settings.stacking_meta_model,
                    "cv_folds": ensemble_settings.stacking_cv_folds,
                    "stack_method": ensemble_settings.stacking_stack_method,
                    "n_jobs": ensemble_settings.stacking_n_jobs,
                    "passthrough": ensemble_settings.stacking_passthrough,
                    "use_probas": True,
                    "random_state": 42,
                },
            }

            # 設定をマージ
            final_config = default_config.copy()
            if ensemble_config:
                final_config.update(ensemble_config)

            return final_config

        else:
            raise ValueError(
                f"サポートされていないトレーナータイプ: {trainer_type}。"
                f"サポートされているタイプ: 'ensemble', 'single'"
            )

    @staticmethod
    def get_available_single_models() -> List[str]:
        """利用可能な単一モデルのリストを取得します。"""
        # unified_configから取得
        return unified_config.ml.ensemble.algorithms

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
                - cv_splits: CV分割数（デフォルト: config.training.cv_folds）

        Returns:
            学習結果の辞書

        Raises:
            MLDataError: データが無効な場合
            MLModelError: 学習に失敗した場合
            ValueError: 無効なパラメータ組み合わせの場合
        """
        # TimeSeriesSplit関連パラメータを設定（未指定の場合）
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
            # CV分割数（configから取得、training_paramsで上書き可能）
            if "cv_splits" not in params:
                params["cv_splits"] = self.config.training.cv_folds

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

    def get_current_model_path(self) -> Optional[str]:
        """
        現在ロードされているモデルのパスを取得

        Returns:
            モデルパス（ロードされていない場合はNone）
        """
        if hasattr(self.trainer, "current_model_path"):
            return self.trainer.current_model_path
        return None

    def get_current_model_info(self) -> Optional[Dict[str, Any]]:
        """
        現在ロードされているモデルのメタデータを取得

        Returns:
            モデルメタデータ（ロードされていない場合はNone）
        """
        if hasattr(self.trainer, "current_model_metadata"):
            return self.trainer.current_model_metadata
        return None

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
        予測信号を生成します。

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

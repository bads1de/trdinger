"""
MLトレーニングサービス

MLモデルのトレーニングフローを制御し、
データの準備、特徴量エンジニアリング、学習、評価、保存を一括管理します。
バックグラウンドタスクとしての実行管理や進捗管理機能も備えています。
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sqlalchemy.orm import Session

from app.services.ml.optimization.optimization_service import (
    OptimizationService,
    OptimizationSettings,
)
from app.services.ml.orchestration.bg_task_orchestration_service import (
    background_task_manager,
)
from app.utils.error_handler import safe_ml_operation
from app.utils.response import api_response
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository

from app.services.ml.models.model_manager import model_manager
from ....config.unified_config import unified_config
from ..common.base_resource_manager import BaseResourceManager, CleanupLevel
from ..common.config import get_default_ensemble_config, get_default_single_model_config
from ..ensemble.ensemble_trainer import EnsembleTrainer
import os

logger = logging.getLogger(__name__)

# --- モデル情報取得ユーティリティ (旧orchestration_utilsより統合) ---

def load_model_metadata_safely(model_path: str) -> Optional[Dict[str, Any]]:
    """モデルファイルからメタデータを安全に読み込む"""
    try:
        metadata_data = model_manager.load_metadata_only(model_path)
        if not metadata_data or "metadata" not in metadata_data:
            return None
        return metadata_data
    except Exception as e:
        logger.warning(f"モデルメタデータ読み込みエラー {model_path}: {e}")
        return None


def get_latest_model_with_info(model_name_pattern: str = "*") -> Optional[Dict[str, Any]]:
    """最新モデルの情報とメトリクスを取得"""
    try:
        latest_model = model_manager.get_latest_model(model_name_pattern)
        if not latest_model or not os.path.exists(latest_model):
            return None

        model_data = load_model_metadata_safely(latest_model)
        if not model_data:
            return None

        metadata = model_data["metadata"]
        metrics = model_manager.extract_model_performance_metrics(latest_model, metadata=metadata)
        file_info = {
            "size_mb": os.path.getsize(latest_model) / (1024 * 1024),
            "modified_at": datetime.fromtimestamp(os.path.getmtime(latest_model)),
        }
        return {"path": latest_model, "metadata": metadata, "metrics": metrics, "file_info": file_info}
    except Exception as e:
        logger.warning(f"最新モデル情報取得エラー: {e}")
        return None


def get_model_info_with_defaults(model_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """モデル情報にデフォルト値を適用"""
    from ..common.evaluation import get_default_metrics
    if not model_info:
        return {
            **get_default_metrics(), "model_type": "No Model",
            "feature_count": 0, "training_samples": 0,
            "last_updated": "未学習", "file_size_mb": 0.0
        }

    m, meta, f = model_info.get("metrics", get_default_metrics()), model_info["metadata"], model_info["file_info"]
    return {
        **m,
        "model_type": meta.get("model_type", "Unknown"),
        "feature_count": meta.get("feature_count", 0),
        "training_samples": meta.get("training_samples", 0),
        "test_samples": meta.get("test_samples", 0),
        "last_updated": f["modified_at"].isoformat(),
        "file_size_mb": f["size_mb"],
        "num_classes": meta.get("num_classes", 2),
        "best_iteration": meta.get("best_iteration", 0),
        "train_test_split": meta.get("train_test_split", 0.8),
        "random_state": meta.get("random_state", 42),
        "feature_importance": meta.get("feature_importance", {}),
        "classification_report": meta.get("classification_report", {}),
    }


# グローバルトレーニング状態管理
training_status = {
    "is_training": False,
    "progress": 0,
    "status": "idle",
    "message": "待機中",
    "start_time": None,
    "end_time": None,
    "model_info": None,
    "error": None,
}


class MLTrainingService(BaseResourceManager):
    """
    MLモデルのトレーニングフローを管理するサービス

    主な責務:
    1. トレーニングのバックグラウンド実行管理と進捗管理
    2. トレーナーの初期化（アンサンブル or 単一モデル）
    3. 学習データの準備と検証
    4. モデル学習と評価の実行（最適化含む）
    5. モデルの永続化
    """

    def __init__(
        self,
        trainer_type: str = "ensemble",
        ensemble_config: Optional[Dict[str, Any]] = None,
        single_model_config: Optional[Dict[str, Any]] = None,
    ):
        """初期化"""
        super().__init__()
        self.trainer_type = trainer_type
        config = self._create_unified_config(
            trainer_type, ensemble_config, single_model_config
        )
        self.trainer = EnsembleTrainer(ensemble_config=config)
        self.optimization_service = OptimizationService()

    def _create_unified_config(
        self,
        trainer_type: str,
        ensemble_config: Optional[Dict[str, Any]],
        single_model_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """設定に基づいて統一設定を作成"""
        if trainer_type == "ensemble":
            if ensemble_config is None:
                ensemble_config = unified_config.ml.ensemble.model_dump()
                if "default_method" in ensemble_config:
                    ensemble_config["method"] = ensemble_config.pop("default_method")
                if "algorithms" in ensemble_config:
                    ensemble_config["models"] = ensemble_config.pop("algorithms")
            return ensemble_config

        elif trainer_type == "single":
            if single_model_config is None:
                single_model_config = {"model_type": "lightgbm"}
            model_type = single_model_config.get("model_type", "lightgbm")
            return {
                "model_type": model_type,
                "models": [model_type],
                "method": "stacking",
                **single_model_config,
            }
        else:
            raise ValueError(f"サポートされていないトレーナータイプ: {trainer_type}")

    # --- オーケストレーション（API向け）機能 ---

    async def start_training(
        self, config, background_tasks, db: Session
    ) -> Dict[str, Any]:
        """MLトレーニングをバックグラウンドで開始"""
        try:
            # 設定の検証
            self._validate_training_config(config)

            # バックグラウンドタスクを追加
            background_tasks.add_task(self._train_in_background, config, db)

            return api_response(
                success=True,
                message="MLトレーニングを開始しました",
                data={
                    "training_id": f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                },
            )
        except Exception as e:
            logger.error(f"MLトレーニング開始エラー: {e}")
            raise

    def _validate_training_config(self, config) -> None:
        """トレーニング設定の検証"""
        if training_status["is_training"]:
            raise ValueError("既にトレーニングが実行中です")

        start_date = datetime.fromisoformat(config.start_date)
        end_date = datetime.fromisoformat(config.end_date)
        if start_date >= end_date:
            raise ValueError("開始日は終了日より前である必要があります")
        if (end_date - start_date).days < 7:
            raise ValueError("トレーニング期間は最低7日間必要です")

    async def get_training_status(self) -> Dict[str, Any]:
        """トレーニング状態を取得"""
        return dict(training_status)

    async def get_model_info(self) -> Dict[str, Any]:
        """現在のモデル情報を取得"""
        try:
            model_info_data = get_latest_model_with_info()
            model_status_base = get_model_info_with_defaults(model_info_data)

            model_status = {
                "is_loaded": model_info_data is not None,
                "model_path": model_info_data["path"] if model_info_data else None,
                **model_status_base,
            }
            return api_response(
                success=True,
                message="MLモデル情報を取得しました",
                data={
                    "model_status": model_status,
                    "last_training": training_status.get("model_info"),
                },
            )
        except Exception as e:
            logger.error(f"MLモデル情報取得エラー: {e}")
            default_status = get_model_info_with_defaults(None)
            default_status.update({"is_loaded": False, "model_path": None})
            return api_response(success=True, data={"model_status": default_status})

    async def stop_training(self) -> Dict[str, Any]:
        """トレーニングを停止"""
        if not training_status["is_training"]:
            raise ValueError("実行中のトレーニングがありません")

        background_task_manager.cleanup_all_tasks()
        training_status.update(
            {
                "is_training": False,
                "status": "stopped",
                "message": "トレーニングが停止されました",
                "end_time": datetime.now().isoformat(),
            }
        )
        return api_response(success=True, message="MLトレーニングを停止しました")

    async def _train_in_background(self, config, db: Session):
        """バックグラウンドトレーニング実行"""
        with background_task_manager.managed_task(
            task_name=f"MLトレーニング_{config.symbol}_{config.timeframe}",
        ) as task_id:
            try:
                training_status.update(
                    {
                        "is_training": True,
                        "progress": 0,
                        "status": "starting",
                        "message": "トレーニングを開始しています...",
                        "start_time": datetime.now().isoformat(),
                        "end_time": None,
                        "error": None,
                        "task_id": task_id,
                    }
                )

                # データ準備
                from app.services.backtest.backtest_data_service import (
                    BacktestDataService,
                )

                data_service = BacktestDataService(
                    OHLCVRepository(db),
                    OpenInterestRepository(db),
                    FundingRateRepository(db),
                )

                training_status.update(
                    {
                        "progress": 10,
                        "status": "loading_data",
                        "message": "データを読み込み中...",
                    }
                )
                training_data = data_service.get_ml_training_data(
                    symbol=config.symbol,
                    timeframe=config.timeframe,
                    start_date=datetime.fromisoformat(config.start_date),
                    end_date=datetime.fromisoformat(config.end_date),
                )

                # トレーナー設定の決定
                trainer_type, ensemble_cfg, single_cfg = self._determine_trainer_config(
                    config
                )

                # 自分自身のインスタンスまたは新しいインスタンスで学習実行
                # ここでは設定が異なる可能性があるため、新しいパラメータで自分を再初期化するか、
                # 別のメソッドで実行する。
                self._execute_actual_training(
                    trainer_type, ensemble_cfg, single_cfg, config, training_data
                )

            except Exception as e:
                logger.error(f"バックグラウンドトレーニングエラー: {e}", exc_info=True)
                training_status.update(
                    {
                        "is_training": False,
                        "status": "error",
                        "message": f"エラー: {e}",
                        "end_time": datetime.now().isoformat(),
                        "error": str(e),
                    }
                )

    def _determine_trainer_config(
        self, config: Any
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """設定からトレーナータイプ等を決定"""
        trainer_type = "ensemble"
        ensemble_cfg = (
            config.ensemble_config.model_dump()
            if config.ensemble_config
            else get_default_ensemble_config()
        )
        if not ensemble_cfg.get("enabled", True):
            trainer_type = "single"

        single_cfg = (
            config.single_model_config.model_dump()
            if config.single_model_config
            else None
        )
        if not single_cfg and trainer_type == "single":
            single_cfg = get_default_single_model_config()

        return trainer_type, ensemble_cfg, single_cfg

    def _execute_actual_training(
        self, trainer_type, ensemble_cfg, single_cfg, config, training_data
    ):
        """実際の学習処理"""
        # 現在のインスタンスを更新
        self.trainer_type = trainer_type
        cfg = self._create_unified_config(trainer_type, ensemble_cfg, single_cfg)
        self.trainer = EnsembleTrainer(ensemble_config=cfg)

        training_status.update(
            {
                "progress": 50,
                "status": "training",
                "message": "モデルをトレーニング中...",
            }
        )

        opt_settings = (
            config.optimization_settings
            if config.optimization_settings and config.optimization_settings.enabled
            else None
        )

        result = self.train_model(
            training_data=training_data,
            save_model=config.save_model,
            optimization_settings=opt_settings,
            test_size=1 - config.train_test_split,
            random_state=config.random_state,
        )

        training_status.update(
            {
                "is_training": False,
                "progress": 100,
                "status": "completed",
                "message": "トレーニングが完了しました",
                "end_time": datetime.now().isoformat(),
                "model_info": result,
            }
        )

    # --- コア学習機能 ---

    @safe_ml_operation(context="モデル学習")
    def train_model(
        self,
        training_data: Any,
        save_model: bool = True,
        optimization_settings: Optional[OptimizationSettings] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """モデルの学習を実行"""
        data_dict = (
            {"ohlcv": training_data}
            if isinstance(training_data, pd.DataFrame)
            else training_data
        )
        ohlcv = data_dict.get("ohlcv")
        fr, oi = data_dict.get("funding_rate"), data_dict.get("open_interest")

        best_params, is_optimized, opt_result = {}, False, None
        if optimization_settings and optimization_settings.enabled:
            logger.info("ハイパーパラメータ最適化を開始")
            try:
                opt_result = self.optimization_service.optimize_parameters(
                    trainer=self.trainer,
                    training_data=ohlcv,
                    optimization_settings=optimization_settings,
                    funding_rate_data=fr,
                    open_interest_data=oi,
                )
                best_params = opt_result.get("best_params", {})
                is_optimized = True
            except Exception as e:
                logger.error(f"最適化失敗: {e}")

        training_params = {
            "test_size": test_size,
            "random_state": random_state,
            "optimize_hyperparameters": False,
            **best_params,
        }
        result = self.trainer.train_model(
            ohlcv,
            funding_rate_data=fr,
            open_interest_data=oi,
            save_model=save_model,
            **training_params,
        )

        if isinstance(result, dict):
            result["is_optimized"] = is_optimized
            if opt_result:
                result["optimization_result"] = opt_result
        return result

    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """予測を実行"""
        return {
            "predictions": self.trainer.predict(features_df),
            "model_type": self.trainer_type,
            "feature_count": len(features_df.columns),
        }

    def generate_signals(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """シグナル生成"""
        return self.trainer.predict_signal(features_df)

    def load_model(self, model_path: str) -> bool:
        """モデルを読み込む"""
        return self.trainer.load_model(model_path)

    def get_current_model_path(self) -> Optional[str]:
        """読み込まれているモデルのパスを取得"""
        return (
            self.trainer.metadata.get("model_path")
            if hasattr(self.trainer, "metadata") and self.trainer.metadata
            else None
        )

    def get_current_model_info(self) -> Optional[Dict[str, Any]]:
        """読み込まれているモデルのメタデータを取得"""
        return getattr(self.trainer, "metadata", None)

    @staticmethod
    def get_available_single_models() -> list[str]:
        """利用可能な単一モデルのリストを取得"""
        return ["lightgbm", "xgboost", "catboost"]

    # --- リソース管理 ---
    def _cleanup_temporary_files(self, level: CleanupLevel):
        pass

    def _cleanup_cache(self, level: CleanupLevel):
        pass

    def _cleanup_models(self, level: CleanupLevel):
        if self.trainer and hasattr(self.trainer, "cleanup_resources"):
            self.trainer.cleanup_resources(level)


# グローバルインスタンス
ml_training_service = MLTrainingService(trainer_type="ensemble")

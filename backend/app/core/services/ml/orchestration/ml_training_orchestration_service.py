"""
MLトレーニング統合管理サービス

APIルーター内に散在していたMLトレーニング関連のビジネスロジックを統合管理します。
責務の分離とSOLID原則に基づいた設計を実現します。
"""

import logging
from typing import Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from app.core.services.ml.ml_training_service import MLTrainingService
from app.core.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.core.services.backtest_data_service import BacktestDataService
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.fear_greed_repository import FearGreedIndexRepository
from app.core.utils.api_utils import APIResponseHelper

logger = logging.getLogger(__name__)

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


class MLTrainingOrchestrationService:
    """
    MLトレーニング統合管理サービス

    MLモデルのトレーニング、状態管理、モデル情報取得等の
    統一的な処理を担当します。
    """

    def __init__(self):
        """初期化"""
        pass

    def get_data_service(self, db: Session) -> BacktestDataService:
        """データサービスの依存性注入"""
        ohlcv_repo = OHLCVRepository(db)
        oi_repo = OpenInterestRepository(db)
        fr_repo = FundingRateRepository(db)
        fear_greed_repo = FearGreedIndexRepository(db)
        return BacktestDataService(ohlcv_repo, oi_repo, fr_repo, fear_greed_repo)

    def validate_training_config(self, config) -> None:
        """
        トレーニング設定の検証

        Args:
            config: MLTrainingConfig

        Raises:
            ValueError: 設定が無効な場合
        """
        # 既にトレーニング中の場合はエラー
        if training_status["is_training"]:
            raise ValueError("既にトレーニングが実行中です")

        # 設定の検証
        start_date = datetime.fromisoformat(config.start_date)
        end_date = datetime.fromisoformat(config.end_date)

        if start_date >= end_date:
            raise ValueError("開始日は終了日より前である必要があります")

        if (end_date - start_date).days < 7:
            raise ValueError("トレーニング期間は最低7日間必要です")

    async def start_training(
        self, config, background_tasks, db: Session
    ) -> Dict[str, Any]:
        """
        MLトレーニングを開始

        Args:
            config: MLTrainingConfig
            background_tasks: BackgroundTasks
            db: データベースセッション

        Returns:
            トレーニング開始結果
        """
        try:
            # 設定の検証
            self.validate_training_config(config)

            # バックグラウンドタスクでトレーニング開始
            background_tasks.add_task(self._train_ml_model_background, config, db)

            return APIResponseHelper.api_response(
                success=True,
                message="MLトレーニングを開始しました",
                data={
                    "training_id": f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                },
            )

        except ValueError as e:
            logger.error(f"MLトレーニング設定エラー: {e}")
            raise
        except Exception as e:
            logger.error(f"MLトレーニング開始エラー: {e}")
            raise

    async def get_training_status(self) -> Dict[str, Any]:
        """
        MLトレーニングの状態を取得

        Returns:
            トレーニング状態
        """
        return dict(training_status)

    async def get_model_info(self) -> Dict[str, Any]:
        """
        現在のMLモデル情報を取得

        Returns:
            モデル情報
        """
        try:
            ml_orchestrator = MLOrchestrator()
            model_status = ml_orchestrator.get_model_status()

            return APIResponseHelper.api_response(
                success=True,
                message="MLモデル情報を取得しました",
                data={
                    "model_status": model_status,
                    "last_training": training_status.get("model_info"),
                },
            )

        except Exception as e:
            logger.error(f"MLモデル情報取得エラー: {e}")
            raise

    async def stop_training(self) -> Dict[str, Any]:
        """
        MLトレーニングを停止

        Returns:
            停止結果
        """
        global training_status

        try:
            if not training_status["is_training"]:
                raise ValueError("実行中のトレーニングがありません")

            # トレーニング停止（実際の実装では、トレーニングプロセスを停止する必要があります）
            training_status.update(
                {
                    "is_training": False,
                    "status": "stopped",
                    "message": "トレーニングが停止されました",
                    "end_time": datetime.now().isoformat(),
                }
            )

            return APIResponseHelper.api_response(
                success=True, message="MLトレーニングを停止しました"
            )

        except ValueError as e:
            logger.error(f"MLトレーニング停止エラー: {e}")
            raise
        except Exception as e:
            logger.error(f"MLトレーニング停止エラー: {e}")
            raise

    async def _train_ml_model_background(self, config, db: Session):
        """バックグラウンドでMLモデルをトレーニング"""
        global training_status

        try:
            # トレーニング開始
            training_status.update(
                {
                    "is_training": True,
                    "progress": 0,
                    "status": "starting",
                    "message": "トレーニングを開始しています...",
                    "start_time": datetime.now().isoformat(),
                    "end_time": None,
                    "error": None,
                }
            )

            logger.info("🚀 MLトレーニング開始")

            # データサービスの取得
            data_service = self.get_data_service(db)

            # データ取得
            training_status.update(
                {
                    "progress": 10,
                    "status": "loading_data",
                    "message": "データを読み込み中...",
                }
            )

            # 統合されたMLトレーニングデータを取得（OHLCV + OI + FR + Fear & Greed）
            training_data = data_service.get_ml_training_data(
                symbol=config.symbol,
                timeframe=config.timeframe,
                start_date=datetime.fromisoformat(config.start_date),
                end_date=datetime.fromisoformat(config.end_date),
            )

            # MLサービスの初期化
            training_status.update(
                {
                    "progress": 30,
                    "status": "initializing",
                    "message": "MLサービスを初期化中...",
                }
            )

            ml_service = MLTrainingService()

            # 最適化設定の準備
            optimization_settings = None
            if config.optimization_settings and config.optimization_settings.enabled:
                optimization_settings = config.optimization_settings

            # AutoML設定の準備
            automl_config_dict = None
            if config.automl_config:
                automl_config_dict = config.automl_config.dict()

            # トレーニング実行
            training_status.update(
                {
                    "progress": 50,
                    "status": "training",
                    "message": "モデルをトレーニング中...",
                }
            )

            training_result = ml_service.train_model(
                training_data=training_data,
                save_model=config.save_model,
                optimization_settings=optimization_settings,
                automl_config=automl_config_dict,
                test_size=1 - config.train_test_split,
                random_state=config.random_state,
            )

            # トレーニング完了
            training_status.update(
                {
                    "is_training": False,
                    "progress": 100,
                    "status": "completed",
                    "message": "トレーニングが完了しました",
                    "end_time": datetime.now().isoformat(),
                    "model_info": training_result,
                }
            )

            logger.info("✅ MLトレーニング完了")

        except Exception as e:
            # エラー処理
            training_status.update(
                {
                    "is_training": False,
                    "status": "error",
                    "message": f"トレーニング中にエラーが発生しました: {str(e)}",
                    "end_time": datetime.now().isoformat(),
                    "error": str(e),
                }
            )
            logger.error(f"❌ MLトレーニングエラー: {e}", exc_info=True)

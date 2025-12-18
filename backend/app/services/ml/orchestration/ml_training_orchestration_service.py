"""
MLトレーニング統合管理サービス
APIルーター内に散在していたMLトレーニング関連のビジネスロジックを統合管理します。

"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from sqlalchemy.orm import Session

from app.services.ml.common.default_configs import (
    get_default_ensemble_config,
    get_default_single_model_config,
)
from app.services.ml.ml_training_service import MLTrainingService
from app.services.ml.orchestration.background_task_manager import (
    background_task_manager,
)
from app.utils.error_handler import safe_ml_operation
from app.utils.response import api_response
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository

from .orchestration_utils import (
    get_latest_model_with_info,
    get_model_info_with_defaults,
)

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

    def get_data_service(self, db: Session):
        """データサービスの依存性注入"""
        # 循環インポートを避けるため、関数内でインポート
        from app.services.backtest.backtest_data_service import BacktestDataService

        ohlcv_repo = OHLCVRepository(db)
        oi_repo = OpenInterestRepository(db)
        fr_repo = FundingRateRepository(db)
        return BacktestDataService(ohlcv_repo, oi_repo, fr_repo)

    def validate_training_config(self, config) -> None:
        """
        トレーニング設定の検証

        Args:
            config: MLTrainingRequest (app.api.ml_training)

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
            config: MLTrainingRequest (app.api.ml_training)
            background_tasks: BackgroundTasks
            db: データベースセッション

        Returns:
            トレーニング開始結果
        """
        try:
            # 設定の検証とログ出力
            self._log_and_validate_config(config)

            # バックグラウンドタスクでトレーニング開始
            background_tasks.add_task(self._train_ml_model_background, config, db)

            return api_response(
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

    def _log_and_validate_config(self, config) -> None:
        """
        設定のログ出力と検証

        Args:
            config: MLTrainingRequest (app.api.ml_training)

        Raises:
            ValueError: 設定が無効な場合
        """
        # 設定の詳細ログ出力
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"📋 受信したconfig全体: {config}")
            logger.debug(f"📋 アンサンブル設定: {config.ensemble_config}")
            logger.debug(
                f"📋 アンサンブル設定enabled: {config.ensemble_config.enabled if config.ensemble_config else 'None'}"
            )
            logger.debug(f"📋 単一モデル設定: {config.single_model_config}")
            logger.debug(
                f"📋 単一モデルタイプ: {config.single_model_config.model_type if config.single_model_config else 'None'}"
            )
            logger.debug(f"📋 最適化設定: {config.optimization_settings}")

            # 設定の詳細確認
            if config.ensemble_config:
                ensemble_dict = config.ensemble_config.model_dump()
                logger.debug(f"📋 アンサンブル設定辞書: {ensemble_dict}")
                logger.debug(
                    f"📋 enabled値確認: {ensemble_dict.get('enabled')} (型: {type(ensemble_dict.get('enabled'))})"
                )

            if config.single_model_config:
                single_dict = config.single_model_config.model_dump()
                logger.debug(f"📋 単一モデル設定辞書: {single_dict}")
        else:
            logger.info(
                f"📋 MLトレーニング設定を受信: {config.symbol} ({config.timeframe})"
            )
            logger.info("  ※詳細な設定内容はDEBUGログで確認可能です")

        # 既存の検証ロジック
        self.validate_training_config(config)

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
            # 最新モデルの情報を取得
            model_info_data = get_latest_model_with_info()

            # デフォルト値を適用して統一されたフォーマットを取得
            model_status_base = get_model_info_with_defaults(model_info_data)

            # is_loadedとmodel_pathを追加
            if model_info_data:
                model_status = {
                    "is_loaded": True,
                    "model_path": model_info_data["path"],
                    **model_status_base,
                }
            else:
                model_status = {
                    "is_loaded": False,
                    "model_path": None,
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
            logger.error(f"MLモデル情報取得エラー: {e}", exc_info=True)
            # エラー時もデフォルト値で応答する（is_loaded=False）
            default_model_status = get_model_info_with_defaults(None)
            default_model_status["is_loaded"] = False
            default_model_status["model_path"] = None

            return api_response(
                success=True,
                message="MLモデル情報を取得しました（デフォルト値）",
                data={
                    "model_status": default_model_status,
                    "last_training": None,
                },
            )

    async def stop_training(self) -> Dict[str, Any]:
        """
        MLトレーニングを停止

        Returns:
            停止結果
        """
        try:
            if not training_status["is_training"]:
                raise ValueError("実行中のトレーニングがありません")

            # バックグラウンドタスクマネージャーのクリーンアップを実行
            background_task_manager.cleanup_all_tasks()

            # トレーニング停止（実際の実装では、トレーニングプロセスを停止する必要があります）
            training_status.update(
                {
                    "is_training": False,
                    "status": "stopped",
                    "message": "トレーニングが停止されました",
                    "end_time": datetime.now().isoformat(),
                }
            )

            return api_response(success=True, message="MLトレーニングを停止しました")

        except ValueError as e:
            logger.error(f"MLトレーニング停止エラー: {e}")
            raise
        except Exception as e:
            logger.error(f"MLトレーニング停止エラー: {e}")
            raise

    async def _train_ml_model_background(self, config, db: Session):
        """バックグラウンドでMLモデルをトレーニング"""
        # バックグラウンドタスクマネージャーを使用してリソース管理
        with background_task_manager.managed_task(
            task_name=f"MLトレーニング_{config.symbol}_{config.timeframe}",
        ) as task_id:
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
                        "task_id": task_id,
                    }
                )

                logger.info(f"🚀 MLトレーニング開始 (タスクID: {task_id})")

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

                # 統合されたMLトレーニングデータを取得（OHLCV + OI + FR）
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

                # トレーナータイプと設定の決定
                trainer_type, ensemble_config_dict, single_model_config_dict = (
                    self._determine_trainer_config(config)
                )

                # MLサービス初期化とトレーニング実行
                self._execute_ml_training_with_error_handling(
                    trainer_type,
                    ensemble_config_dict,
                    single_model_config_dict,
                    config,
                    training_data,
                )
            except Exception as e:
                logger.error(
                    f"MLトレーニングのバックグラウンド処理でエラーが発生: {e}",
                    exc_info=True,
                )
                training_status.update(
                    {
                        "is_training": False,
                        "status": "error",
                        "message": f"トレーニング中にエラーが発生しました: {e}",
                        "end_time": datetime.now().isoformat(),
                        "error": str(e),
                    }
                )

    def _determine_trainer_config(
        self, config: Any
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        トレーナータイプと設定を決定

        Args:
            config: トレーニング設定

        Returns:
            Tuple[trainer_type, ensemble_config_dict, single_model_config_dict]
        """
        trainer_type = "ensemble"
        ensemble_config_dict = None
        single_model_config_dict = None

        try:
            # アンサンブル設定の処理
            if config.ensemble_config:
                ensemble_config_dict = config.ensemble_config.model_dump()
                if not ensemble_config_dict.get("enabled", True):
                    trainer_type = "single"
            else:
                ensemble_config_dict = get_default_ensemble_config()

            # 単一モデル設定の処理
            if config.single_model_config:
                single_model_config_dict = config.single_model_config.model_dump()
            elif trainer_type == "single":
                single_model_config_dict = get_default_single_model_config()

            logger.info(
                f"🎯 トレーナータイプ決定: {trainer_type} "
                f"(Ensemble: {'有効' if trainer_type == 'ensemble' else '無効'})"
            )

        except Exception as e:
            logger.error(f"❌ トレーナー設定決定エラー: {e}")
            trainer_type = "ensemble"
            ensemble_config_dict = ensemble_config_dict or get_default_ensemble_config()

        return trainer_type, ensemble_config_dict, single_model_config_dict

    @safe_ml_operation(context="MLトレーニング実行")
    def _execute_ml_training_with_error_handling(
        self,
        trainer_type: str,
        ensemble_config_dict: Dict[str, Any],
        single_model_config_dict: Dict[str, Any],
        config,
        training_data,
    ) -> None:
        """
        エラーハンドリング付きMLトレーニング実行

        Args:
            trainer_type: トレーナータイプ
            ensemble_config_dict: アンサンブル設定辞書
            single_model_config_dict: 単一モデル設定辞書
            config: トレーニング設定
            training_data: 学習データ
        """
        try:
            logger.info("🔧 MLTrainingService初期化開始")
            ml_service = MLTrainingService(
                trainer_type=trainer_type,
                ensemble_config=ensemble_config_dict,
                single_model_config=single_model_config_dict,
            )
            logger.info(f"✅ MLTrainingService初期化完了: {ml_service.trainer_type}")

            # 実際に作成されたトレーナーの確認
            trainer_class_name = type(ml_service.trainer).__name__
            logger.info(f"✅ 作成されたトレーナー: {trainer_class_name}")

            if hasattr(ml_service.trainer, "model_type"):
                logger.info(f"✅ モデルタイプ: {ml_service.trainer.model_type}")

        except Exception as e:
            logger.error(f"❌ MLTrainingService初期化エラー: {e}")
            logger.error(f"❌ エラー詳細: {type(e).__name__}: {str(e)}")
            raise

        # 最適化設定の準備
        optimization_settings = None
        if config.optimization_settings and config.optimization_settings.enabled:
            optimization_settings = config.optimization_settings

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




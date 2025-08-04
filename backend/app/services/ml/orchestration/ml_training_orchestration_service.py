"""
MLトレーニング統合管理サービス

APIルーター内に散在していたMLトレーニング関連のビジネスロジックを統合管理します。
責務の分離とSOLID原則に基づいた設計を実現します。
"""

import logging
from datetime import datetime
from typing import Any, Dict

from sqlalchemy.orm import Session

from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.services.backtest.backtest_data_service import BacktestDataService
from app.services.ml.ml_training_service import MLTrainingService
from app.services.ml.orchestration.background_task_manager import (
    background_task_manager,
)
from app.utils.api_utils import APIResponseHelper
from app.utils.unified_error_handler import safe_ml_operation
from database.repositories.fear_greed_repository import FearGreedIndexRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository

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

    @staticmethod
    def get_default_automl_config() -> Dict[str, Any]:
        """デフォルトのAutoML設定を取得"""
        return {
            "tsfresh": {
                "enabled": True,
                "feature_selection": True,
                "fdr_level": 0.05,
                "feature_count_limit": 100,
                "parallel_jobs": 2,
            },
            "autofeat": {
                "enabled": True,
                "max_features": 50,
                "generations": 10,
                "population_size": 30,
                "tournament_size": 3,
            },
        }

    @staticmethod
    def get_financial_optimized_automl_config() -> Dict[str, Any]:
        """金融データ最適化AutoML設定を取得"""
        return {
            "tsfresh": {
                "enabled": True,
                "feature_selection": True,
                "fdr_level": 0.01,
                "feature_count_limit": 200,
                "parallel_jobs": 4,
            },
            "autofeat": {
                "enabled": True,
                "max_features": 100,
                "generations": 20,
                "population_size": 50,
                "tournament_size": 3,
            },
        }

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

            # AutoMLプロセスのクリーンアップ処理を実行
            self._cleanup_automl_processes()

            # バックグラウンドタスクマネージャーのクリーンアップも実行
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

        # バックグラウンドタスクマネージャーを使用してリソース管理
        with background_task_manager.managed_task(
            task_name=f"MLトレーニング_{config.symbol}_{config.timeframe}",
            cleanup_callbacks=[self._cleanup_automl_processes],
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

                # トレーナータイプの決定
                ensemble_config_dict = None
                single_model_config_dict = None
                trainer_type = "ensemble"  # デフォルト

                try:
                    logger.info("🔍 トレーナータイプ決定プロセス開始")

                    if config.ensemble_config:
                        logger.info(
                            f"📋 受信したアンサンブル設定: {config.ensemble_config}"
                        )
                        ensemble_config_dict = config.ensemble_config.model_dump()
                        logger.info(
                            f"📋 変換後のアンサンブル設定辞書: {ensemble_config_dict}"
                        )

                        # アンサンブルが無効化されている場合は単一モデルを使用
                        enabled_value = ensemble_config_dict.get("enabled", True)
                        logger.info(
                            f"🔍 enabled値の確認: {enabled_value} (型: {type(enabled_value)})"
                        )

                        if not enabled_value:
                            trainer_type = "single"
                            logger.info(
                                "🔄 アンサンブルが無効化されているため、単一モデルトレーニングを使用します"
                            )
                            logger.info(
                                f"📋 アンサンブル設定確認: enabled={enabled_value}"
                            )
                        else:
                            logger.info(
                                "🔗 アンサンブルが有効化されているため、アンサンブルトレーニングを使用します"
                            )
                    else:
                        logger.info(
                            "📋 アンサンブル設定が提供されていません。デフォルト（アンサンブル）を使用します"
                        )
                        # デフォルトのアンサンブル設定を作成
                        ensemble_config_dict = {
                            "enabled": True,
                            "method": "stacking",
                            "bagging_params": {
                                "n_estimators": 5,
                                "bootstrap_fraction": 0.8,
                                "base_model_type": "lightgbm",
                                "random_state": 42,
                            },
                            "stacking_params": {
                                "base_models": [
                                    "lightgbm",
                                    "xgboost",
                                    "gradient_boosting",
                                    "random_forest",
                                ],
                                "meta_model": "lightgbm",
                                "cv_folds": 5,
                                "use_probas": True,
                                "random_state": 42,
                            },
                        }

                    # 単一モデル設定の準備
                    if config.single_model_config:
                        logger.info(
                            f"📋 受信した単一モデル設定: {config.single_model_config}"
                        )
                        single_model_config_dict = (
                            config.single_model_config.model_dump()
                        )
                        logger.info(
                            f"📋 変換後の単一モデル設定辞書: {single_model_config_dict}"
                        )

                        if trainer_type == "single":
                            logger.info(
                                f"📋 単一モデル設定を使用: {single_model_config_dict}"
                            )
                    else:
                        logger.info("📋 単一モデル設定が提供されていません")
                        if trainer_type == "single":
                            # 単一モデルが選択されているが設定がない場合はデフォルトを使用
                            single_model_config_dict = {"model_type": "lightgbm"}
                            logger.info(
                                f"📋 デフォルト単一モデル設定を使用: {single_model_config_dict}"
                            )

                except Exception as e:
                    logger.error(f"❌ トレーナータイプ決定中にエラーが発生: {e}")
                    logger.error(f"❌ エラー詳細: {type(e).__name__}: {str(e)}")
                    logger.warning(
                        "⚠️ エラーのため、デフォルト（アンサンブル）を使用します"
                    )
                    trainer_type = "ensemble"

                # AutoML設定の準備
                automl_config_dict = None
                if config.automl_config:
                    automl_config_dict = config.automl_config.model_dump()

                # トレーナータイプの最終確認
                logger.info(f"🎯 最終決定されたトレーナータイプ: {trainer_type}")
                if trainer_type == "single":
                    logger.info("🤖 単一モデルトレーニングを実行します")
                    logger.info(
                        f"🤖 使用する単一モデル設定: {single_model_config_dict}"
                    )
                else:
                    logger.info("🔗 アンサンブルトレーニングを実行します")
                    logger.info(f"🔗 使用するアンサンブル設定: {ensemble_config_dict}")

                # MLサービス初期化とトレーニング実行
                self._execute_ml_training_with_error_handling(
                    trainer_type,
                    automl_config_dict,
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

    def _cleanup_automl_processes(self):
        """AutoMLプロセスのクリーンアップ処理"""
        try:
            logger.info("🧹 AutoMLプロセスのクリーンアップを開始")

            # メモリ使用量を記録
            import psutil

            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024

            # AutoML関連のリソースをクリーンアップ
            self._cleanup_autofeat_resources()
            self._cleanup_tsfresh_resources()
            self._cleanup_enhanced_feature_service()
            self._cleanup_ml_training_service()
            self._cleanup_data_preprocessor()

            # 強制ガベージコレクション
            import gc

            collected = gc.collect()

            # メモリ使用量の変化を記録
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_freed = memory_before - memory_after

            logger.info(
                f"✅ AutoMLクリーンアップ完了: "
                f"{collected}オブジェクト回収, "
                f"{memory_freed:.2f}MB解放"
            )

        except Exception as e:
            logger.error(f"AutoMLクリーンアップエラー: {e}")

    def _cleanup_autofeat_resources(self):
        """AutoFeat関連リソースのクリーンアップ"""
        try:

            # AutoFeatの一時ファイルとキャッシュをクリア
            import os
            import shutil
            import tempfile

            # AutoFeatが作成する一時ディレクトリをクリーンアップ
            temp_dir = tempfile.gettempdir()
            for item in os.listdir(temp_dir):
                if item.startswith("autofeat_"):
                    temp_path = os.path.join(temp_dir, item)
                    try:
                        if os.path.isdir(temp_path):
                            shutil.rmtree(temp_path)
                        else:
                            os.remove(temp_path)

                    except Exception as e:
                        logger.warning(
                            f"AutoFeat一時ファイル削除エラー {temp_path}: {e}"
                        )

            logger.debug("AutoFeatリソースクリーンアップ完了")

        except Exception as e:
            logger.warning(f"AutoFeatクリーンアップエラー: {e}")

    def _cleanup_tsfresh_resources(self):
        """TSFresh関連リソースのクリーンアップ"""
        try:

            # TSFreshの内部キャッシュをクリア
            try:
                from tsfresh.utilities.dataframe_functions import clear_cache

                clear_cache()
                logger.debug("TSFreshキャッシュクリア完了")
            except ImportError:
                logger.debug("TSFreshキャッシュクリア機能が利用できません")
            except Exception as e:
                logger.warning(f"TSFreshキャッシュクリアエラー: {e}")

            # TSFreshの一時ファイルをクリーンアップ
            import os
            import tempfile

            temp_dir = tempfile.gettempdir()
            for item in os.listdir(temp_dir):
                if item.startswith("tsfresh_") or item.startswith("tmp_tsfresh"):
                    temp_path = os.path.join(temp_dir, item)
                    try:
                        if os.path.isfile(temp_path):
                            os.remove(temp_path)
                        logger.debug(f"TSFresh一時ファイル削除: {temp_path}")
                    except Exception as e:
                        logger.warning(
                            f"TSFresh一時ファイル削除エラー {temp_path}: {e}"
                        )

            logger.debug("TSFreshリソースクリーンアップ完了")

        except Exception as e:
            logger.warning(f"TSFreshクリーンアップエラー: {e}")

    def _cleanup_enhanced_feature_service(self):
        """FeatureEngineeringService関連リソースのクリーンアップ"""
        try:
            # FeatureEngineeringServiceのインスタンスを作成してクリーンアップ
            from app.services.ml.feature_engineering.feature_engineering_service import (
                FeatureEngineeringService,
            )

            # 一時的なインスタンスを作成してクリーンアップメソッドを呼び出し
            temp_service = FeatureEngineeringService()
            temp_service.cleanup_resources()

            # インスタンスを削除
            del temp_service

        except Exception as e:
            logger.warning(f"FeatureEngineeringServiceクリーンアップエラー: {e}")

    def _cleanup_ml_training_service(self):
        """MLTrainingService関連リソースのクリーンアップ"""
        try:
            # グローバルMLTrainingServiceインスタンスのクリーンアップ
            from app.services.ml.ml_training_service import ml_training_service

            if hasattr(ml_training_service, "cleanup_resources"):
                ml_training_service.cleanup_resources()

        except Exception as e:
            logger.warning(f"MLTrainingServiceクリーンアップエラー: {e}")

    def _cleanup_data_preprocessor(self):
        """DataPreprocessor関連リソースのクリーンアップ"""
        try:
            # グローバルDataPreprocessorインスタンスのクリーンアップ
            from app.utils.data_preprocessing import data_preprocessor

            if hasattr(data_preprocessor, "clear_cache"):
                data_preprocessor.clear_cache()

        except Exception as e:
            logger.warning(f"DataPreprocessorクリーンアップエラー: {e}")

    @safe_ml_operation(context="MLトレーニング実行")
    def _execute_ml_training_with_error_handling(
        self,
        trainer_type: str,
        automl_config_dict: Dict[str, Any],
        ensemble_config_dict: Dict[str, Any],
        single_model_config_dict: Dict[str, Any],
        config,
        training_data,
    ) -> None:
        """
        エラーハンドリング付きMLトレーニング実行

        Args:
            trainer_type: トレーナータイプ
            automl_config_dict: AutoML設定辞書
            ensemble_config_dict: アンサンブル設定辞書
            single_model_config_dict: 単一モデル設定辞書
            config: トレーニング設定
            training_data: 学習データ
        """
        global training_status

        try:
            logger.info("🔧 MLTrainingService初期化開始")
            ml_service = MLTrainingService(
                trainer_type=trainer_type,
                automl_config=automl_config_dict,
                ensemble_config=ensemble_config_dict,
                single_model_config=single_model_config_dict,
            )
            logger.info(
                f"✅ MLTrainingService初期化完了: trainer_type={ml_service.trainer_type}"
            )

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
            automl_config=automl_config_dict,
            test_size=1 - config.train_test_split,
            random_state=config.random_state,
        )

        # トレーニング完了後のクリーンアップ処理
        self._cleanup_automl_processes()

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

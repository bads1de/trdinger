"""
MLトレーニングAPI

機械学習モデルのトレーニングと管理を行うAPIエンドポイント
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from app.core.services.ml.ml_training_service import MLTrainingService
from app.core.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.core.services.backtest_data_service import BacktestDataService
from app.core.utils.unified_error_handler import UnifiedErrorHandler
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository

# AutoML設定モデルをインポート
from app.api.routes.automl_features import (
    AutoMLConfigModel,
    TSFreshConfigModel,
    FeaturetoolsConfigModel,
    AutoFeatConfigModel,
)

from database.connection import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml-training", tags=["ML Training"])


def get_default_automl_config() -> AutoMLConfigModel:
    """デフォルトのAutoML設定を取得"""
    return AutoMLConfigModel(
        tsfresh=TSFreshConfigModel(
            enabled=True,
            feature_selection=True,
            fdr_level=0.05,
            feature_count_limit=100,
            parallel_jobs=2,
        ),
        featuretools=FeaturetoolsConfigModel(
            enabled=True,
            max_depth=2,
            max_features=50,
        ),
        autofeat=AutoFeatConfigModel(
            enabled=True,
            max_features=50,
            generations=10,  # API層ではgenerationsを使用
            population_size=30,
            tournament_size=3,
        ),
    )


def get_financial_optimized_automl_config() -> AutoMLConfigModel:
    """金融データ最適化AutoML設定を取得"""
    return AutoMLConfigModel(
        tsfresh=TSFreshConfigModel(
            enabled=True,
            feature_selection=True,
            fdr_level=0.01,  # より厳しい選択
            feature_count_limit=200,  # 金融データ用に増加
            parallel_jobs=4,
        ),
        featuretools=FeaturetoolsConfigModel(
            enabled=True,
            max_depth=3,  # より深い特徴量合成
            max_features=100,  # 金融データ用に増加
        ),
        autofeat=AutoFeatConfigModel(
            enabled=True,
            max_features=100,
            generations=20,  # より多くの世代（API層ではgenerationsを使用）
            population_size=50,
            tournament_size=3,
        ),
    )


class ParameterSpaceConfig(BaseModel):
    """パラメータ空間設定"""

    type: str = Field(..., description="パラメータ型 (real, integer, categorical)")
    low: Optional[float] = Field(None, description="最小値 (real, integer)")
    high: Optional[float] = Field(None, description="最大値 (real, integer)")
    categories: Optional[list] = Field(None, description="カテゴリ一覧 (categorical)")


class OptimizationSettingsConfig(BaseModel):
    """最適化設定"""

    enabled: bool = Field(default=False, description="最適化を有効にするか")
    method: str = Field(default="optuna", description="最適化手法 (optuna)")
    n_calls: int = Field(default=50, description="最適化試行回数")
    parameter_space: Dict[str, ParameterSpaceConfig] = Field(
        default_factory=dict, description="パラメータ空間設定"
    )


# グローバルなトレーニング状態管理
training_status = {
    "is_training": False,
    "progress": 0,
    "status": "idle",
    "message": "",
    "start_time": None,
    "end_time": None,
    "model_info": None,
    "error": None,
}


class MLTrainingConfig(BaseModel):
    """MLトレーニング設定"""

    symbol: str = Field(..., description="取引ペア（例: BTC/USDT:USDT）")
    timeframe: str = Field(default="1h", description="時間軸")
    start_date: str = Field(..., description="開始日（YYYY-MM-DD）")
    end_date: str = Field(..., description="終了日（YYYY-MM-DD）")
    validation_split: float = Field(default=0.2, description="検証データ分割比率")
    prediction_horizon: int = Field(default=24, description="予測期間（時間）")
    threshold_up: float = Field(default=0.02, description="上昇判定閾値")
    threshold_down: float = Field(default=-0.02, description="下落判定閾値")
    save_model: bool = Field(default=True, description="モデルを保存するか")
    # 新しい設定項目
    train_test_split: float = Field(
        default=0.8, description="トレーニング/テスト分割比率"
    )
    cross_validation_folds: int = Field(
        default=5, description="クロスバリデーション分割数"
    )
    random_state: int = Field(default=42, description="ランダムシード")
    early_stopping_rounds: int = Field(default=100, description="早期停止ラウンド数")
    max_depth: int = Field(default=10, description="最大深度")
    n_estimators: int = Field(default=100, description="推定器数")
    learning_rate: float = Field(default=0.1, description="学習率")

    # 最適化設定
    optimization_settings: Optional[OptimizationSettingsConfig] = Field(
        None, description="ハイパーパラメータ最適化設定"
    )

    # AutoML特徴量エンジニアリング設定
    automl_config: Optional[AutoMLConfigModel] = Field(
        None, description="AutoML特徴量エンジニアリング設定"
    )


class MLTrainingResponse(BaseModel):
    """MLトレーニング応答"""

    success: bool
    message: str
    training_id: Optional[str] = None


class MLStatusResponse(BaseModel):
    """MLステータス応答"""

    is_training: bool
    progress: int
    status: str
    message: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def get_data_service():
    """データサービスの依存性注入"""
    db = next(get_db())
    try:
        ohlcv_repo = OHLCVRepository(db)
        oi_repo = OpenInterestRepository(db)
        fr_repo = FundingRateRepository(db)
        return BacktestDataService(ohlcv_repo, oi_repo, fr_repo)
    finally:
        db.close()


async def train_ml_model_background(config: MLTrainingConfig):
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

        # データサービスを初期化
        data_service = get_data_service()

        # データ取得
        training_status.update(
            {
                "progress": 10,
                "status": "loading_data",
                "message": "トレーニングデータを読み込んでいます...",
            }
        )

        start_date = datetime.fromisoformat(config.start_date)
        end_date = datetime.fromisoformat(config.end_date)

        training_data = data_service.get_data_for_backtest(
            symbol=config.symbol,
            timeframe=config.timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        if training_data.empty:
            raise ValueError(f"指定された期間のデータが見つかりません: {config.symbol}")

        # MLサービスを初期化
        training_status.update(
            {
                "progress": 20,
                "status": "initializing",
                "message": "MLサービスを初期化しています...",
            }
        )

        ml_service = MLTrainingService()

        # モデルトレーニング
        training_status.update(
            {
                "progress": 30,
                "status": "training",
                "message": "モデルをトレーニングしています...",
            }
        )

        # ファンディングレートとオープンインタレストデータを分離
        funding_rate_data = None
        open_interest_data = None

        if "FundingRate" in training_data.columns:
            funding_rate_data = training_data[["FundingRate"]].copy()

        if "OpenInterest" in training_data.columns:
            open_interest_data = training_data[["OpenInterest"]].copy()

        # OHLCVデータのみを抽出
        ohlcv_data = training_data[["Open", "High", "Low", "Close", "Volume"]].copy()

        # 最適化設定を準備
        optimization_settings = None
        if config.optimization_settings and config.optimization_settings.enabled:
            from app.core.services.ml.ml_training_service import OptimizationSettings

            logger.info("=" * 60)
            logger.info("🎯 ハイパーパラメータ最適化が有効化されました")
            logger.info("📊 最適化手法: optuna")
            logger.info(f"🔄 試行回数: {config.optimization_settings.n_calls}")
            logger.info(
                f"📋 最適化対象パラメータ数: {len(config.optimization_settings.parameter_space)}"
            )

            # パラメータ空間の詳細をログ出力
            for (
                param_name,
                param_config,
            ) in config.optimization_settings.parameter_space.items():
                if param_config.type in ["real", "integer"]:
                    logger.info(
                        f"  - {param_name} ({param_config.type}): [{param_config.low}, {param_config.high}]"
                    )
                else:
                    logger.info(
                        f"  - {param_name} ({param_config.type}): {param_config.categories}"
                    )
            logger.info("=" * 60)

            # ParameterSpaceConfigを辞書形式に変換
            parameter_space_dict = {}
            for (
                param_name,
                param_config,
            ) in config.optimization_settings.parameter_space.items():
                parameter_space_dict[param_name] = {
                    "type": param_config.type,
                    "low": param_config.low,
                    "high": param_config.high,
                    "categories": param_config.categories,
                }

            optimization_settings = OptimizationSettings(
                enabled=config.optimization_settings.enabled,
                n_calls=config.optimization_settings.n_calls,
                parameter_space=parameter_space_dict,
            )

            training_status.update(
                {
                    "message": f"ハイパーパラメータ最適化を実行中 ({config.optimization_settings.method})"
                }
            )
        else:
            logger.info("📝 通常のMLトレーニングを実行します（最適化なし）")

        # AutoML設定の処理
        automl_config = config.automl_config
        if automl_config is None:
            # デフォルトのAutoML設定を使用
            automl_config = get_financial_optimized_automl_config()
            logger.info("🤖 デフォルトの金融最適化AutoML設定を使用します")
        else:
            logger.info("🤖 カスタムAutoML設定を使用します")

        # AutoML設定を辞書形式に変換
        automl_config_dict = {
            "tsfresh": automl_config.tsfresh.model_dump(),
            "featuretools": automl_config.featuretools.model_dump(),
            "autofeat": automl_config.autofeat.model_dump(),
        }

        # トレーニング実行
        training_result = ml_service.train_model(
            training_data=ohlcv_data,
            funding_rate_data=funding_rate_data,
            open_interest_data=open_interest_data,
            save_model=config.save_model,
            optimization_settings=optimization_settings,
            automl_config=automl_config_dict,  # AutoML設定を追加
            # 新しいMLTrainingServiceは設定から自動的にパラメータを取得
            test_size=1 - config.train_test_split,
            random_state=config.random_state,
        )

        # トレーニング完了
        training_status.update(
            {
                "progress": 100,
                "status": "completed",
                "message": "トレーニングが完了しました",
                "end_time": datetime.now().isoformat(),
                "is_training": False,
                "model_info": {
                    "accuracy": training_result.get("accuracy", 0.0),
                    "loss": training_result.get("loss", 0.0),
                    "model_path": training_result.get("model_path", ""),
                    "feature_count": training_result.get("feature_count", 0),
                    "training_samples": len(training_data),
                    "validation_split": config.validation_split,
                },
            }
        )

        logger.info(f"MLモデルトレーニング完了: {config.symbol}")

    except Exception as e:
        logger.error(f"MLトレーニングエラー: {e}")
        training_status.update(
            {
                "is_training": False,
                "status": "error",
                "message": f"トレーニングエラー: {str(e)}",
                "end_time": datetime.now().isoformat(),
                "error": str(e),
            }
        )


@router.post("/train", response_model=MLTrainingResponse)
async def start_ml_training(
    config: MLTrainingConfig, background_tasks: BackgroundTasks
):
    """
    MLモデルのトレーニングを開始
    """
    logger.info("🚀 /api/ml-training/train エンドポイントが呼び出されました")
    logger.info(f"📋 最適化設定: {config.optimization_settings}")
    global training_status

    async def _start_training():
        # 既にトレーニング中の場合はエラー
        if training_status["is_training"]:
            from fastapi import HTTPException

            raise HTTPException(status_code=400, detail="既にトレーニングが実行中です")

        # 設定の検証
        start_date = datetime.fromisoformat(config.start_date)
        end_date = datetime.fromisoformat(config.end_date)

        if start_date >= end_date:
            raise ValueError("開始日は終了日より前である必要があります")

        if (end_date - start_date).days < 7:
            raise ValueError("トレーニング期間は最低7日間必要です")

        # バックグラウンドタスクでトレーニング開始
        background_tasks.add_task(train_ml_model_background, config)

        return MLTrainingResponse(
            success=True,
            message="MLトレーニングを開始しました",
            training_id=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    return await UnifiedErrorHandler.safe_execute_async(_start_training)


@router.get("/training/status", response_model=MLStatusResponse)
async def get_ml_training_status():
    """
    MLトレーニングの状態を取得
    """
    return MLStatusResponse(**training_status)


@router.get("/model-info")
async def get_ml_model_info():
    """
    現在のMLモデル情報を取得
    """

    async def _get_model_info():
        ml_orchestrator = MLOrchestrator()
        model_status = ml_orchestrator.get_model_status()

        return {
            "success": True,
            "model_status": model_status,
            "last_training": training_status.get("model_info"),
        }

    return await UnifiedErrorHandler.safe_execute_async(_get_model_info)


@router.post("/stop")
async def stop_ml_training():
    """
    MLトレーニングを停止
    """
    global training_status

    async def _stop_training():
        if not training_status["is_training"]:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=400, detail="実行中のトレーニングがありません"
            )

        # トレーニング停止（実際の実装では、トレーニングプロセスを停止する必要があります）
        training_status.update(
            {
                "is_training": False,
                "status": "stopped",
                "message": "トレーニングが停止されました",
                "end_time": datetime.now().isoformat(),
            }
        )

        return {"success": True, "message": "トレーニングを停止しました"}

    return await UnifiedErrorHandler.safe_execute_async(_stop_training)

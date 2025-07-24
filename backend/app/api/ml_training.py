"""
MLトレーニングAPI

機械学習モデルのトレーニングと管理を行うAPIエンドポイント
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.services.ml.orchestration.ml_training_orchestration_service import (
    MLTrainingOrchestrationService,
)
from app.core.utils.unified_error_handler import UnifiedErrorHandler

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


# グローバル状態管理は削除（OrchestrationServiceに移動）


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


# データサービス関数とバックグラウンド関数は削除（OrchestrationServiceに移動）


@router.post("/train", response_model=MLTrainingResponse)
async def start_ml_training(
    config: MLTrainingConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    MLモデルのトレーニングを開始
    """
    logger.info("🚀 /api/ml-training/train エンドポイントが呼び出されました")
    logger.info(f"📋 最適化設定: {config.optimization_settings}")

    async def _start_training():
        # ビジネスロジックをサービス層に委譲
        orchestration_service = MLTrainingOrchestrationService()
        return await orchestration_service.start_training(
            config=config, background_tasks=background_tasks, db=db
        )

    return await UnifiedErrorHandler.safe_execute_async(_start_training)


@router.get("/training/status", response_model=MLStatusResponse)
async def get_ml_training_status():
    """
    MLトレーニングの状態を取得
    """
    # ビジネスロジックをサービス層に委譲
    orchestration_service = MLTrainingOrchestrationService()
    status = await orchestration_service.get_training_status()
    return MLStatusResponse(**status)


@router.get("/model-info")
async def get_ml_model_info():
    """
    現在のMLモデル情報を取得
    """

    async def _get_model_info():
        # ビジネスロジックをサービス層に委譲
        orchestration_service = MLTrainingOrchestrationService()
        return await orchestration_service.get_model_info()

    return await UnifiedErrorHandler.safe_execute_async(_get_model_info)


@router.post("/stop")
async def stop_ml_training():
    """
    MLトレーニングを停止
    """

    async def _stop_training():
        # ビジネスロジックをサービス層に委譲
        orchestration_service = MLTrainingOrchestrationService()
        return await orchestration_service.stop_training()

    return await UnifiedErrorHandler.safe_execute_async(_stop_training)

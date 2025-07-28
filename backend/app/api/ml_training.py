"""
MLトレーニングAPI

機械学習モデルのトレーニングと管理を行うAPIエンドポイント
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.services.ml.orchestration.ml_training_orchestration_service import (
    MLTrainingOrchestrationService,
)
from app.utils.unified_error_handler import UnifiedErrorHandler


from app.api.automl_features import (
    AutoMLConfigModel,
    TSFreshConfigModel,
    FeaturetoolsConfigModel,
    AutoFeatConfigModel,
)

from database.connection import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml-training", tags=["ML Training"])


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


class BaggingParamsConfig(BaseModel):
    """バギングパラメータ設定"""

    n_estimators: int = Field(default=5, description="ベースモデル数")
    bootstrap_fraction: float = Field(
        default=0.8, description="ブートストラップサンプリング比率"
    )
    base_model_type: str = Field(
        default="lightgbm",
        description="ベースモデルタイプ（lightgbm, gradient_boosting, random_forest, xgboost等）",
    )
    mixed_models: Optional[List[str]] = Field(
        default=None,
        description="混合バギング用モデルリスト（指定時はbase_model_typeより優先、多様性確保）",
    )
    random_state: Optional[int] = Field(default=None, description="ランダムシード")


class StackingParamsConfig(BaseModel):
    """スタッキングパラメータ設定"""

    base_models: List[str] = Field(
        default=["lightgbm", "random_forest"], description="ベースモデルのリスト"
    )
    meta_model: str = Field(default="lightgbm", description="メタモデル")
    cv_folds: int = Field(default=5, description="クロスバリデーション分割数")
    use_probas: bool = Field(default=True, description="確率値を使用するか")


class EnsembleConfig(BaseModel):
    """アンサンブル学習設定"""

    enabled: bool = Field(default=True, description="アンサンブル学習を有効にするか")
    method: str = Field(
        default="bagging", description="アンサンブル手法 (bagging, stacking)"
    )
    bagging_params: BaggingParamsConfig = Field(
        default_factory=BaggingParamsConfig, description="バギングパラメータ"
    )
    stacking_params: StackingParamsConfig = Field(
        default_factory=StackingParamsConfig, description="スタッキングパラメータ"
    )


# グローバル状態管理は削除（OrchestrationServiceに移動）


class MLTrainingConfig(BaseModel):
    """
    MLトレーニング設定

    アンサンブル学習をデフォルトとしたML学習の設定を定義します。
    バギングとスタッキングの両方をサポートし、複数のモデルを組み合わせて
    予測精度と頑健性を向上させます。
    """

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

    # アンサンブル学習設定
    ensemble_config: Optional[EnsembleConfig] = Field(
        default_factory=lambda: EnsembleConfig(
            enabled=True,
            method="stacking",  # デフォルトをスタッキングに変更（多様性重視）
            bagging_params=BaggingParamsConfig(n_estimators=5, bootstrap_fraction=0.8),
            stacking_params=StackingParamsConfig(
                base_models=[
                    "lightgbm",
                    "xgboost",
                    "gradient_boosting",
                    "random_forest",
                ],  # 4種類のモデルで多様性確保
                meta_model="lightgbm",
                cv_folds=5,
                use_probas=True,
            ),
        ),
        description="アンサンブル学習設定（デフォルト: 多様性重視のスタッキング）",
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
    アンサンブル学習によるMLモデルのトレーニングを開始

    デフォルトでバギング手法を使用し、複数のLightGBMモデルを組み合わせて
    予測精度と頑健性を向上させます。スタッキング手法も選択可能です。

    Args:
        config: MLトレーニング設定（アンサンブル設定を含む）
        background_tasks: バックグラウンドタスク管理
        db: データベースセッション

    Returns:
        MLTrainingResponse: トレーニング開始応答
    """
    logger.info("🚀 /api/ml-training/train エンドポイントが呼び出されました")
    logger.info(f"📋 アンサンブル設定: {config.ensemble_config}")
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

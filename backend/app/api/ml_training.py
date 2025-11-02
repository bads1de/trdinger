"""
MLトレーニングAPI

機械学習モデルのトレーニングと管理を行うAPIエンドポイント
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.automl_features import AutoMLConfigModel
from app.services.ml.orchestration.ml_training_orchestration_service import (
    MLTrainingOrchestrationService,
)
from app.api.dependencies import get_ml_training_orchestration_service
from app.utils.error_handler import ErrorHandler
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
    """バギングパラメータ設定（scikit-learn BaggingClassifier対応）"""

    n_estimators: int = Field(default=5, description="ベースモデル数")
    bootstrap_fraction: float = Field(
        default=0.8, description="ブートストラップサンプリング比率（max_samples）"
    )
    base_model_type: str = Field(
        default="lightgbm",
        description="ベースモデルタイプ（lightgbm, xgboost, tabnet）",
    )
    mixed_models: Optional[List[str]] = Field(
        default=None,
        description="混合バギング用モデルリスト（指定時はbase_model_typeより優先、多様性確保）",
    )
    random_state: Optional[int] = Field(default=42, description="ランダムシード")
    n_jobs: int = Field(default=-1, description="並列処理数（-1で全CPU使用）")
    bootstrap: bool = Field(
        default=True, description="ブートストラップサンプリングを使用するか"
    )
    max_features: float = Field(
        default=1.0, description="各ベースモデルで使用する特徴量の割合"
    )


class StackingParamsConfig(BaseModel):
    """スタッキングパラメータ設定（scikit-learn StackingClassifier対応）"""

    base_models: List[str] = Field(
        default=["lightgbm", "xgboost"], description="ベースモデルのリスト（Essential 2 Modelsのみ）"
    )
    meta_model: str = Field(
        default="logistic_regression",
        description="メタモデル（logistic_regression, lightgbm, xgboost, tabnet）",
    )
    cv_folds: int = Field(default=5, description="クロスバリデーション分割数")
    stack_method: str = Field(
        default="predict_proba", description="スタック方法（predict_proba, predict）"
    )
    random_state: Optional[int] = Field(default=42, description="ランダムシード")
    n_jobs: int = Field(default=-1, description="並列処理数（-1で全CPU使用）")
    passthrough: bool = Field(
        default=False, description="元の特徴量をメタモデルに渡すか"
    )


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


class SingleModelConfig(BaseModel):
    """単一モデル学習設定"""

    model_type: str = Field(
        default="lightgbm",
        description="使用するモデルタイプ (lightgbm, xgboost, tabnet)",
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
        default=None,  # デフォルトをNoneに変更してフロントエンドからの設定を優先
        description="アンサンブル学習設定（フロントエンドから明示的に設定）",
    )

    # 単一モデル学習設定
    single_model_config: Optional[SingleModelConfig] = Field(
        default=None,  # デフォルトをNoneに変更してフロントエンドからの設定を優先
        description="単一モデル学習設定（アンサンブル無効時に使用）",
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


@router.post("/train", response_model=MLTrainingResponse)
async def start_ml_training(
    config: MLTrainingConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    orchestration_service: MLTrainingOrchestrationService = Depends(
        get_ml_training_orchestration_service
    ),
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

    async def _start_training():
        return await orchestration_service.start_training(
            config=config, background_tasks=background_tasks, db=db
        )

    return await ErrorHandler.safe_execute_async(_start_training)


@router.get("/training/status", response_model=MLStatusResponse)
async def get_ml_training_status(
    orchestration_service: MLTrainingOrchestrationService = Depends(
        get_ml_training_orchestration_service
    ),
):
    """
    MLトレーニングの状態を取得
    """
    status = await orchestration_service.get_training_status()
    return MLStatusResponse(**status)


@router.get("/model-info")
async def get_ml_model_info(
    orchestration_service: MLTrainingOrchestrationService = Depends(
        get_ml_training_orchestration_service
    ),
):
    """
    現在のMLモデル情報を取得
    """

    async def _get_model_info():
        return await orchestration_service.get_model_info()

    return await ErrorHandler.safe_execute_async(_get_model_info)


@router.post("/stop")
async def stop_ml_training(
    orchestration_service: MLTrainingOrchestrationService = Depends(
        get_ml_training_orchestration_service
    ),
):
    """
    MLトレーニングを停止
    """

    async def _stop_training():
        return await orchestration_service.stop_training()

    return await ErrorHandler.safe_execute_async(_stop_training)

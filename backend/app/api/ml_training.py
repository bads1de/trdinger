"""
MLトレーニングAPI

機械学習モデルのトレーニングと管理を行うAPIエンドポイント
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.dependencies import get_ml_training_orchestration_service
from app.services.ml.orchestration.ml_training_orchestration_service import (
    MLTrainingOrchestrationService,
)
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


class StackingParamsConfig(BaseModel):
    """スタッキングパラメータ設定（scikit-learn StackingClassifier対応）"""

    base_models: List[str] = Field(
        default=["lightgbm", "xgboost"],
        description="ベースモデルのリスト（Essential 2 Modelsのみ）",
    )
    meta_model: str = Field(
        default="logistic_regression",
        description="メタモデル（logistic_regression, lightgbm, xgboost）",
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


class EnsembleRequest(BaseModel):
    """アンサンブル学習設定（APIリクエスト用）

    Note:
        このクラスは app.config.unified_config.EnsembleConfig（環境変数ベースの設定）
        とは異なり、APIリクエストボディのスキーマを定義します。
    """

    enabled: bool = Field(default=True, description="アンサンブル学習を有効にするか")
    method: str = Field(default="stacking", description="アンサンブル手法 (stacking)")
    stacking_params: StackingParamsConfig = Field(
        default_factory=StackingParamsConfig, description="スタッキングパラメータ"
    )


class SingleModelConfig(BaseModel):
    """単一モデル学習設定"""

    model_type: str = Field(
        default="lightgbm",
        description="使用するモデルタイプ (lightgbm, xgboost)",
    )


# グローバル状態管理は削除（OrchestrationServiceに移動）


class MLTrainingRequest(BaseModel):
    """MLトレーニング設定（APIリクエスト用）

    アンサンブル学習をデフォルトとしたML学習の設定を定義します。
    バギングとスタッキングの両方をサポートし、複数のモデルを組み合わせて
    予測精度と頑健性を向上させます。

    Note:
        このクラスは app.config.unified_config.MLTrainingConfig（環境変数ベースの設定）
        とは異なり、APIリクエストボディのスキーマを定義します。
    """

    symbol: str = Field(..., description="取引ペア（例: BTC/USDT:USDT）")
    timeframe: str = Field(default="1h", description="時間軸")
    start_date: str = Field(..., description="開始日（YYYY-MM-DD）")
    end_date: str = Field(..., description="終了日（YYYY-MM-DD）")
    validation_split: float = Field(default=0.2, description="検証データ分割比率")
    prediction_horizon: int = Field(default=24, description="予測期間（時間）")
    threshold_up: float = Field(default=0.02, description="上昇判定閾値（方向予測用）")
    threshold_down: float = Field(
        default=-0.02, description="下落判定閾値（方向予測用）"
    )
    # ボラティリティ予測用パラメータ
    quantile_threshold: float = Field(
        default=0.33, description="トレンド判定の分位数閾値（上位N%）"
    )
    threshold_method: str = Field(
        default="TREND_SCANNING",
        description="閾値判定方法 (TREND_SCANNING, TRIPLE_BARRIER)",
    )
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

    # AutoML特徴量エンジニアリング設定は削除されました（autofeat機能の削除に伴う）
    # 特徴量プロファイル設定も削除されました（研究目的専用のためシンプル化）

    # アンサンブル学習設定
    ensemble_config: Optional[EnsembleRequest] = Field(
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
    config: MLTrainingRequest,
    background_tasks: BackgroundTasks,
    orchestration_service: MLTrainingOrchestrationService = Depends(
        get_ml_training_orchestration_service
    ),
    db: Session = Depends(get_db),
):
    """
    スタッキングアンサンブル学習またはシングルモデルによるMLモデルのトレーニングを開始

    アンサンブル学習が有効な場合、スタッキング手法を使用して異なるモデルの
    予測を統合し、予測精度と頑健性を向上させます。
    アンサンブル学習が無効な場合は、単一のモデルでトレーニングを行います。

    Args:
        config: MLトレーニング設定（アンサンブル設定を含む）
        background_tasks: バックグラウンドタスク管理
        orchestration_service: MLトレーニング管理サービス
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

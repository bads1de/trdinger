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


from app.api.automl_features import AutoMLConfigModel
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


class SingleModelConfig(BaseModel):
    """単一モデル学習設定"""

    model_type: str = Field(
        default="lightgbm",
        description="使用するモデルタイプ (lightgbm, xgboost, catboost, tabnet, knn)",
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
    logger.info(f"📋 受信したconfig全体: {config}")
    logger.info(f"📋 アンサンブル設定: {config.ensemble_config}")
    logger.info(
        f"📋 アンサンブル設定enabled: {config.ensemble_config.enabled if config.ensemble_config else 'None'}"
    )
    logger.info(f"📋 単一モデル設定: {config.single_model_config}")
    logger.info(
        f"📋 単一モデルタイプ: {config.single_model_config.model_type if config.single_model_config else 'None'}"
    )
    logger.info(f"📋 最適化設定: {config.optimization_settings}")

    # 設定の詳細確認
    if config.ensemble_config:
        ensemble_dict = config.ensemble_config.model_dump()
        logger.info(f"📋 アンサンブル設定辞書: {ensemble_dict}")
        logger.info(
            f"📋 enabled値確認: {ensemble_dict.get('enabled')} (型: {type(ensemble_dict.get('enabled'))})"
        )

    if config.single_model_config:
        single_dict = config.single_model_config.model_dump()
        logger.info(f"📋 単一モデル設定辞書: {single_dict}")

    async def _start_training():
        # アルゴリズム名の検証（algorithm_registry 非依存）
        if config.single_model_config:
            from app.services.ml.ml_training_service import MLTrainingService

            model_type = config.single_model_config.model_type
            available_models = MLTrainingService.get_available_single_models()

            if model_type not in available_models:
                return {
                    "success": False,
                    "error": f"指定されたアルゴリズム '{model_type}' は利用できません",
                    "available_models": available_models,
                    "message": f"利用可能なモデル: {', '.join(available_models)}",
                }

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

    orchestration_service = MLTrainingOrchestrationService()
    status = await orchestration_service.get_training_status()
    return MLStatusResponse(**status)


@router.get("/model-info")
async def get_ml_model_info():
    """
    現在のMLモデル情報を取得
    """

    async def _get_model_info():

        orchestration_service = MLTrainingOrchestrationService()
        return await orchestration_service.get_model_info()

    return await UnifiedErrorHandler.safe_execute_async(_get_model_info)


@router.post("/stop")
async def stop_ml_training():
    """
    MLトレーニングを停止
    """

    async def _stop_training():

        orchestration_service = MLTrainingOrchestrationService()
        return await orchestration_service.stop_training()

    return await UnifiedErrorHandler.safe_execute_async(_stop_training)


@router.get("/available-models")
async def get_available_models():
    """
    利用可能な単一モデルのリストを取得
    """

    async def _get_available_models():
        from app.services.ml.ml_training_service import MLTrainingService

        available_models = MLTrainingService.get_available_single_models()

        return {
            "success": True,
            "available_models": available_models,
            "message": f"{len(available_models)}個のモデルが利用可能です",
        }

    return await UnifiedErrorHandler.safe_execute_async(_get_available_models)


@router.get("/algorithms")
async def get_available_algorithms():
    """
    利用可能なアルゴリズム名のリストを取得（軽量版）
    フロントエンドは定数を使用するため、検証用の簡単なリストのみ返す
    """

    async def _get_available_algorithms():
        # algorithm_registry からは取得せず、MLTrainingServiceの一覧を使用
        from app.services.ml.ml_training_service import MLTrainingService

        algorithms = MLTrainingService.get_available_single_models()

        return {
            "success": True,
            "algorithms": algorithms,
            "total_count": len(algorithms),
            "message": f"{len(algorithms)}個のアルゴリズムが利用可能です",
        }

    return await UnifiedErrorHandler.safe_execute_async(_get_available_algorithms)


@router.get("/algorithms/{algorithm_name}")
async def validate_algorithm(algorithm_name: str):
    """
    指定されたアルゴリズムが利用可能かどうかを検証
    フロントエンドは定数を使用するため、検証のみ行う
    """

    async def _validate_algorithm():
        # algorithm_registry 非依存で検証
        from app.services.ml.ml_training_service import MLTrainingService

        available_algorithms = MLTrainingService.get_available_single_models()
        is_valid = algorithm_name in available_algorithms

        if not is_valid:
            return {
                "success": False,
                "error": f"アルゴリズム '{algorithm_name}' が見つかりません",
                "available_algorithms": available_algorithms,
                "message": f"利用可能なアルゴリズム: {', '.join(available_algorithms)}",
            }

        return {
            "success": True,
            "algorithm_name": algorithm_name,
            "is_valid": True,
            "message": f"アルゴリズム '{algorithm_name}' は利用可能です",
        }

    return await UnifiedErrorHandler.safe_execute_async(_validate_algorithm)

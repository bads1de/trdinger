"""
AutoML特徴量エンジニアリング API エンドポイント

AutoML特徴量生成・選択機能のREST APIを提供します。
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import BaseModel, Field, ValidationError

from app.api.dependencies import get_automl_feature_generation_service
from app.services.ml.feature_engineering.automl_feature_generation_service import (
    AutoMLFeatureGenerationService,
)
from app.services.ml.feature_engineering.automl_features.automl_config import (
    AutoMLConfig,
)
from app.services.ml.feature_engineering.enhanced_feature_engineering_service import (
    EnhancedFeatureEngineeringService,
)
from app.utils.unified_error_handler import UnifiedErrorHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/automl-features", tags=["AutoML Features"])


# Pydantic モデル定義
class TSFreshConfigModel(BaseModel):
    """TSFresh設定モデル"""

    enabled: bool = True
    feature_selection: bool = True
    fdr_level: float = Field(0.05, ge=0.001, le=0.1)
    feature_count_limit: int = Field(100, ge=10, le=500)
    parallel_jobs: int = Field(4, ge=1, le=8)
    performance_mode: str = Field(
        "balanced", pattern="^(fast|balanced|financial_optimized|comprehensive)$"
    )


# Featuretools は削除済みのため、Pydantic モデル定義からも完全に除去しました


class AutoFeatConfigModel(BaseModel):
    """AutoFeat設定モデル"""

    enabled: bool = True
    max_features: int = Field(100, ge=10, le=200)
    generations: int = Field(20, ge=5, le=50)
    population_size: int = Field(50, ge=20, le=200)
    tournament_size: int = Field(3, ge=2, le=10)


class AutoMLConfigModel(BaseModel):
    """AutoML設定モデル"""

    tsfresh: TSFreshConfigModel
    autofeat: AutoFeatConfigModel


class FeatureGenerationRequest(BaseModel):
    """特徴量生成リクエストモデル"""

    symbol: str = Field(..., description="取引シンボル")
    timeframe: str = Field("1h", description="時間枠")
    limit: int = Field(1000, ge=100, le=10000, description="データ数")
    automl_config: Optional[AutoMLConfigModel] = None
    include_target: bool = Field(False, description="ターゲット変数を含むか")


class FeatureGenerationResponse(BaseModel):
    """特徴量生成レスポンスモデル"""

    success: bool
    message: str
    feature_count: int
    processing_time: float
    statistics: Dict[str, Any]
    feature_names: List[str]


class ConfigValidationResponse(BaseModel):
    """設定検証レスポンスモデル"""

    success: bool
    valid: bool
    errors: List[str]
    warnings: List[str]


@router.post("/generate", response_model=FeatureGenerationResponse)
async def generate_features(
    request: FeatureGenerationRequest,
    background_tasks: BackgroundTasks,
    service: AutoMLFeatureGenerationService = Depends(
        get_automl_feature_generation_service
    ),
):
    """
    AutoML特徴量を生成

    Args:
        request: 特徴量生成リクエスト
        background_tasks: バックグラウンドタスク
        service: 特徴量エンジニアリングサービス

    Returns:
        特徴量生成結果
    """

    async def _generate():
        logger.info(f"AutoML特徴量生成開始: {request.symbol}, {request.timeframe}")

        # AutoML設定を辞書形式に変換
        automl_config_dict = None
        if request.automl_config:
            automl_config_dict = request.automl_config.model_dump()

        # AutoMLFeatureGenerationServiceを使用して特徴量を生成
        result_df, stats = await service.generate_features(
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=request.limit,
            automl_config=automl_config_dict,
            include_target=request.include_target,
        )

        # 特徴量名を取得
        feature_names = service.get_feature_names(result_df)

        # 処理時間を取得
        processing_time = service.get_processing_time(stats)

        return FeatureGenerationResponse(
            success=True,
            message="特徴量生成が正常に完了しました",
            feature_count=len(feature_names),
            processing_time=processing_time,
            statistics=stats,
            feature_names=feature_names,
        )

    return await UnifiedErrorHandler.safe_execute_async(
        _generate, message="特徴量生成に失敗しました"
    )


@router.post("/validate-config", response_model=ConfigValidationResponse)
async def validate_config(
    request: Request,
    service: EnhancedFeatureEngineeringService = Depends(
        EnhancedFeatureEngineeringService
    ),
):
    """
    AutoML設定を検証

    Args:
        request: HTTPリクエスト
        service: 特徴量エンジニアリングサービス

    Returns:
        設定検証結果
    """

    async def _validate():
        try:
            # 生のリクエストデータを取得（ログには詳細を出さない）
            raw_data = await request.json()

            # Pydanticモデルでバリデーション（成功時も詳細はログ出力しない）
            config = AutoMLConfigModel(**raw_data)

            config_dict = config.model_dump()
            validation_result = service.validate_automl_config(config_dict)

            return ConfigValidationResponse(
                success=True,
                valid=validation_result["valid"],
                errors=validation_result.get("errors", []),
                warnings=validation_result.get("warnings", []),
            )
        except ValidationError as e:
            logger.error(f"Pydanticバリデーションエラー: {e}")
            return ConfigValidationResponse(
                success=False,
                valid=False,
                errors=[f"設定形式エラー: {str(e)}"],
                warnings=[],
            )
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            return ConfigValidationResponse(
                success=False,
                valid=False,
                errors=[f"設定検証エラー: {str(e)}"],
                warnings=[],
            )

    return await UnifiedErrorHandler.safe_execute_async(
        _validate, message="設定検証に失敗しました"
    )


@router.get("/default-config")
async def get_default_config():
    """
    デフォルトAutoML設定を取得

    Returns:
        デフォルト設定
    """

    async def _get_config():
        default_config = AutoMLConfig.get_financial_optimized_config()
        return {"success": True, "config": default_config.to_dict()}

    return await UnifiedErrorHandler.safe_execute_async(
        _get_config, message="デフォルト設定取得に失敗しました"
    )


@router.post("/clear-cache")
async def clear_cache(
    service: EnhancedFeatureEngineeringService = Depends(
        EnhancedFeatureEngineeringService
    ),
):
    """
    AutoMLキャッシュをクリア

    Returns:
        クリア結果
    """

    async def _clear():
        service.clear_automl_cache()
        return {"success": True, "message": "AutoMLキャッシュをクリアしました"}

    return await UnifiedErrorHandler.safe_execute_async(
        _clear, message="キャッシュクリアに失敗しました"
    )

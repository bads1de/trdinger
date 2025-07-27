"""
AutoML特徴量エンジニアリング API エンドポイント

AutoML特徴量生成・選択機能のREST APIを提供します。
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import logging

from app.services.ml.feature_engineering.enhanced_feature_engineering_service import (
    EnhancedFeatureEngineeringService,
)
from app.services.ml.feature_engineering.automl_features.automl_config import (
    AutoMLConfig,
)


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


class FeaturetoolsConfigModel(BaseModel):
    """Featuretools設定モデル"""

    enabled: bool = True
    max_depth: int = Field(2, ge=1, le=5)
    max_features: int = Field(50, ge=10, le=200)


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
    featuretools: FeaturetoolsConfigModel
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

    valid: bool
    errors: List[str]
    warnings: List[str]


# 依存性注入
def get_feature_service() -> EnhancedFeatureEngineeringService:
    """特徴量エンジニアリングサービスを取得"""
    return EnhancedFeatureEngineeringService()


@router.post("/generate", response_model=FeatureGenerationResponse)
async def generate_features(
    request: FeatureGenerationRequest,
    background_tasks: BackgroundTasks,
    service: EnhancedFeatureEngineeringService = Depends(get_feature_service),
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
    try:
        logger.info(f"AutoML特徴量生成開始: {request.symbol}, {request.timeframe}")

        # AutoML設定を適用
        if request.automl_config:
            config_dict = request.automl_config.dict()
            service._update_automl_config(config_dict)

        # サンプルデータを生成（実際の実装では外部データソースから取得）
        sample_data = _generate_sample_ohlcv_data(request.limit)

        # ターゲット変数を生成（必要な場合）
        target = None
        if request.include_target:
            target = _generate_sample_target(len(sample_data))

        # 特徴量生成を実行
        result_df = service.calculate_enhanced_features(
            ohlcv_data=sample_data, target=target
        )

        # 統計情報を取得
        stats = service.get_enhancement_stats()

        # 特徴量名を取得
        feature_names = list(result_df.columns)

        return FeatureGenerationResponse(
            success=True,
            message="特徴量生成が正常に完了しました",
            feature_count=len(feature_names),
            processing_time=stats.get("total_time", 0),
            statistics=stats,
            feature_names=feature_names,
        )

    except Exception as e:
        logger.error(f"特徴量生成エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"特徴量生成に失敗しました: {str(e)}"
        )


@router.post("/validate-config", response_model=ConfigValidationResponse)
async def validate_config(
    config: AutoMLConfigModel,
    service: EnhancedFeatureEngineeringService = Depends(get_feature_service),
):
    """
    AutoML設定を検証

    Args:
        config: AutoML設定
        service: 特徴量エンジニアリングサービス

    Returns:
        設定検証結果
    """
    try:
        config_dict = config.dict()
        validation_result = service.validate_automl_config(config_dict)

        return ConfigValidationResponse(
            valid=validation_result["valid"],
            errors=validation_result.get("errors", []),
            warnings=validation_result.get("warnings", []),
        )

    except Exception as e:
        logger.error(f"設定検証エラー: {e}")
        raise HTTPException(status_code=500, detail=f"設定検証に失敗しました: {str(e)}")


@router.get("/available-features")
async def get_available_features(
    service: EnhancedFeatureEngineeringService = Depends(get_feature_service),
):
    """
    利用可能なAutoML特徴量のリストを取得

    Returns:
        利用可能な特徴量のリスト
    """
    try:
        features = service.get_available_automl_features()
        return {
            "success": True,
            "features": features,
            "total_count": sum(len(feature_list) for feature_list in features.values()),
        }

    except Exception as e:
        logger.error(f"特徴量リスト取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"特徴量リスト取得に失敗しました: {str(e)}"
        )


@router.get("/default-config")
async def get_default_config():
    """
    デフォルトAutoML設定を取得

    Returns:
        デフォルト設定
    """
    try:
        default_config = AutoMLConfig.get_financial_optimized_config()
        return {"success": True, "config": default_config.to_dict()}

    except Exception as e:
        logger.error(f"デフォルト設定取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"デフォルト設定取得に失敗しました: {str(e)}"
        )


@router.post("/clear-cache")
async def clear_cache(
    service: EnhancedFeatureEngineeringService = Depends(get_feature_service),
):
    """
    AutoMLキャッシュをクリア

    Returns:
        クリア結果
    """
    try:
        service.clear_automl_cache()
        return {"success": True, "message": "AutoMLキャッシュをクリアしました"}

    except Exception as e:
        logger.error(f"キャッシュクリアエラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"キャッシュクリアに失敗しました: {str(e)}"
        )


@router.get("/statistics")
async def get_statistics(
    service: EnhancedFeatureEngineeringService = Depends(get_feature_service),
):
    """
    最後の処理統計情報を取得

    Returns:
        処理統計情報
    """
    try:
        stats = service.get_enhancement_stats()
        return {"success": True, "statistics": stats}

    except Exception as e:
        logger.error(f"統計情報取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"統計情報取得に失敗しました: {str(e)}"
        )


# ヘルパー関数
def _generate_sample_ohlcv_data(rows: int) -> pd.DataFrame:
    """サンプルOHLCVデータを生成"""
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=rows, freq="1h")

    base_price = 50000
    price_changes = np.random.normal(0, 0.02, rows)
    prices = [base_price]

    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))

    prices = np.array(prices)

    data = {
        "Open": prices * (1 + np.random.normal(0, 0.001, rows)),
        "High": prices * (1 + np.abs(np.random.normal(0, 0.005, rows))),
        "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, rows))),
        "Close": prices,
        "Volume": np.random.lognormal(10, 1, rows),
    }

    df = pd.DataFrame(data, index=dates)

    # High >= Close >= Low の制約を満たす
    df["High"] = np.maximum(df["High"], df[["Open", "Close"]].max(axis=1))
    df["Low"] = np.minimum(df["Low"], df[["Open", "Close"]].min(axis=1))

    return df


def _generate_sample_target(rows: int) -> pd.Series:
    """サンプルターゲット変数を生成"""
    import numpy as np

    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1, 2], size=rows), name="target")

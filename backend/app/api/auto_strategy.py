import logging

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from functools import lru_cache

from app.core.services.auto_strategy import AutoStrategyService
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.orchestration.auto_strategy_orchestration_service import (
    AutoStrategyOrchestrationService,
)
from app.core.utils.unified_error_handler import UnifiedErrorHandler
from app.core.utils.api_utils import APIResponseHelper


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auto-strategy", tags=["auto-strategy"])


@lru_cache()
def get_auto_strategy_service_cached() -> AutoStrategyService:
    try:
        return AutoStrategyService()
    except Exception as e:
        logger.error(f"AutoStrategyService初期化エラー: {e}", exc_info=True)
        # この例外は後続の依存関係で捕捉される
        raise


def get_auto_strategy_service() -> AutoStrategyService:
    """AutoStrategyServiceの依存性注入"""
    try:
        service = get_auto_strategy_service_cached()
        return service
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AutoStrategyServiceが利用できません。サーバーログを確認してください。",
        )


# リクエスト・レスポンスモデル


class GAGenerationRequest(BaseModel):
    """GA戦略生成リクエスト"""

    experiment_name: str = Field(..., description="実験名")
    base_config: Dict[str, Any] = Field(..., description="基本バックテスト設定")
    ga_config: Dict[str, Any] = Field(..., description="GA設定")

    class Config:
        json_schema_extra = {
            "example": {
                "experiment_name": "BTC_Strategy_Gen_001",
                "base_config": {
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-19",
                    "initial_capital": 100000,
                    "commission_rate": 0.00055,
                },
                "ga_config": {
                    "population_size": 10,
                    "generations": 5,
                    "crossover_rate": 0.8,
                    "mutation_rate": 0.1,
                    "elite_size": 2,
                    "max_indicators": 3,
                    "allowed_indicators": ["SMA", "EMA", "RSI", "MACD", "BB", "ATR"],
                    # 多目的最適化設定（オプション）
                    "enable_multi_objective": False,
                    "objectives": ["total_return"],
                    "objective_weights": [1.0],
                },
            }
        }


class GAGenerationResponse(BaseModel):
    """GA戦略生成レスポンス"""

    success: bool
    experiment_id: str
    message: str


class GAProgressResponse(BaseModel):
    """GA進捗レスポンス"""

    success: bool
    progress: Optional[Dict[str, Any]] = None
    message: str


class GAResultResponse(BaseModel):
    """GA結果レスポンス"""

    success: bool
    result: Optional[Dict[str, Any]] = None
    message: str


class MultiObjectiveResultResponse(BaseModel):
    """多目的最適化GA結果レスポンス"""

    success: bool
    result: Optional[Dict[str, Any]] = None
    pareto_front: Optional[List[Dict[str, Any]]] = None
    objectives: Optional[List[str]] = None
    message: str


class StrategyTestRequest(BaseModel):
    """戦略テストリクエスト"""

    strategy_gene: Dict[str, Any] = Field(..., description="戦略遺伝子")
    backtest_config: Dict[str, Any] = Field(..., description="バックテスト設定")


class StrategyTestResponse(BaseModel):
    """戦略テストレスポンス"""

    success: bool
    result: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    message: str


class DefaultConfigResponse(BaseModel):
    config: Dict[str, Any]


class PresetsResponse(BaseModel):
    presets: Dict[str, Any]


class StopExperimentResponse(BaseModel):
    success: bool
    message: str


class ListExperimentsResponse(BaseModel):
    experiments: List[Dict[str, Any]]


# APIエンドポイント
@router.post(
    "/generate",
    response_model=GAGenerationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_strategy(
    request: GAGenerationRequest,
    background_tasks: BackgroundTasks,
    auto_strategy_service: AutoStrategyService = Depends(get_auto_strategy_service),
):
    """
    GA戦略生成を開始

    遺伝的アルゴリズムを使用して取引戦略を自動生成します。
    バックグラウンドで実行され、進捗は別のエンドポイントで確認できます。
    """

    async def _generate_strategy():
        logger.info("=== GA戦略生成API呼び出し開始 ===")
        logger.info(f"実験名: {request.experiment_name}")

        # 戦略生成を開始（バックグラウンド実行）
        experiment_id = auto_strategy_service.start_strategy_generation(
            experiment_name=request.experiment_name,
            ga_config_dict=request.ga_config,
            backtest_config_dict=request.base_config,
            background_tasks=background_tasks,
        )
        logger.info(f"戦略生成タスクをバックグラウンドで開始: {experiment_id}")

        return APIResponseHelper.api_response(
            success=True,
            message="GA戦略生成を開始しました",
            data={"experiment_id": experiment_id},
        )

    return await UnifiedErrorHandler.safe_execute_async(_generate_strategy)


@router.get("/experiments", response_model=ListExperimentsResponse)
async def list_experiments(
    auto_strategy_service: AutoStrategyService = Depends(get_auto_strategy_service),
):
    """
    実験一覧を取得

    実行中・完了済みの全実験の一覧を取得します。
    """

    async def _list_experiments():
        experiments = auto_strategy_service.list_experiments()
        return ListExperimentsResponse(experiments=experiments)

    return await UnifiedErrorHandler.safe_execute_async(_list_experiments)


@router.post(
    "/experiments/{experiment_id}/stop",
    response_model=StopExperimentResponse,
    status_code=status.HTTP_200_OK,
)
async def stop_experiment(
    experiment_id: str,
    auto_strategy_service: AutoStrategyService = Depends(get_auto_strategy_service),
):
    """
    実験を停止

    実行中の実験を停止します。
    """

    async def _stop_experiment():
        success = auto_strategy_service.stop_experiment(experiment_id)

        if not success:
            logger.warning(
                f"実験 {experiment_id} を停止できませんでした（存在しないか、既に完了している可能性があります）"
            )
            from fastapi import HTTPException

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="実験を停止できませんでした（存在しないか、既に完了している可能性があります）",
            )

        return StopExperimentResponse(success=True, message="実験を停止しました")

    return await UnifiedErrorHandler.safe_execute_async(_stop_experiment)


@router.get("/experiments/{experiment_id}/results", response_model=GAResultResponse)
async def get_experiment_results(
    experiment_id: str,
    auto_strategy_service: AutoStrategyService = Depends(get_auto_strategy_service),
):
    """
    実験結果を取得

    指定された実験IDの結果を取得します。
    多目的最適化の場合はパレート最適解も含まれます。
    """

    async def _get_experiment_results():
        result = auto_strategy_service.get_experiment_result(experiment_id)

        if result is None:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"実験 {experiment_id} が見つかりません",
            )

        # 多目的最適化の結果かどうかを判定
        if "pareto_front" in result and "objectives" in result:
            return APIResponseHelper.api_response(
                success=True,
                message="多目的最適化実験結果を取得しました",
                data={
                    "result": result,
                    "pareto_front": result.get("pareto_front"),
                    "objectives": result.get("objectives"),
                },
            )
        else:
            return APIResponseHelper.api_response(
                success=True, message="実験結果を取得しました", data={"result": result}
            )

    return await UnifiedErrorHandler.safe_execute_async(_get_experiment_results)


@router.post("/test-strategy", response_model=StrategyTestResponse)
async def test_strategy(
    request: StrategyTestRequest,
    auto_strategy_service: AutoStrategyService = Depends(get_auto_strategy_service),
):
    """
    単一戦略のテスト実行

    指定された戦略遺伝子から戦略を生成し、バックテストを実行します。
    GA実行前の戦略検証に使用できます。
    """

    async def _test_strategy():
        # ビジネスロジックをサービス層に委譲
        orchestration_service = AutoStrategyOrchestrationService()
        result = await orchestration_service.test_strategy(
            request=request, auto_strategy_service=auto_strategy_service
        )

        return StrategyTestResponse(
            success=result["success"],
            result=result.get("result"),
            errors=result.get("errors"),
            message=result["message"],
        )

    return await UnifiedErrorHandler.safe_execute_async(_test_strategy)


@router.get("/config/default", response_model=DefaultConfigResponse)
async def get_default_config():
    """
    デフォルトGA設定を取得

    推奨されるGA設定のデフォルト値を返します。
    """

    async def _get_default_config():
        default_config = GAConfig.create_default()
        return DefaultConfigResponse(config=default_config.to_dict())

    return await UnifiedErrorHandler.safe_execute_async(_get_default_config)


@router.get("/config/presets", response_model=PresetsResponse)
async def get_config_presets():
    """
    GA設定プリセットを取得

    用途別のGA設定プリセット（高速、標準、徹底）を返します。
    """

    async def _get_config_presets():
        presets = {
            "fast": GAConfig.create_fast().to_dict(),
            "default": GAConfig.create_default().to_dict(),
            "thorough": GAConfig.create_thorough().to_dict(),
        }
        return PresetsResponse(presets=presets)

    return await UnifiedErrorHandler.safe_execute_async(_get_config_presets)

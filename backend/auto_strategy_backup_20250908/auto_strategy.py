"""
自動戦略生成APIエンドポイント

遺伝的アルゴリズムを用いた取引戦略の自動生成、管理、テストに関するAPIを提供します。
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, status
from pydantic import BaseModel, Field

from app.api.dependencies import (
    get_auto_strategy_service,
)
from app.services.auto_strategy import AutoStrategyService
from app.services.auto_strategy.config.auto_strategy_config import GAConfig
from app.utils.response import api_response, error_response
from app.utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auto-strategy", tags=["auto-strategy"])


class GAGenerationRequest(BaseModel):
    """GA戦略生成リクエスト"""

    experiment_id: str = Field(
        ..., description="実験ID（フロントエンドで生成されたUUID）"
    )
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
    message: str
    data: Dict[str, Any]
    timestamp: str


class GAResultResponse(BaseModel):
    """GA結果レスポンス"""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str


class ExperimentResultsResponse(BaseModel):
    """実験結果レスポンス"""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str


class ExperimentStatusResponse(BaseModel):
    """実験ステータスレスポンス"""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str


class StopExperimentResponse(BaseModel):
    success: bool
    message: str


class ListExperimentsResponse(BaseModel):
    experiments: List[Dict[str, Any]]




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
        try:
            logger.info("=== GA戦略生成API呼び出し開始 ===")
            logger.info(f"実験名: {request.experiment_name}")

            # 戦略生成を開始（バックグラウンド実行）
            # フロントエンドから送信されたexperiment_idを使用
            experiment_id = auto_strategy_service.start_strategy_generation(
                experiment_id=request.experiment_id,  # フロントエンドで生成されたUUID
                experiment_name=request.experiment_name,
                ga_config_dict=request.ga_config,
                backtest_config_dict=request.base_config,
                background_tasks=background_tasks,
            )
            logger.info(f"戦略生成タスクをバックグラウンドで開始: {experiment_id}")

            return api_response(
                success=True,
                message="GA戦略生成を開始しました",
                data={"experiment_id": experiment_id},
            )
        except Exception as e:
            logger.error(f"戦略生成エラー: {e}", exc_info=True)
            # 詳細なエラー情報を返す（デバッグ用）
            return error_response(
                message=f"戦略生成に失敗しました: {str(e)}",
                details={"error_type": type(e).__name__, "error_details": str(e)},
            )

    return await ErrorHandler.safe_execute_async(_generate_strategy)


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

    return await ErrorHandler.safe_execute_async(_list_experiments)


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
        try:
            result = auto_strategy_service.stop_experiment(experiment_id)
            return StopExperimentResponse(
                success=result.get("success", False),
                message=result.get("message", "実験を停止しました"),
            )
        except ValueError as e:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

    return await ErrorHandler.safe_execute_async(_stop_experiment)


@router.get(
    "/experiments/{experiment_id}/results",
    response_model=ExperimentResultsResponse,
)
async def get_experiment_results(
    experiment_id: str,
    auto_strategy_service: AutoStrategyService = Depends(get_auto_strategy_service),
):
    """
    実験結果を取得

    GA実験の完了した結果情報を取得します。実験が実行中または未完了の場合、適切なメッセージを返します。
    """

    async def _get_experiment_results():
        try:
            logger.info(f"=== 実験結果取得API呼び出し開始 ===")
            logger.info(f"実験ID: {experiment_id}")

            # 実験結果を取得
            results = auto_strategy_service.get_experiment_results(experiment_id)

            if results.get("status") == "completed":
                return api_response(
                    success=True,
                    message="実験結果を取得しました",
                    data=results,
                )
            elif results.get("status") == "running":
                return api_response(
                    success=True,
                    message="実験が実行中です。完了後に再度お試しください。",
                    data=results,
                )
            else:
                return api_response(
                    success=False,
                    message="実験結果が見つかりません",
                    data={"experiment_id": experiment_id, "status": "not_found"},
                )
        except Exception as e:
            logger.error(f"実験結果取得エラー: {e}", exc_info=True)
            return error_response(
                message=f"実験結果の取得に失敗しました: {str(e)}",
                details={"error_type": type(e).__name__, "experiment_id": experiment_id},
            )

    return await ErrorHandler.safe_execute_async(_get_experiment_results)


@router.get(
    "/experiments/{experiment_id}/status",
    response_model=ExperimentStatusResponse,
)
async def get_experiment_status(
    experiment_id: str,
    auto_strategy_service: AutoStrategyService = Depends(get_auto_strategy_service),
):
    """
    実験ステータスを取得

    実行中または完了済みの実験の現在の進行状況を取得します。
    """

    async def _get_experiment_status():
        try:
            logger.info(f"=== 実験ステータス取得API呼び出し開始 ===")
            logger.info(f"実験ID: {experiment_id}")

            # 実験ステータスを取得
            status_info = auto_strategy_service.get_experiment_status(experiment_id)

            return api_response(
                success=True,
                message="実験ステータスを取得しました",
                data=status_info,
            )
        except Exception as e:
            logger.error(f"実験ステータス取得エラー: {e}", exc_info=True)
            return error_response(
                message=f"実験ステータスの取得に失敗しました: {str(e)}",
                details={"error_type": type(e).__name__, "experiment_id": experiment_id},
            )

    return await ErrorHandler.safe_execute_async(_get_experiment_status)




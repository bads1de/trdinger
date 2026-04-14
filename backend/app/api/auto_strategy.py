"""
自動戦略生成APIエンドポイント

遺伝的アルゴリズムを用いた取引戦略の自動生成、管理、テストに関するAPIを提供します。
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import (
    get_auto_strategy_service,
)
from app.config.constants import DEFAULT_MARKET_SYMBOL
from app.services.auto_strategy import AutoStrategyService
from app.utils.error_handler import ErrorHandler
from app.utils.response import result_response

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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "experiment_name": "BTC_Strategy_Gen_001",
                "base_config": {
                    "symbol": DEFAULT_MARKET_SYMBOL,
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
                    # 多目的最適化設定（オプション）
                    "enable_multi_objective": False,
                    "objectives": ["total_return"],
                    "objective_weights": [1.0],
                },
            }
        }
    )


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


class StopExperimentResponse(BaseModel):
    """実験停止レスポンス"""
    success: bool
    message: str


class ListExperimentsResponse(BaseModel):
    """実験一覧レスポンス"""
    experiments: List[Dict[str, Any]]


class ExperimentDetailResponse(BaseModel):
    """実験詳細レスポンス"""
    id: Optional[int] = None
    experiment_id: Optional[str] = None
    name: Optional[str] = None
    status: Optional[str] = None
    progress: Optional[float] = None
    current_generation: Optional[int] = None
    total_generations: Optional[int] = None
    best_fitness: Optional[float] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


# APIエンドポイント
@router.get(
    "/experiments/{experiment_id}",
    response_model=ExperimentDetailResponse,
)
async def get_experiment_detail(
    experiment_id: str,
    auto_strategy_service: AutoStrategyService = Depends(get_auto_strategy_service),
):
    """
    実験詳細を取得

    指定された実験の進捗状況（世代数、進捗率、最高フィットネス等）を取得します。
    フロントエンドは本エンドポイントをポーリングすることでリアルタイム進捗 monitoring を実現します。

    Args:
        experiment_id: 対象実験のID
        auto_strategy_service: 自動戦略生成サービス（依存性注入）

    Returns:
        ExperimentDetailResponse: 実験詳細情報

    Raises:
        HTTPException: 実験が見つからない場合
    """
    from fastapi import HTTPException

    async def _get_detail():
        detail = auto_strategy_service.get_experiment_detail(experiment_id)
        if not detail:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"実験が見つかりません: {experiment_id}",
            )
        return ExperimentDetailResponse(**detail)

    return await ErrorHandler.safe_execute_async(_get_detail)


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
    遺伝的アルゴリズム（GA）を用いた取引戦略の自動生成を開始します。

    このエンドポイントは「非同期処理（Fire and Forget）」を採用しており、リクエストを受け付けると即座に応答を返します。
    実際の戦略生成プロセスはバックグラウンドで実行されます。

    ### 実行プロセス:
    1. リクエスト設定のバリデーション。
    2. バックグラウンドタスク（GAエンジン）のスケジューリング。
    3. `experiment_id` を返却（クライアントはこのIDを用いて進捗を追跡）。

    Args:
        request (GAGenerationRequest): 戦略生成の設定。
            - `experiment_id`: クライアント側で生成された一意のID。
            - `base_config`: 市場データ、期間、初期資金等のバックテスト条件。
            - `ga_config`: 世代数、個体数、突然変異率、目的関数等のアルゴリズム設定。
        background_tasks (BackgroundTasks): FastAPIのバックグラウンドタスク管理。
        auto_strategy_service (AutoStrategyService): GAプロセスを統括するサービス。

    Returns:
        GAGenerationResponse: タスクの受付状態と `experiment_id` を含むレスポンス。

    Note:
        進捗状況や生成された最良戦略は `/api/auto-strategy/experiments/{experiment_id}` で確認可能です。
    """

    async def _generate_strategy():
        """戦略生成のメインロジックを実行します。"""
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
                task_scheduler=background_tasks,
            )
            logger.info(f"戦略生成タスクをバックグラウンドで開始: {experiment_id}")

            return result_response(
                success=True,
                message="GA戦略生成を開始しました",
                data={"experiment_id": experiment_id},
            )
        except Exception as e:
            logger.error(f"戦略生成エラー: {e}", exc_info=True)
            # 詳細なエラー情報を返す（デバッグ用）
            return result_response(
                success=False,
                message=f"戦略生成に失敗しました: {str(e)}",
                details={"error_type": type(e).__name__, "error_details": str(e)},
                data={},
            )

    return await ErrorHandler.safe_execute_async(_generate_strategy)


@router.get("/experiments", response_model=ListExperimentsResponse)
async def list_experiments(
    auto_strategy_service: AutoStrategyService = Depends(get_auto_strategy_service),
):
    """
    実験一覧を取得

    実行中・完了済みの全GA実験の一覧を取得します。
    各実験の状態、進捗率、開始・終了時刻などの情報が含まれます。

    Args:
        auto_strategy_service: 自動戦略生成サービス（依存性注入）

    Returns:
        ListExperimentsResponse: 実験一覧情報
    """

    async def _list_experiments():
        """実験一覧を取得します。"""
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

    実行中のGA実験を安全に停止します。
    停止時点までの最良の戦略は保存されます。

    Args:
        experiment_id: 停止対象の実験ID
        auto_strategy_service: 自動戦略生成サービス（依存性注入）

    Returns:
        StopExperimentResponse: 停止処理結果

    Raises:
        HTTPException: 実験IDが存在しない場合や既に完了している場合
    """

    async def _stop_experiment():
        """指定された実験を停止します。"""
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

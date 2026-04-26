"""
ロング/ショート比率データAPIルーター
"""

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from app.api.dependencies import (
    get_long_short_ratio_repository,
    get_long_short_ratio_service,
)
from app.services.data_collection.bybit.long_short_ratio_service import (
    BybitLongShortRatioService,
)
from app.utils.datetime_utils import parse_datetime_optional
from database.repositories.long_short_ratio_repository import (
    LongShortRatioRepository,
)
from app.utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/long-short-ratio", tags=["long-short-ratio"])


@router.get("/")
@ErrorHandler.api_endpoint("ロング/ショート比率データの取得に失敗しました")
async def get_long_short_ratio_data(
    symbol: str = Query(..., description="取引ペア（例: BTC/USDT:USDT）"),
    period: str = Query(..., description="期間（例: 5min, 1h, 1d）"),
    limit: int = Query(100, ge=1, le=1000, description="取得件数"),
    start_date: Optional[str] = Query(None, description="開始日時（ISO形式）"),
    end_date: Optional[str] = Query(None, description="終了日時（ISO形式）"),
    repository: LongShortRatioRepository = Depends(
        get_long_short_ratio_repository
    ),
):
    """
    ロング/ショート比率データを取得

    データベースに保存されたロング/ショート比率データを取得します。
    期間指定または件数制限でデータをフィルタリングできます。

    Args:
        symbol: 取引ペア（例: BTC/USDT:USDT）
        period: データ期間（5min, 15min, 30min, 1h, 4h, 1d）
        limit: 取得件数（1-1000）
        start_date: 取得開始日時（ISO形式）
        end_date: 取得終了日時（ISO形式）
        repository: ロング/ショート比率リポジトリ（依存性注入）

    Returns:
        ロング/ショート比率データのリスト

    Raises:
        HTTPException: パラメータが無効な場合やデータベースエラーが発生した場合
    """
    start_dt = parse_datetime_optional(start_date)
    if start_date and start_dt is None:
        raise HTTPException(
            status_code=400, detail=f"無効なstart_dateです: {start_date}"
        )

    end_dt = parse_datetime_optional(end_date)
    if end_date and end_dt is None:
        raise HTTPException(
            status_code=400, detail=f"無効なend_dateです: {end_date}"
        )

    # データベースから取得
    records = repository.get_long_short_ratio_data(
        symbol=symbol,
        period=period,
        limit=limit,
        start_time=start_dt,
        end_time=end_dt,
    )

    # 辞書形式に変換して返却
    return [repository.to_dict(record) for record in records]


@router.post("/collect")
@ErrorHandler.api_endpoint("ロング/ショート比率データの収集開始に失敗しました")
async def collect_long_short_ratio_data(
    background_tasks: BackgroundTasks,
    symbol: str = Query(..., description="取引ペア（例: BTC/USDT:USDT）"),
    period: str = Query(..., description="期間（例: 5min, 1h, 1d）"),
    mode: str = Query(
        "incremental",
        enum=["incremental", "historical"],
        description="収集モード",
    ),
    service: BybitLongShortRatioService = Depends(
        get_long_short_ratio_service
    ),
    repository: LongShortRatioRepository = Depends(
        get_long_short_ratio_repository
    ),
):
    """
    ロング/ショート比率データの収集を実行（バックグラウンド）

    Bybit取引所からロング/ショート比率データを取得し、データベースに保存します。
    差分収集（最新データのみ）または履歴収集（全期間）のモードを選択できます。
    バックグラウンドタスクで実行されるため、即時に応答が返されます。

    Args:
        background_tasks: FastAPIバックグラウンドタスクマネージャー
        symbol: 取引ペア（例: BTC/USDT:USDT）
        period: データ期間（5min, 15min, 30min, 1h, 4h, 1d）
        mode: 収集モード（incremental: 差分収集、historical: 履歴収集）
        service: Bybitロング/ショート比率サービス（依存性注入）
        repository: ロング/ショート比率リポジトリ（依存性注入）

    Returns:
        収集開始通知を含むJSONレスポンス
    """

    async def _collect_task():
        try:
            if mode == "incremental":
                result = await service.fetch_incremental_long_short_ratio_data(
                    symbol, period, repository
                )
                logger.info(f"差分収集完了: {result}")
            else:
                # 履歴収集（全期間）
                # 現在は簡易的にサービスを呼び出す（実運用では非同期タスク管理が必要かも）
                count = await service.collect_historical_long_short_ratio_data(
                    symbol, period, repository
                )
                logger.info(f"履歴収集完了: {count}件")
        except Exception as e:
            logger.error(f"データ収集バックグラウンドタスクエラー: {e}")

    # バックグラウンドタスクとして登録
    background_tasks.add_task(_collect_task)

    return {
        "message": f"データ収集タスクを開始しました (mode: {mode})",
        "symbol": symbol,
    }

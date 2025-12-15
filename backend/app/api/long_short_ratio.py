"""
ロング/ショート比率データAPIルーター
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from app.api.dependencies import (
    get_long_short_ratio_repository,
    get_long_short_ratio_service,
)
from app.services.data_collection.bybit.long_short_ratio_service import (
    BybitLongShortRatioService,
)

from database.repositories.long_short_ratio_repository import LongShortRatioRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/long-short-ratio", tags=["long-short-ratio"])


@router.get("/")
async def get_long_short_ratio_data(
    symbol: str = Query(..., description="取引ペア（例: BTC/USDT:USDT）"),
    period: str = Query(..., description="期間（例: 5min, 1h, 1d）"),
    limit: int = Query(100, ge=1, le=1000, description="取得件数"),
    start_date: Optional[str] = Query(None, description="開始日時（ISO形式）"),
    end_date: Optional[str] = Query(None, description="終了日時（ISO形式）"),
    repository: LongShortRatioRepository = Depends(get_long_short_ratio_repository),
):
    """
    ロング/ショート比率データを取得
    """
    try:
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

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

    except Exception as e:
        logger.error(f"データ取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collect")
async def collect_long_short_ratio_data(
    background_tasks: BackgroundTasks,
    symbol: str = Query(..., description="取引ペア（例: BTC/USDT:USDT）"),
    period: str = Query(..., description="期間（例: 5min, 1h, 1d）"),
    mode: str = Query(
        "incremental", enum=["incremental", "historical"], description="収集モード"
    ),
    service: BybitLongShortRatioService = Depends(get_long_short_ratio_service),
    repository: LongShortRatioRepository = Depends(get_long_short_ratio_repository),
):
    """
    ロング/ショート比率データの収集を実行（バックグラウンド）
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



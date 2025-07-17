"""
外部市場データ API エンドポイント

yfinance APIから取得した外部市場データ（SP500、NASDAQ、DXY、VIXなど）の
取得、収集、管理機能を提供します。
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database.connection import get_db
from database.repositories.external_market_repository import ExternalMarketRepository
from data_collector.external_market_collector import ExternalMarketDataCollector
from app.core.utils.api_utils import APIResponseHelper

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/external-market", tags=["external-market"])


@router.get("/")
async def get_external_market_data(
    symbol: Optional[str] = Query(None, description="シンボル（例: ^GSPC, ^IXIC）"),
    start_time: Optional[str] = Query(None, description="開始時刻（ISO形式）"),
    end_time: Optional[str] = Query(None, description="終了時刻（ISO形式）"),
    limit: int = Query(100, description="取得件数制限"),
    db: Session = Depends(get_db),
) -> Dict:
    """
    外部市場データを取得

    Args:
        symbol: シンボル（指定しない場合は全シンボル）
        start_time: 開始時刻
        end_time: 終了時刻
        limit: 取得件数制限
        db: データベースセッション

    Returns:
        外部市場データのリスト
    """
    try:
        repository = ExternalMarketRepository(db)

        # 時刻の変換
        start_datetime = None
        end_datetime = None
        
        if start_time:
            try:
                start_datetime = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            except ValueError as e:
                raise HTTPException(
                    status_code=400, detail=f"開始時刻の形式が無効です: {e}"
                )

        if end_time:
            try:
                end_datetime = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            except ValueError as e:
                raise HTTPException(
                    status_code=400, detail=f"終了時刻の形式が無効です: {e}"
                )

        # データ取得
        data = repository.get_external_market_data(
            symbol=symbol,
            start_time=start_datetime,
            end_time=end_datetime,
            limit=limit,
        )

        # 辞書形式に変換
        data_dicts = [record.to_dict() for record in data]

        return APIResponseHelper.api_response(
            success=True,
            message=f"外部市場データを {len(data_dicts)} 件取得しました",
            data=data_dicts,
        )

    except Exception as e:
        logger.error(f"外部市場データ取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest")
async def get_latest_external_market_data(
    symbol: Optional[str] = Query(None, description="シンボル（例: ^GSPC, ^IXIC）"),
    limit: int = Query(30, description="取得件数制限"),
    db: Session = Depends(get_db),
) -> Dict:
    """
    最新の外部市場データを取得

    Args:
        symbol: シンボル（指定しない場合は全シンボル）
        limit: 取得件数制限
        db: データベースセッション

    Returns:
        最新の外部市場データのリスト
    """
    try:
        repository = ExternalMarketRepository(db)

        data = repository.get_latest_external_market_data(symbol=symbol, limit=limit)

        # 辞書形式に変換
        data_dicts = [record.to_dict() for record in data]

        return APIResponseHelper.api_response(
            success=True,
            message=f"最新の外部市場データを {len(data_dicts)} 件取得しました",
            data=data_dicts,
        )

    except Exception as e:
        logger.error(f"最新外部市場データ取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols")
async def get_available_symbols(db: Session = Depends(get_db)) -> Dict:
    """
    利用可能なシンボルの一覧を取得

    Args:
        db: データベースセッション

    Returns:
        シンボル一覧
    """
    try:
        repository = ExternalMarketRepository(db)
        
        # データベースに存在するシンボル
        db_symbols = repository.get_symbols()
        
        # サービスで定義されているシンボル
        from app.core.services.data_collection.external_market_service import ExternalMarketService
        service = ExternalMarketService()
        available_symbols = service.get_available_symbols()

        return APIResponseHelper.api_response(
            success=True,
            message="シンボル一覧を取得しました",
            data={
                "available_symbols": available_symbols,
                "database_symbols": db_symbols,
            },
        )

    except Exception as e:
        logger.error(f"シンボル一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_external_market_data_status(db: Session = Depends(get_db)) -> Dict:
    """
    外部市場データの状態を取得

    Args:
        db: データベースセッション

    Returns:
        データ状態情報
    """
    try:
        async with ExternalMarketDataCollector() as collector:
            status = await collector.get_external_market_data_status(db_session=db)

        if status["success"]:
            return APIResponseHelper.api_response(
                success=True,
                message="外部市場データの状態を取得しました",
                data=status,
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"データ状態取得に失敗しました: {status.get('error', 'Unknown error')}",
            )

    except Exception as e:
        logger.error(f"外部市場データ状態取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collect")
async def collect_external_market_data(
    symbols: Optional[List[str]] = Query(None, description="取得するシンボルのリスト"),
    period: str = Query("1mo", description="取得期間"),
    db: Session = Depends(get_db),
) -> Dict:
    """
    外部市場データを収集

    Args:
        symbols: 取得するシンボルのリスト
        period: 取得期間
        db: データベースセッション

    Returns:
        収集結果
    """
    try:
        async with ExternalMarketDataCollector() as collector:
            result = await collector.collect_external_market_data(
                symbols=symbols, period=period, db_session=db
            )

        if result["success"]:
            return APIResponseHelper.api_response(
                success=True,
                message=result["message"],
                data=result,
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"データ収集に失敗しました: {result.get('error', 'Unknown error')}",
            )

    except Exception as e:
        logger.error(f"外部市場データ収集エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collect-incremental")
async def collect_incremental_external_market_data(
    symbols: Optional[List[str]] = Query(None, description="取得するシンボルのリスト"),
    db: Session = Depends(get_db),
) -> Dict:
    """
    外部市場データの差分データを収集

    Args:
        symbols: 取得するシンボルのリスト
        db: データベースセッション

    Returns:
        差分収集結果
    """
    try:
        async with ExternalMarketDataCollector() as collector:
            result = await collector.collect_incremental_external_market_data(
                symbols=symbols, db_session=db
            )

        if result["success"]:
            return APIResponseHelper.api_response(
                success=True,
                message=result["message"],
                data=result,
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"差分データ収集に失敗しました: {result.get('error', 'Unknown error')}",
            )

    except Exception as e:
        logger.error(f"外部市場データ差分収集エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collect-historical")
async def collect_historical_external_market_data(
    symbols: Optional[List[str]] = Query(None, description="取得するシンボルのリスト"),
    period: str = Query("2y", description="取得期間"),
    db: Session = Depends(get_db),
) -> Dict:
    """
    外部市場データの履歴データを収集

    Args:
        symbols: 取得するシンボルのリスト
        period: 取得期間
        db: データベースセッション

    Returns:
        履歴収集結果
    """
    try:
        async with ExternalMarketDataCollector() as collector:
            result = await collector.collect_historical_external_market_data(
                symbols=symbols, period=period, db_session=db
            )

        if result["success"]:
            return APIResponseHelper.api_response(
                success=True,
                message=result["message"],
                data=result,
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"履歴データ収集に失敗しました: {result.get('error', 'Unknown error')}",
            )

    except Exception as e:
        logger.error(f"外部市場データ履歴収集エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

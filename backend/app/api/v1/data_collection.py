"""
データ収集API

バックテスト用のOHLCVデータ収集エンドポイント
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict

from app.core.services.historical_data_service import HistoricalDataService
from database.connection import get_db
from database.repository import OHLCVRepository
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data-collection", tags=["data-collection"])


@router.post("/historical")
async def collect_historical_data(
    background_tasks: BackgroundTasks,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    db: Session = Depends(get_db)
) -> Dict:
    """
    履歴データを包括的に収集
    
    Args:
        symbol: 取引ペア（デフォルト: BTC/USDT）
        timeframe: 時間軸（デフォルト: 1h）
        db: データベースセッション
        
    Returns:
        収集開始レスポンス
    """
    try:
        # バックグラウンドタスクとして実行
        background_tasks.add_task(
            _collect_historical_background,
            symbol,
            timeframe,
            db
        )
        
        return {
            "success": True,
            "message": f"{symbol} {timeframe} の履歴データ収集を開始しました",
            "status": "started"
        }
        
    except Exception as e:
        logger.error(f"履歴データ収集開始エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update")
async def update_incremental_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    db: Session = Depends(get_db)
) -> Dict:
    """
    差分データを更新
    
    Args:
        symbol: 取引ペア（デフォルト: BTC/USDT）
        timeframe: 時間軸（デフォルト: 1h）
        db: データベースセッション
        
    Returns:
        更新結果
    """
    try:
        service = HistoricalDataService()
        repository = OHLCVRepository(db)
        
        result = await service.collect_incremental_data(symbol, timeframe, repository)
        
        return result
        
    except Exception as e:
        logger.error(f"差分データ更新エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bitcoin-full")
async def collect_bitcoin_full_data(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> Dict:
    """
    ビットコインの全時間軸データを収集（ベータ版機能）
    
    Args:
        db: データベースセッション
        
    Returns:
        収集開始レスポンス
    """
    try:
        # 全時間軸でビットコインデータを収集
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        
        for timeframe in timeframes:
            background_tasks.add_task(
                _collect_historical_background,
                "BTC/USDT",
                timeframe,
                db
            )
        
        return {
            "success": True,
            "message": "ビットコインの全時間軸データ収集を開始しました",
            "timeframes": timeframes,
            "status": "started"
        }
        
    except Exception as e:
        logger.error(f"ビットコイン全データ収集開始エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{symbol}/{timeframe}")
async def get_collection_status(
    symbol: str,
    timeframe: str,
    db: Session = Depends(get_db)
) -> Dict:
    """
    データ収集状況を確認
    
    Args:
        symbol: 取引ペア
        timeframe: 時間軸
        db: データベースセッション
        
    Returns:
        データ収集状況
    """
    try:
        repository = OHLCVRepository(db)
        
        # データ件数を取得
        data_count = repository.get_data_count(symbol, timeframe)
        
        # 最新・最古タイムスタンプを取得
        latest_timestamp = repository.get_latest_timestamp(symbol, timeframe)
        oldest_timestamp = repository.get_oldest_timestamp(symbol, timeframe)
        
        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "data_count": data_count,
            "latest_timestamp": latest_timestamp.isoformat() if latest_timestamp else None,
            "oldest_timestamp": oldest_timestamp.isoformat() if oldest_timestamp else None
        }
        
    except Exception as e:
        logger.error(f"収集状況確認エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _collect_historical_background(symbol: str, timeframe: str, db: Session):
    """バックグラウンドでの履歴データ収集"""
    try:
        service = HistoricalDataService()
        repository = OHLCVRepository(db)
        
        result = await service.collect_historical_data(symbol, timeframe, repository)
        
        if result["success"]:
            logger.info(f"バックグラウンド収集完了: {symbol} {timeframe} - {result['saved_count']}件保存")
        else:
            logger.error(f"バックグラウンド収集失敗: {symbol} {timeframe} - {result.get('message')}")
            
    except Exception as e:
        logger.error(f"バックグラウンド収集エラー: {symbol} {timeframe} - {e}")
    finally:
        db.close()

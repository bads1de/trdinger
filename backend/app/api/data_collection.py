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
from app.config.market_config import MarketDataConfig  # 追加
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data-collection", tags=["data-collection"])


@router.post("/historical")
async def collect_historical_data(
    background_tasks: BackgroundTasks,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    db: Session = Depends(get_db),
) -> Dict:
    """
    履歴データを包括的に収集
    データベースにデータが存在しない場合のみ新規収集を行います。

    Args:
        symbol: 取引ペア（デフォルト: BTC/USDT）
        timeframe: 時間軸（デフォルト: 1h）
        db: データベースセッション

    Returns:
        収集開始レスポンスまたは既存データ情報
    """
    try:
        # シンボルと時間軸のバリデーション (market_config を使用)
        try:
            normalized_symbol = MarketDataConfig.normalize_symbol(symbol)
            if not MarketDataConfig.validate_timeframe(timeframe):
                raise ValueError(f"無効な時間軸: {timeframe}")
        except ValueError as ve:
            logger.warning(
                f"パラメータ検証エラー: {ve} (symbol: {symbol}, timeframe: {timeframe})"
            )
            # ここでログに出力されるエラーメッセージが問題の "サポートされていない通貨ペアです。利用可能: BTC/USDT" と一致するか確認
            # MarketDataConfig.normalize_symbol から raise されるエラーはもっと詳細なはず
            # もしこの ValueError が原因でなければ、他の箇所でエラーが発生している
            raise HTTPException(status_code=400, detail=str(ve))

        repository = OHLCVRepository(db)
        data_exists = repository.get_data_count(normalized_symbol, timeframe) > 0

        if data_exists:
            logger.info(
                f"{normalized_symbol} {timeframe} のデータは既にデータベースに存在します。"
            )
            return {
                "success": True,
                "message": f"{normalized_symbol} {timeframe} のデータは既に存在します。新規収集は行いません。",
                "status": "exists",
            }

        # バックグラウンドタスクとして実行
        background_tasks.add_task(
            _collect_historical_background,
            normalized_symbol,  # 正規化されたシンボルを使用
            timeframe,
            db,
        )

        return {
            "success": True,
            "message": f"{normalized_symbol} {timeframe} の履歴データ収集を開始しました",
            "status": "started",
        }

    except HTTPException:  # HTTPException はそのまま re-raise
        raise
    except Exception as e:
        logger.error(f"履歴データ収集開始エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update")
async def update_incremental_data(
    symbol: str = "BTC/USDT", timeframe: str = "1h", db: Session = Depends(get_db)
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
    background_tasks: BackgroundTasks, db: Session = Depends(get_db)
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
                _collect_historical_background, "BTC/USDT", timeframe, db
            )

        return {
            "success": True,
            "message": "ビットコインの全時間軸データ収集を開始しました",
            "timeframes": timeframes,
            "status": "started",
        }

    except Exception as e:
        logger.error(f"ビットコイン全データ収集開始エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{symbol:path}/{timeframe}")
async def get_collection_status(
    symbol: str,
    timeframe: str,
    auto_fetch: bool = False,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
) -> Dict:
    """
    データ収集状況を確認

    Args:
        symbol: 取引ペア
        timeframe: 時間軸
        auto_fetch: データが存在しない場合に自動フェッチを開始するか
        background_tasks: バックグラウンドタスク
        db: データベースセッション

    Returns:
        データ収集状況
    """
    try:
        # シンボルと時間軸のバリデーション
        try:
            normalized_symbol = MarketDataConfig.normalize_symbol(symbol)
            if not MarketDataConfig.validate_timeframe(timeframe):
                raise ValueError(f"無効な時間軸: {timeframe}")
        except ValueError as ve:
            logger.warning(
                f"パラメータ検証エラー: {ve} (symbol: {symbol}, timeframe: {timeframe})"
            )
            raise HTTPException(status_code=400, detail=str(ve))

        repository = OHLCVRepository(db)

        # 正規化されたシンボルでデータ件数を取得
        data_count = repository.get_data_count(normalized_symbol, timeframe)

        # データが存在しない場合の処理
        if data_count == 0:
            if auto_fetch and background_tasks:
                # 自動フェッチを開始
                background_tasks.add_task(
                    _collect_historical_background,
                    normalized_symbol,
                    timeframe,
                    db,
                )
                logger.info(f"自動フェッチを開始: {normalized_symbol} {timeframe}")

                return {
                    "success": True,
                    "symbol": normalized_symbol,
                    "original_symbol": symbol,
                    "timeframe": timeframe,
                    "data_count": 0,
                    "status": "auto_fetch_started",
                    "message": f"{normalized_symbol} {timeframe} のデータが存在しないため、自動収集を開始しました。",
                }
            else:
                # フェッチを提案
                return {
                    "success": True,
                    "symbol": normalized_symbol,
                    "original_symbol": symbol,
                    "timeframe": timeframe,
                    "data_count": 0,
                    "status": "no_data",
                    "message": f"{normalized_symbol} {timeframe} のデータが存在しません。新規収集が必要です。",
                    "suggestion": {
                        "manual_fetch": f"/api/data-collection/historical?symbol={normalized_symbol}&timeframe={timeframe}",
                        "auto_fetch": f"/api/data-collection/status/{symbol}/{timeframe}?auto_fetch=true",
                    },
                }

        # 最新・最古タイムスタンプを取得
        latest_timestamp = repository.get_latest_timestamp(normalized_symbol, timeframe)
        oldest_timestamp = repository.get_oldest_timestamp(normalized_symbol, timeframe)

        return {
            "success": True,
            "symbol": normalized_symbol,
            "original_symbol": symbol,
            "timeframe": timeframe,
            "data_count": data_count,
            "status": "data_exists",
            "latest_timestamp": (
                latest_timestamp.isoformat() if latest_timestamp else None
            ),
            "oldest_timestamp": (
                oldest_timestamp.isoformat() if oldest_timestamp else None
            ),
        }

    except HTTPException:  # HTTPException はそのまま re-raise
        raise
    except Exception as e:
        logger.error(f"収集状況確認エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supported-symbols")
async def get_supported_symbols() -> Dict:
    """
    サポートされている取引ペアの一覧を取得

    Returns:
        サポートされている取引ペアの一覧
    """
    try:
        return {
            "success": True,
            "symbols": MarketDataConfig.SUPPORTED_SYMBOLS,
            "timeframes": MarketDataConfig.SUPPORTED_TIMEFRAMES,
            "default_symbol": MarketDataConfig.DEFAULT_SYMBOL,
            "default_timeframe": MarketDataConfig.DEFAULT_TIMEFRAME,
            "message": "サポートされている取引ペアと時間軸の一覧です。",
        }

    except Exception as e:
        logger.error(f"サポートシンボル一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _collect_historical_background(symbol: str, timeframe: str, db: Session):
    """バックグラウンドでの履歴データ収集"""
    try:
        service = HistoricalDataService()
        repository = OHLCVRepository(db)

        result = await service.collect_historical_data(symbol, timeframe, repository)

        if result["success"]:
            logger.info(
                f"バックグラウンド収集完了: {symbol} {timeframe} - {result['saved_count']}件保存"
            )
        else:
            logger.error(
                f"バックグラウンド収集失敗: {symbol} {timeframe} - {result.get('message')}"
            )

    except Exception as e:
        logger.error(f"バックグラウンド収集エラー: {symbol} {timeframe} - {e}")
    finally:
        db.close()

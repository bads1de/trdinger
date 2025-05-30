"""
オープンインタレストAPIエンドポイント

ファンディングレートAPIの実装パターンを参考に、
オープンインタレスト（建玉残高）データの取得・収集機能を提供します。
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.services.open_interest_service import BybitOpenInterestService
from database.connection import get_db, ensure_db_initialized
from database.repositories.open_interest_repository import OpenInterestRepository

# ログ設定
logger = logging.getLogger(__name__)

# ルーター作成
router = APIRouter(tags=["open-interest"])


@router.get("/open-interest")
async def get_open_interest_data(
    symbol: str = Query(..., description="取引ペアシンボル（例: 'BTC/USDT'）"),
    start_date: Optional[str] = Query(None, description="開始日時（ISO形式）"),
    end_date: Optional[str] = Query(None, description="終了日時（ISO形式）"),
    limit: Optional[int] = Query(1000, description="取得件数制限（最大1000）"),
    db: Session = Depends(get_db),
):
    """
    オープンインタレストデータを取得します

    データベースに保存されているオープンインタレストデータを指定された条件で取得します。

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USDT'）
        start_date: 開始日時（ISO形式、例: '2024-01-01T00:00:00Z'）
        end_date: 終了日時（ISO形式、例: '2024-01-31T23:59:59Z'）
        limit: 取得件数制限（1-1000）
        db: データベースセッション

    Returns:
        オープンインタレストデータを含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やデータベースエラーが発生した場合
    """
    try:
        logger.info(
            f"オープンインタレストデータ取得リクエスト: symbol={symbol}, limit={limit}"
        )

        # データベースリポジトリを作成
        repository = OpenInterestRepository(db)

        # 日付パラメータの変換
        start_time = None
        end_time = None

        if start_date:
            start_time = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        if end_date:
            end_time = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

        # シンボルの正規化
        service = BybitOpenInterestService()
        normalized_symbol = service.normalize_symbol(symbol)

        # データベースからオープンインタレストデータを取得
        open_interest_records = repository.get_open_interest_data(
            symbol=normalized_symbol,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

        # レスポンス形式に変換
        open_interest_data = []
        for record in open_interest_records:
            open_interest_data.append(
                {
                    "symbol": record.symbol,
                    "open_interest_value": record.open_interest_value,
                    "data_timestamp": record.data_timestamp.isoformat(),
                    "timestamp": record.timestamp.isoformat(),
                }
            )

        logger.info(f"オープンインタレストデータ取得成功: {len(open_interest_data)}件")

        return {
            "success": True,
            "data": {
                "symbol": normalized_symbol,
                "count": len(open_interest_data),
                "open_interest": open_interest_data,
            },
            "message": f"{len(open_interest_data)}件のオープンインタレストデータを取得しました",
        }

    except Exception as e:
        logger.error(f"オープンインタレストデータ取得エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"オープンインタレストデータの取得中にエラーが発生しました: {str(e)}",
        )


@router.post("/open-interest/collect")
async def collect_open_interest_data(
    symbol: str = Query(..., description="取引ペアシンボル（例: 'BTC/USDT'）"),
    limit: Optional[int] = Query(
        100, description="取得するデータ数（1-1000、fetch_all=trueの場合は無視）"
    ),
    fetch_all: bool = Query(False, description="全期間のデータを取得するかどうか"),
    db: Session = Depends(get_db),
):
    """
    オープンインタレストデータを収集してデータベースに保存します

    Bybit取引所からオープンインタレストデータを取得し、データベースに保存します。

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USDT'）
        limit: 取得するデータ数（1-1000、fetch_all=trueの場合は無視）
        fetch_all: 全期間のデータを取得するかどうか
        db: データベースセッション

    Returns:
        収集結果を含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やAPI/データベースエラーが発生した場合
    """
    try:
        logger.info(
            f"オープンインタレストデータ収集開始: symbol={symbol}, fetch_all={fetch_all}"
        )

        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        # オープンインタレストサービスを作成
        service = BybitOpenInterestService()

        # データベースリポジトリを作成
        repository = OpenInterestRepository(db)

        # オープンインタレストデータを取得・保存
        result = await service.fetch_and_save_open_interest_data(
            symbol=symbol,
            limit=limit,
            repository=repository,
            fetch_all=fetch_all,
        )

        logger.info(f"オープンインタレストデータ収集完了: {result}")

        return {
            "success": True,
            "data": result,
            "message": f"{result['saved_count']}件のオープンインタレストデータを保存しました",
        }

    except Exception as e:
        logger.error(f"オープンインタレストデータ収集エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"オープンインタレストデータの収集中にエラーが発生しました: {str(e)}",
        )


@router.get("/open-interest/current")
async def get_current_open_interest(
    symbol: str = Query(..., description="取引ペアシンボル（例: 'BTC/USDT'）"),
):
    """
    現在のオープンインタレストを取得します

    Bybit取引所から現在のオープンインタレストデータを直接取得します。

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USDT'）

    Returns:
        現在のオープンインタレストデータを含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やAPIエラーが発生した場合
    """
    try:
        logger.info(f"現在のオープンインタレスト取得開始: symbol={symbol}")

        # オープンインタレストサービスを作成
        service = BybitOpenInterestService()

        # 現在のオープンインタレストを取得
        current_open_interest = await service.fetch_current_open_interest(symbol)

        logger.info(f"現在のオープンインタレスト取得完了: {current_open_interest}")

        return {
            "success": True,
            "data": current_open_interest,
            "message": f"{symbol}の現在のオープンインタレストを取得しました",
        }

    except Exception as e:
        logger.error(f"現在のオープンインタレスト取得エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"現在のオープンインタレストの取得中にエラーが発生しました: {str(e)}",
        )


@router.post("/open-interest/bulk-collect")
async def bulk_collect_open_interest(
    db: Session = Depends(get_db),
):
    """
    BTC・ETHシンボルのオープンインタレストデータを一括収集します

    BTC・ETHの無期限契約シンボルの全期間オープンインタレストデータを一括で取得・保存します。

    Args:
        db: データベースセッション

    Returns:
        一括収集結果を含むJSONレスポンス

    Raises:
        HTTPException: データベースエラーが発生した場合
    """
    try:
        logger.info("オープンインタレスト一括収集開始: BTC・ETH全期間データ")

        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        # BTC・ETHの無期限契約シンボル（USDTのみ）
        symbols = [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
        ]

        # オープンインタレストサービスを作成
        service = BybitOpenInterestService()
        repository = OpenInterestRepository(db)

        # 各シンボルのデータを収集
        results = []
        total_saved = 0
        successful_symbols = 0
        failed_symbols = []

        for symbol in symbols:
            try:
                result = await service.fetch_and_save_open_interest_data(
                    symbol=symbol,
                    repository=repository,
                    fetch_all=True,  # 全期間のデータを取得
                )
                results.append(result)
                total_saved += result["saved_count"]
                successful_symbols += 1

                logger.info(f"✅ {symbol}: {result['saved_count']}件保存")

                # レート制限対応
                import asyncio

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"❌ {symbol} 収集エラー: {e}")
                failed_symbols.append({"symbol": symbol, "error": str(e)})

        logger.info(
            f"オープンインタレスト一括収集完了: {successful_symbols}/{len(symbols)}成功"
        )

        return {
            "success": True,
            "data": {
                "results": results,
                "summary": {
                    "total_symbols": len(symbols),
                    "successful_symbols": successful_symbols,
                    "failed_symbols": len(failed_symbols),
                    "total_saved": total_saved,
                },
                "failed_symbols": failed_symbols,
            },
            "message": f"{successful_symbols}/{len(symbols)}シンボルで合計{total_saved}件のオープンインタレストデータを保存しました",
        }

    except Exception as e:
        logger.error(f"オープンインタレスト一括収集エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"オープンインタレスト一括収集中にエラーが発生しました: {str(e)}",
        )

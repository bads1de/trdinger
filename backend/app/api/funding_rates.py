"""
ファンディングレートAPI

ファンディングレートデータの取得・収集機能を提供するAPIエンドポイント
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
import logging

from database.connection import get_db, ensure_db_initialized
from database.repositories.funding_rate_repository import FundingRateRepository
from app.core.services.funding_rate_service import BybitFundingRateService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["funding-rates"])


@router.get("/funding-rates")
async def get_funding_rates(
    symbol: str = Query(..., description="取引ペアシンボル（例: 'BTC/USDT'）"),
    limit: Optional[int] = Query(100, description="取得するデータ数（1-1000）"),
    start_date: Optional[str] = Query(None, description="開始日時（ISO形式）"),
    end_date: Optional[str] = Query(None, description="終了日時（ISO形式）"),
    db: Session = Depends(get_db),
):
    """
    ファンディングレートデータを取得します

    データベースに保存されたファンディングレートデータを取得します。

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USDT'）
        limit: 取得するデータ数（1-1000）
        start_date: 開始日時（ISO形式）
        end_date: 終了日時（ISO形式）
        db: データベースセッション

    Returns:
        ファンディングレートデータを含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やデータベースエラーが発生した場合
    """
    try:
        logger.info(
            f"ファンディングレートデータ取得リクエスト: symbol={symbol}, limit={limit}"
        )

        # データベースリポジトリを作成
        repository = FundingRateRepository(db)

        # 日付パラメータの変換
        start_time = None
        end_time = None

        if start_date:
            start_time = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        if end_date:
            end_time = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

        # シンボルの正規化
        service = BybitFundingRateService()
        normalized_symbol = service.normalize_symbol(symbol)

        # データベースからファンディングレートデータを取得
        funding_rate_records = repository.get_funding_rate_data(
            symbol=normalized_symbol,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

        # レスポンス形式に変換
        funding_rates = []
        for record in funding_rate_records:
            funding_rates.append(
                {
                    "symbol": record.symbol,
                    "funding_rate": record.funding_rate,
                    "funding_timestamp": record.funding_timestamp.isoformat(),
                    "timestamp": record.timestamp.isoformat(),
                    "next_funding_timestamp": (
                        record.next_funding_timestamp.isoformat()
                        if record.next_funding_timestamp
                        else None
                    ),
                    "mark_price": record.mark_price,
                    "index_price": record.index_price,
                }
            )

        logger.info(f"ファンディングレートデータ取得成功: {len(funding_rates)}件")

        return {
            "success": True,
            "data": {
                "symbol": normalized_symbol,
                "count": len(funding_rates),
                "funding_rates": funding_rates,
            },
            "message": f"{len(funding_rates)}件のファンディングレートデータを取得しました",
        }

    except Exception as e:
        logger.error(f"ファンディングレートデータ取得エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"ファンディングレートデータの取得中にエラーが発生しました: {str(e)}",
        )


@router.post("/funding-rates/collect")
async def collect_funding_rate_data(
    symbol: str = Query(..., description="取引ペアシンボル（例: 'BTC/USDT'）"),
    limit: Optional[int] = Query(
        100, description="取得するデータ数（1-1000、fetch_all=trueの場合は無視）"
    ),
    fetch_all: bool = Query(False, description="全期間のデータを取得するかどうか"),
    db: Session = Depends(get_db),
):
    """
    ファンディングレートデータを収集してデータベースに保存します

    Bybit取引所からファンディングレートデータを取得し、データベースに保存します。

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
            f"ファンディングレートデータ収集開始: symbol={symbol}, fetch_all={fetch_all}"
        )

        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        # ファンディングレートサービスを作成
        service = BybitFundingRateService()

        # データベースリポジトリを作成
        repository = FundingRateRepository(db)

        # ファンディングレートデータを取得・保存
        result = await service.fetch_and_save_funding_rate_data(
            symbol=symbol,
            limit=limit,
            repository=repository,
            fetch_all=fetch_all,
        )

        logger.info(f"ファンディングレートデータ収集完了: {result}")

        return {
            "success": True,
            "data": result,
            "message": f"{result['saved_count']}件のファンディングレートデータを保存しました",
        }

    except Exception as e:
        logger.error(f"ファンディングレートデータ収集エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"ファンディングレートデータの収集中にエラーが発生しました: {str(e)}",
        )


@router.get("/funding-rates/current")
async def get_current_funding_rate(
    symbol: str = Query(..., description="取引ペアシンボル（例: 'BTC/USDT'）"),
):
    """
    現在のファンディングレートを取得します

    Bybit取引所から現在のファンディングレートをリアルタイムで取得します。

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USDT'）

    Returns:
        現在のファンディングレートを含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やAPIエラーが発生した場合
    """
    try:
        logger.info(f"現在のファンディングレート取得開始: symbol={symbol}")

        # ファンディングレートサービスを作成
        service = BybitFundingRateService()

        # 現在のファンディングレートを取得
        current_rate = await service.fetch_current_funding_rate(symbol)

        logger.info(f"現在のファンディングレート取得完了: {symbol}")

        return {
            "success": True,
            "data": {
                "symbol": current_rate["symbol"],
                "funding_rate": current_rate["fundingRate"],
                "funding_timestamp": current_rate.get("fundingDatetime"),
                "next_funding_timestamp": current_rate.get("nextFundingDatetime"),
                "mark_price": current_rate.get("markPrice"),
                "index_price": current_rate.get("indexPrice"),
                "timestamp": current_rate.get("datetime"),
            },
            "message": f"{symbol}の現在のファンディングレートを取得しました",
        }

    except Exception as e:
        logger.error(f"現在のファンディングレート取得エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"現在のファンディングレートの取得中にエラーが発生しました: {str(e)}",
        )


@router.post("/funding-rates/bulk-collect")
async def bulk_collect_funding_rates(
    db: Session = Depends(get_db),
):
    """
    BTCシンボルのファンディングレートデータを一括収集します

    BTCの無期限契約シンボルの全期間ファンディングレートデータを一括で取得・保存します。
    ETHは分析対象から除外されています。

    Args:
        db: データベースセッション

    Returns:
        一括収集結果を含むJSONレスポンス

    Raises:
        HTTPException: データベースエラーが発生した場合
    """
    try:
        logger.info("ファンディングレート一括収集開始: BTC全期間データ")

        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        # BTCの無期限契約シンボル（USDTのみ、ETHは除外）
        symbols = [
            "BTC/USDT:USDT",
        ]

        # ファンディングレートサービスを作成
        service = BybitFundingRateService()
        repository = FundingRateRepository(db)

        # 各シンボルのデータを収集
        results = []
        total_saved = 0
        successful_symbols = 0
        failed_symbols = []

        for symbol in symbols:
            try:
                result = await service.fetch_and_save_funding_rate_data(
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
            f"ファンディングレート一括収集完了: {successful_symbols}/{len(symbols)}成功"
        )

        return {
            "success": True,
            "data": {
                "total_symbols": len(symbols),
                "successful_symbols": successful_symbols,
                "failed_symbols": len(failed_symbols),
                "total_saved_records": total_saved,
                "results": results,
                "failures": failed_symbols,
            },
            "message": f"{successful_symbols}/{len(symbols)}シンボル（BTC）で合計{total_saved}件のファンディングレートデータを保存しました",
        }

    except Exception as e:
        logger.error(f"ファンディングレート一括収集エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"ファンディングレートの一括収集中にエラーが発生しました: {str(e)}",
        )

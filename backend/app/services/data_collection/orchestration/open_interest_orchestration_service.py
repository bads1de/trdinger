"""
オープンインタレストデータ収集統合管理サービス

APIルーター内に散在していたオープンインタレスト関連のビジネスロジックを統合管理します。
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.utils.response import api_response, error_response
from database.repositories.open_interest_repository import OpenInterestRepository

from ..bybit.open_interest_service import BybitOpenInterestService

logger = logging.getLogger(__name__)


class OpenInterestOrchestrationService:
    """
    オープンインタレストデータ収集統合管理サービス

    オープンインタレストデータの収集、取得、一括処理等の
    統一的な処理を担当します。APIルーターからビジネスロジックを分離し、
    責務を明確化します。
    """

    def __init__(self):
        """初期化"""

    async def collect_open_interest_data(
        self,
        symbol: str,
        limit: Optional[int] = None,
        fetch_all: bool = False,
        db_session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        オープンインタレストデータの収集

        Args:
            symbol: 取引ペア
            limit: 取得件数制限（オプション）
            fetch_all: 全データ取得フラグ
            db_session: データベースセッション

        Returns:
            収集結果を含む辞書
        """
        try:
            logger.info(
                f"オープンインタレストデータ収集開始: symbol={symbol}, fetch_all={fetch_all}"
            )

            # データベースセッションの取得
            if db_session is None:
                from database.connection import get_db

                with next(get_db()) as session:
                    db_session = session

            # サービスとリポジトリの初期化
            service = BybitOpenInterestService()
            repository = OpenInterestRepository(db_session)

            # データ収集実行
            result = await service.fetch_and_save_open_interest_data(
                symbol=symbol,
                limit=limit,
                repository=repository,
                fetch_all=fetch_all,
            )

            if result.get("success", False):
                return api_response(
                    success=True,
                    message=f"{result['saved_count']}件のオープンインタレストデータを保存しました",
                    data={
                        "saved_count": result["saved_count"],
                        "symbol": symbol,
                        "fetch_all": fetch_all,
                        "limit": limit,
                    },
                )
            else:
                return error_response(
                    message=f"オープンインタレストデータ収集に失敗しました: {result.get('error', 'Unknown error')}",
                    details={
                        "saved_count": result.get("saved_count", 0),
                        "symbol": symbol,
                        "error": result.get("error"),
                    },
                )

        except Exception as e:
            logger.error(f"オープンインタレストデータ収集エラー: {e}", exc_info=True)
            return error_response(
                message=f"オープンインタレストデータ収集中にエラーが発生しました: {str(e)}",
                details={
                    "saved_count": 0,
                    "symbol": symbol,
                    "error": str(e),
                },
            )

    async def get_open_interest_data(
        self,
        symbol: str,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        db_session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        オープンインタレストデータの取得
        """
        try:
            logger.info(
                f"オープンインタレストデータ取得開始: symbol={symbol}, limit={limit}"
            )

            if db_session is None:
                from database.connection import get_db

                db_session = next(get_db())

            repository = OpenInterestRepository(db_session)
            BybitOpenInterestService()

            start_time = (
                datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                if start_date
                else None
            )
            end_time = (
                datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                if end_date
                else None
            )
            normalized_symbol = (
                symbol
                if ":" in symbol
                else (
                    f"{symbol}:USDT"
                    if symbol.endswith("/USDT")
                    else (
                        f"{symbol}:USD" if symbol.endswith("/USD") else f"{symbol}:USDT"
                    )
                )
            )

            records = repository.get_open_interest_data(
                symbol=normalized_symbol,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )

            data = [
                {
                    "symbol": r.symbol,
                    "open_interest_value": r.open_interest_value,
                    "data_timestamp": r.data_timestamp.isoformat(),
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in records
            ]

            logger.info(f"オープンインタレストデータ取得成功: {len(data)}件")
            return api_response(
                data={
                    "symbol": normalized_symbol,
                    "count": len(data),
                    "open_interest": data,
                },
                message=f"{len(data)}件のオープンインタレストデータを取得しました",
                success=True,
            )
        except Exception as e:
            logger.error(f"オープンインタレストデータ取得エラー: {e}", exc_info=True)
            return error_response(
                message=f"オープンインタレストデータ取得中にエラーが発生しました: {str(e)}",
                details={
                    "symbol": symbol,
                    "count": 0,
                    "open_interest": [],
                    "error": str(e),
                },
            )

    async def collect_bulk_open_interest_data(
        self,
        symbols: List[str],
        db_session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        複数シンボルのオープンインタレストデータを一括収集

        Args:
            symbols: 取引ペアのリスト
            db_session: データベースセッション

        Returns:
            一括収集結果を含む辞書
        """
        try:
            logger.info(f"オープンインタレスト一括収集開始: {len(symbols)}シンボル")

            # データベースセッションの取得
            if db_session is None:
                from database.connection import get_db

                with next(get_db()) as session:
                    db_session = session

            service = BybitOpenInterestService()
            repository = OpenInterestRepository(db_session)

            results = []
            total_saved = 0
            successful_symbols = 0
            failed_symbols = []

            for symbol in symbols:
                try:
                    result = await service.fetch_and_save_open_interest_data(
                        symbol=symbol,
                        repository=repository,
                        fetch_all=True,
                    )
                    results.append(result)
                    total_saved += result["saved_count"]
                    successful_symbols += 1

                    logger.info(f"✅ {symbol}: {result['saved_count']}件保存")

                    # レート制限対策
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"❌ {symbol} 収集エラー: {e}")
                    failed_symbols.append({"symbol": symbol, "error": str(e)})

            logger.info(
                f"オープンインタレスト一括収集完了: {successful_symbols}/{len(symbols)}成功"
            )

            return api_response(
                success=True,
                message=f"オープンインタレスト一括収集完了: {successful_symbols}/{len(symbols)}成功",
                data={
                    "total_saved": total_saved,
                    "successful_symbols": successful_symbols,
                    "failed_symbols": failed_symbols,
                    "results": results,
                    "symbols": symbols,
                },
            )

        except Exception as e:
            logger.error(f"オープンインタレスト一括収集エラー: {e}", exc_info=True)
            return error_response(
                message=f"一括収集中にエラーが発生しました: {str(e)}",
                details={
                    "total_saved": 0,
                    "successful_symbols": 0,
                    "failed_symbols": [],
                    "error": str(e),
                },
            )

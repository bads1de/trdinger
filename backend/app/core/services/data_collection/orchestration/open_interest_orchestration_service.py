"""
オープンインタレストデータ収集統合管理サービス

APIルーター内に散在していたオープンインタレスト関連のビジネスロジックを統合管理します。
責務の分離とSOLID原則に基づいた設計を実現します。
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session

from database.repositories.open_interest_repository import OpenInterestRepository
from ..bybit.open_interest_service import BybitOpenInterestService
from app.core.utils.api_utils import APIResponseHelper

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
        pass

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
                return APIResponseHelper.api_response(
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
                return APIResponseHelper.api_response(
                    success=False,
                    message=f"オープンインタレストデータ収集に失敗しました: {result.get('error', 'Unknown error')}",
                    data={
                        "saved_count": result.get("saved_count", 0),
                        "symbol": symbol,
                        "error": result.get("error"),
                    },
                )

        except Exception as e:
            logger.error(f"オープンインタレストデータ収集エラー: {e}", exc_info=True)
            return APIResponseHelper.api_response(
                success=False,
                message=f"オープンインタレストデータ収集中にエラーが発生しました: {str(e)}",
                data={
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

        Args:
            symbol: 取引ペア
            limit: 取得件数制限（オプション）
            start_date: 開始日（ISO形式文字列）
            end_date: 終了日（ISO形式文字列）
            db_session: データベースセッション

        Returns:
            取得結果を含む辞書
        """
        try:
            logger.info(
                f"オープンインタレストデータ取得開始: symbol={symbol}, limit={limit}"
            )

            # データベースセッションの取得
            if db_session is None:
                from database.connection import get_db

                with next(get_db()) as session:
                    db_session = session

            # サービスとリポジトリの初期化
            service = BybitOpenInterestService()
            repository = OpenInterestRepository(db_session)

            # 日付パラメータの変換
            start_time = None
            end_time = None

            if start_date:
                start_time = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            if end_date:
                end_time = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

            # シンボルの正規化
            normalized_symbol = service.normalize_symbol(symbol)

            # データベースからオープンインタレストデータを取得
            open_interest_records = repository.get_open_interest_data(
                symbol=normalized_symbol,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )

            # データの変換
            open_interest_data = []
            for record in open_interest_records:
                # recordが辞書の場合とオブジェクトの場合の両方に対応
                if isinstance(record, dict):
                    open_interest_data.append(
                        {
                            "symbol": record.get("symbol"),
                            "open_interest": float(record.get("open_interest", 0)),
                            "timestamp": record.get("timestamp"),
                        }
                    )
                else:
                    open_interest_data.append(
                        {
                            "symbol": record.symbol,
                            "open_interest": float(record.open_interest),
                            "timestamp": record.timestamp.isoformat(),
                        }
                    )

            return APIResponseHelper.api_response(
                success=True,
                message=f"オープンインタレストデータを{len(open_interest_data)}件取得しました",
                data={
                    "open_interest_data": open_interest_data,
                    "count": len(open_interest_data),
                    "symbol": normalized_symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": limit,
                },
            )

        except Exception as e:
            logger.error(f"オープンインタレストデータ取得エラー: {e}", exc_info=True)
            return APIResponseHelper.api_response(
                success=False,
                message=f"オープンインタレストデータ取得中にエラーが発生しました: {str(e)}",
                data={
                    "open_interest_data": [],
                    "count": 0,
                    "symbol": symbol,
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

            return APIResponseHelper.api_response(
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
            return APIResponseHelper.api_response(
                success=False,
                message=f"一括収集中にエラーが発生しました: {str(e)}",
                data={
                    "total_saved": 0,
                    "successful_symbols": 0,
                    "failed_symbols": [],
                    "error": str(e),
                },
            )

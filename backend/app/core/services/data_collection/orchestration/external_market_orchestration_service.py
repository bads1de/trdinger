"""
外部市場データ収集統合管理サービス

APIルーター内に散在していた外部市場データ関連のビジネスロジックを統合管理します。
責務の分離とSOLID原則に基づいた設計を実現します。
"""

import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from data_collector.external_market_collector import ExternalMarketDataCollector
from app.core.utils.api_utils import APIResponseHelper

logger = logging.getLogger(__name__)


class ExternalMarketOrchestrationService:
    """
    外部市場データ収集統合管理サービス

    外部市場データの差分収集、履歴データ収集、データ状態取得等の
    統一的な処理を担当します。APIルーターからビジネスロジックを分離し、
    責務を明確化します。
    """

    def __init__(self):
        """初期化"""
        pass

    async def collect_incremental_data(
        self,
        symbols: Optional[List[str]] = None,
        db_session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        外部市場データの差分収集

        Args:
            symbols: 取得するシンボルのリスト（オプション）
            db_session: データベースセッション

        Returns:
            収集結果を含む辞書
        """
        try:
            logger.info(f"外部市場データ差分収集開始: symbols={symbols or 'all'}")

            async with ExternalMarketDataCollector() as collector:
                result = await collector.collect_incremental_external_market_data(
                    symbols=symbols, db_session=db_session
                )

            if result["success"]:
                return APIResponseHelper.api_response(
                    success=True,
                    message=result["message"],
                    data={
                        "fetched_count": result["fetched_count"],
                        "inserted_count": result["inserted_count"],
                        "collection_type": result.get("collection_type", "incremental"),
                        "symbols": symbols,
                    },
                )
            else:
                return APIResponseHelper.api_response(
                    success=False,
                    message=f"差分データ収集に失敗しました: {result.get('error', 'Unknown error')}",
                    data={
                        "fetched_count": result.get("fetched_count", 0),
                        "inserted_count": result.get("inserted_count", 0),
                        "error": result.get("error"),
                    },
                )

        except Exception as e:
            logger.error(f"外部市場データ差分収集エラー: {e}", exc_info=True)
            return APIResponseHelper.api_response(
                success=False,
                message=f"差分データ収集中にエラーが発生しました: {str(e)}",
                data={
                    "fetched_count": 0,
                    "inserted_count": 0,
                    "error": str(e),
                },
            )

    async def get_data_status(
        self, db_session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        外部市場データの状態取得

        Args:
            db_session: データベースセッション

        Returns:
            データ状態情報を含む辞書
        """
        try:
            logger.info("外部市場データ状態取得開始")

            async with ExternalMarketDataCollector() as collector:
                status = await collector.get_external_market_data_status(
                    db_session=db_session
                )

            if status["success"]:
                return APIResponseHelper.api_response(
                    success=True,
                    message="外部市場データの状態を取得しました",
                    data={
                        "total_records": status.get("total_records", 0),
                        "symbols": status.get("symbols", []),
                        "latest_timestamp": status.get("latest_timestamp"),
                        "oldest_timestamp": status.get("oldest_timestamp"),
                        "status_details": status,
                    },
                )
            else:
                return APIResponseHelper.api_response(
                    success=False,
                    message=f"データ状態取得に失敗しました: {status.get('error', 'Unknown error')}",
                    data={"error": status.get("error")},
                )

        except Exception as e:
            logger.error(f"外部市場データ状態取得エラー: {e}", exc_info=True)
            return APIResponseHelper.api_response(
                success=False,
                message=f"データ状態取得中にエラーが発生しました: {str(e)}",
                data={"error": str(e)},
            )

    async def collect_historical_data(
        self,
        symbols: Optional[List[str]] = None,
        period: str = "1mo",
        db_session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        外部市場データの履歴収集

        Args:
            symbols: 取得するシンボルのリスト（オプション）
            period: 取得期間（デフォルト: 1mo）
            db_session: データベースセッション

        Returns:
            収集結果を含む辞書
        """
        try:
            logger.info(
                f"外部市場データ履歴収集開始: symbols={symbols or 'all'}, period={period}"
            )

            async with ExternalMarketDataCollector() as collector:
                result = await collector.collect_external_market_data(
                    symbols=symbols, period=period, db_session=db_session
                )

            if result["success"]:
                return APIResponseHelper.api_response(
                    success=True,
                    message=result["message"],
                    data={
                        "fetched_count": result["fetched_count"],
                        "inserted_count": result["inserted_count"],
                        "period": period,
                        "symbols": symbols,
                    },
                )
            else:
                return APIResponseHelper.api_response(
                    success=False,
                    message=f"履歴データ収集に失敗しました: {result.get('error', 'Unknown error')}",
                    data={
                        "fetched_count": result.get("fetched_count", 0),
                        "inserted_count": result.get("inserted_count", 0),
                        "error": result.get("error"),
                    },
                )

        except Exception as e:
            logger.error(f"外部市場データ履歴収集エラー: {e}", exc_info=True)
            return APIResponseHelper.api_response(
                success=False,
                message=f"履歴データ収集中にエラーが発生しました: {str(e)}",
                data={
                    "fetched_count": 0,
                    "inserted_count": 0,
                    "error": str(e),
                },
            )

    async def get_available_symbols(self) -> Dict[str, Any]:
        """
        利用可能なシンボル一覧の取得

        Returns:
            利用可能なシンボル情報を含む辞書
        """
        try:
            logger.info("利用可能なシンボル一覧取得開始")

            async with ExternalMarketDataCollector() as collector:
                # ExternalMarketServiceから利用可能なシンボルを取得
                if (
                    hasattr(collector, "external_market_service")
                    and collector.external_market_service
                ):
                    symbols = collector.external_market_service.get_available_symbols()
                else:
                    # フォールバック: デフォルトシンボル
                    symbols = {
                        "^GSPC": "S&P 500",
                        "^IXIC": "NASDAQ Composite",
                        "DX-Y.NYB": "US Dollar Index",
                        "^VIX": "CBOE Volatility Index",
                    }

            return APIResponseHelper.api_response(
                success=True,
                message="利用可能なシンボル一覧を取得しました",
                data={
                    "symbols": symbols,
                    "count": len(symbols),
                },
            )

        except Exception as e:
            logger.error(f"利用可能なシンボル一覧取得エラー: {e}", exc_info=True)
            return APIResponseHelper.api_response(
                success=False,
                message=f"シンボル一覧取得中にエラーが発生しました: {str(e)}",
                data={"error": str(e)},
            )

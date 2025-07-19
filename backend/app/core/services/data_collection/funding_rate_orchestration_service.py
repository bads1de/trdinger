"""
ファンディングレートデータ収集統合管理サービス

APIルーター内に散在していたファンディングレート関連のビジネスロジックを統合管理します。
責務の分離とSOLID原則に基づいた設計を実現します。
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from database.repositories.funding_rate_repository import FundingRateRepository
from app.core.services.data_collection.funding_rate_service import (
    BybitFundingRateService,
)
from app.core.utils.api_utils import APIResponseHelper

logger = logging.getLogger(__name__)


class FundingRateOrchestrationService:
    """
    ファンディングレートデータ収集統合管理サービス

    ファンディングレートデータの収集、取得、管理等の
    統一的な処理を担当します。APIルーターからビジネスロジックを分離し、
    責務を明確化します。
    """

    def __init__(self):
        """初期化"""
        pass

    async def collect_funding_rate_data(
        self,
        symbol: str,
        limit: Optional[int] = None,
        fetch_all: bool = False,
        db_session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        ファンディングレートデータの収集

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
                f"ファンディングレートデータ収集開始: symbol={symbol}, fetch_all={fetch_all}"
            )

            # サービスとリポジトリの初期化
            service = BybitFundingRateService()
            repository = FundingRateRepository(db_session)

            # データ収集実行
            result = await service.fetch_and_save_funding_rate_data(
                symbol=symbol,
                limit=limit,
                repository=repository,
                fetch_all=fetch_all,
            )

            if result.get("success", False):
                return APIResponseHelper.api_response(
                    success=True,
                    message=f"{result['saved_count']}件のファンディングレートデータを保存しました",
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
                    message=f"ファンディングレートデータ収集に失敗しました: {result.get('error', 'Unknown error')}",
                    data={
                        "saved_count": result.get("saved_count", 0),
                        "symbol": symbol,
                        "error": result.get("error"),
                    },
                )

        except Exception as e:
            logger.error(f"ファンディングレートデータ収集エラー: {e}", exc_info=True)
            return APIResponseHelper.api_response(
                success=False,
                message=f"ファンディングレートデータ収集中にエラーが発生しました: {str(e)}",
                data={
                    "saved_count": 0,
                    "symbol": symbol,
                    "error": str(e),
                },
            )

    async def get_funding_rate_data(
        self,
        symbol: str,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        db_session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        ファンディングレートデータの取得

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
                f"ファンディングレートデータ取得開始: symbol={symbol}, limit={limit}"
            )

            # サービスとリポジトリの初期化
            service = BybitFundingRateService()
            repository = FundingRateRepository(db_session)

            # 日付パラメータの変換
            start_time = None
            end_time = None

            if start_date:
                start_time = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            if end_date:
                end_time = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

            # シンボルの正規化
            normalized_symbol = service.normalize_symbol(symbol)

            # データベースからファンディングレートデータを取得
            funding_rate_records = repository.get_funding_rate_data(
                symbol=normalized_symbol,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )

            # データの変換
            funding_rates = []
            for record in funding_rate_records:
                # recordが辞書の場合とオブジェクトの場合の両方に対応
                if isinstance(record, dict):
                    funding_rates.append(
                        {
                            "symbol": record.get("symbol"),
                            "funding_rate": float(record.get("funding_rate", 0)),
                            "funding_time": record.get("funding_time"),
                            "mark_price": (
                                float(record.get("mark_price", 0))
                                if record.get("mark_price")
                                else None
                            ),
                            "index_price": (
                                float(record.get("index_price", 0))
                                if record.get("index_price")
                                else None
                            ),
                        }
                    )
                else:
                    funding_rates.append(
                        {
                            "symbol": record.symbol,
                            "funding_rate": float(record.funding_rate),
                            "funding_time": record.funding_time.isoformat(),
                            "mark_price": (
                                float(record.mark_price) if record.mark_price else None
                            ),
                            "index_price": (
                                float(record.index_price)
                                if record.index_price
                                else None
                            ),
                        }
                    )

            return APIResponseHelper.api_response(
                success=True,
                message=f"ファンディングレートデータを{len(funding_rates)}件取得しました",
                data={
                    "funding_rates": funding_rates,
                    "count": len(funding_rates),
                    "symbol": normalized_symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": limit,
                },
            )

        except Exception as e:
            logger.error(f"ファンディングレートデータ取得エラー: {e}", exc_info=True)
            return APIResponseHelper.api_response(
                success=False,
                message=f"ファンディングレートデータ取得中にエラーが発生しました: {str(e)}",
                data={
                    "funding_rates": [],
                    "count": 0,
                    "symbol": symbol,
                    "error": str(e),
                },
            )

    async def get_funding_rate_status(
        self, db_session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        ファンディングレートデータの状態取得

        Args:
            db_session: データベースセッション

        Returns:
            データ状態情報を含む辞書
        """
        try:
            logger.info("ファンディングレートデータ状態取得開始")

            repository = FundingRateRepository(db_session)

            # データ統計の取得
            total_count = repository.get_total_count()
            symbols = repository.get_available_symbols()
            latest_data = repository.get_latest_data()
            oldest_data = repository.get_oldest_data()

            return APIResponseHelper.api_response(
                success=True,
                message="ファンディングレートデータの状態を取得しました",
                data={
                    "total_records": total_count,
                    "symbols": symbols,
                    "latest_timestamp": (
                        latest_data.funding_time.isoformat() if latest_data else None
                    ),
                    "oldest_timestamp": (
                        oldest_data.funding_time.isoformat() if oldest_data else None
                    ),
                    "symbol_count": len(symbols),
                },
            )

        except Exception as e:
            logger.error(
                f"ファンディングレートデータ状態取得エラー: {e}", exc_info=True
            )
            return APIResponseHelper.api_response(
                success=False,
                message=f"データ状態取得中にエラーが発生しました: {str(e)}",
                data={"error": str(e)},
            )

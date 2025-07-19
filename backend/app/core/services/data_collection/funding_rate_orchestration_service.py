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
from database.connection import get_db

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

            # データベースセッションの取得
            if db_session is None:
                with next(get_db()) as session:
                    return await self._collect_funding_rate_data_with_session(
                        symbol, limit, fetch_all, session
                    )
            else:
                return await self._collect_funding_rate_data_with_session(
                    symbol, limit, fetch_all, db_session
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

    async def _collect_funding_rate_data_with_session(
        self,
        symbol: str,
        limit: Optional[int] = None,
        fetch_all: bool = False,
        db_session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        実際のファンディングレートデータ収集処理（セッション指定版）

        Args:
            symbol: 取引ペア
            limit: 取得件数制限（オプション）
            fetch_all: 全データ取得フラグ
            db_session: データベースセッション（必須）

        Returns:
            収集結果を含む辞書
        """
        if db_session is None:
            raise ValueError("db_session is required")

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

            # データベースセッションの取得
            if db_session is None:
                with next(get_db()) as session:
                    return await self._get_funding_rate_data_with_session(
                        symbol, limit, start_date, end_date, session
                    )
            else:
                return await self._get_funding_rate_data_with_session(
                    symbol, limit, start_date, end_date, db_session
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

    async def _get_funding_rate_data_with_session(
        self,
        symbol: str,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        db_session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        実際のファンディングレートデータ取得処理（セッション指定版）

        Args:
            symbol: 取引ペア
            limit: 取得件数制限（オプション）
            start_date: 開始日（ISO形式文字列）
            end_date: 終了日（ISO形式文字列）
            db_session: データベースセッション（必須）

        Returns:
            取得結果を含む辞書
        """
        if db_session is None:
            raise ValueError("db_session is required")

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

        # データの変換 - 辞書形式で扱うことで型エラーを回避
        funding_rates = []
        for record in funding_rate_records:
            # モデルインスタンスから辞書形式でデータを取得
            record_dict = {
                "symbol": getattr(record, "symbol", normalized_symbol),
                "funding_rate": getattr(record, "funding_rate", 0.0),
                "funding_timestamp": getattr(record, "funding_timestamp", None),
                "mark_price": getattr(record, "mark_price", None),
                "index_price": getattr(record, "index_price", None),
            }

            funding_rates.append(
                {
                    "symbol": (
                        str(record_dict["symbol"])
                        if record_dict["symbol"]
                        else normalized_symbol
                    ),
                    "funding_rate": (
                        float(record_dict["funding_rate"])
                        if record_dict["funding_rate"] is not None
                        else 0.0
                    ),
                    "funding_time": (
                        record_dict["funding_timestamp"].isoformat()
                        if record_dict["funding_timestamp"]
                        else None
                    ),
                    "mark_price": (
                        float(record_dict["mark_price"])
                        if record_dict["mark_price"] is not None
                        else None
                    ),
                    "index_price": (
                        float(record_dict["index_price"])
                        if record_dict["index_price"] is not None
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

            # データベースセッションの取得
            if db_session is None:
                with next(get_db()) as session:
                    return await self._get_funding_rate_status_with_session(session)
            else:
                return await self._get_funding_rate_status_with_session(db_session)

        except Exception as e:
            logger.error(
                f"ファンディングレートデータ状態取得エラー: {e}", exc_info=True
            )
            return APIResponseHelper.api_response(
                success=False,
                message=f"データ状態取得中にエラーが発生しました: {str(e)}",
                data={"error": str(e)},
            )

    async def _get_funding_rate_status_with_session(
        self, db_session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        実際のファンディングレートデータ状態取得処理（セッション指定版）

        Args:
            db_session: データベースセッション（必須）

        Returns:
            データ状態情報を含む辞書
        """
        if db_session is None:
            raise ValueError("db_session is required")

        repository = FundingRateRepository(db_session)

        # 利用可能なシンボルの取得
        from sqlalchemy import distinct

        symbols_query = db_session.query(distinct(repository.model_class.symbol))
        symbols = [str(row[0]) for row in symbols_query.all() if row[0] is not None]

        # 総レコード数の取得
        total_count = db_session.query(repository.model_class).count()

        # 最新データの取得
        latest_record = (
            db_session.query(repository.model_class)
            .order_by(repository.model_class.funding_timestamp.desc())
            .first()
        )

        # 最古データの取得
        oldest_record = (
            db_session.query(repository.model_class)
            .order_by(repository.model_class.funding_timestamp.asc())
            .first()
        )

        # 辞書形式でデータを取得することで型エラーを回避
        latest_data = None
        if latest_record:
            latest_data = {
                "funding_timestamp": getattr(latest_record, "funding_timestamp", None)
            }

        oldest_data = None
        if oldest_record:
            oldest_data = {
                "funding_timestamp": getattr(oldest_record, "funding_timestamp", None)
            }

        return APIResponseHelper.api_response(
            success=True,
            message="ファンディングレートデータの状態を取得しました",
            data={
                "total_records": int(total_count),
                "symbols": symbols,
                "latest_timestamp": (
                    latest_data["funding_timestamp"].isoformat()
                    if latest_data and latest_data["funding_timestamp"]
                    else None
                ),
                "oldest_timestamp": (
                    oldest_data["funding_timestamp"].isoformat()
                    if oldest_data and oldest_data["funding_timestamp"]
                    else None
                ),
                "symbol_count": len(symbols),
            },
        )

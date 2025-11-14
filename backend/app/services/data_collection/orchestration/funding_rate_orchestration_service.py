"""
ファンディングレートデータ収集オーケストレーションサービス

このモジュールは、異なる取引所からのファンディングレートデータの収集、
処理、永続化を調整する役割を担います。
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import Depends
from sqlalchemy.orm import Session

from app.services.data_collection.bybit.funding_rate_service import (
    BybitFundingRateService,
)
from database.models import FundingRateData
from database.repositories.funding_rate_repository import FundingRateRepository

logger = logging.getLogger(__name__)


class FundingRateOrchestrationService:
    """
    ファンディングレートデータの収集と管理を統括するサービスクラス
    """

    def __init__(
        self,
        bybit_service: BybitFundingRateService = Depends(BybitFundingRateService),
    ):
        """
        サービスの初期化

        Args:
            bybit_service: Bybitファンディングレートサービス
        """
        self.bybit_service = bybit_service

    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """
        文字列の日付をdatetimeオブジェクトに変換する

        Args:
            date_str: 日付文字列（ISO形式、例: "2023-01-01T00:00:00"）

        Returns:
            datetimeオブジェクト、またはNone
        """
        if not date_str:
            return None
        try:
            # ISO形式の日付文字列をパース（例: "2023-01-01T00:00:00"）
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError as e:
            logger.error(f"日付文字列のパースに失敗しました: {date_str}, エラー: {e}")
            return None

    async def get_funding_rate_data(
        self,
        symbol: str,
        limit: int,
        start_date: Optional[str],
        end_date: Optional[str],
        db_session: Session,
    ) -> List[FundingRateData]:
        """
        指定された条件でファンディングレートデータを取得する

        Args:
            symbol: 通貨ペアシンボル
            limit: 取得件数
            start_date: 開始日
            end_date: 終了日
            db_session: データベースセッション

        Returns:
            ファンディングレートデータのリスト
        """
        logger.info(f"{symbol}のファンディングレートデータを{limit}件取得します")

        # 文字列の日付をdatetimeオブジェクトに変換
        start_datetime = self._parse_datetime(start_date)
        end_datetime = self._parse_datetime(end_date)

        funding_rate_repo = FundingRateRepository(db_session)
        return funding_rate_repo.get_funding_rate_data(
            symbol=symbol,
            limit=limit,
            start_time=start_datetime,
            end_time=end_datetime,
        )

    async def collect_funding_rate_data(
        self,
        symbol: str,
        limit: int,
        fetch_all: bool,
        db_session: Session,
    ) -> dict:
        """
        単一シンボルのファンディングレートデータを収集・保存する

        Args:
            symbol: 通貨ペアシンボル
            limit: 取得件数
            fetch_all: 全期間取得フラグ
            db_session: データベースセッション

        Returns:
            収集結果
        """
        logger.info(f"{symbol}のファンディングレートデータ収集を開始します")
        if fetch_all:
            rates_data = await self.bybit_service.fetch_all_funding_rate_history(symbol)
        else:
            rates_data = await self.bybit_service.fetch_funding_rate_history(
                symbol, limit
            )

        if not rates_data:
            logger.warning(
                f"{symbol}のファンディングレートデータが見つかりませんでした"
            )
            return {"status": "success", "message": "データなし", "count": 0}

        funding_rate_repo = FundingRateRepository(db_session)
        inserted_count = funding_rate_repo.insert_funding_rate_data(rates_data)
        logger.info(
            f"{symbol}のファンディングレートデータ{inserted_count}件を保存しました"
        )
        return {
            "status": "success",
            "message": "データ収集完了",
            "count": inserted_count,
        }

    async def collect_bulk_funding_rate_data(
        self, symbols: List[str], db_session: Session
    ) -> dict:
        """
        複数シンボルのファンディングレートデータを一括で収集・保存する

        Args:
            symbols: 通貨ペアシンボルのリスト
            db_session: データベースセッション

        Returns:
            収集結果
        """
        logger.info(f"{len(symbols)}シンボルの一括データ収集を開始します")
        total_count = 0
        for symbol in symbols:
            try:
                result = await self.collect_funding_rate_data(
                    symbol=symbol, limit=200, fetch_all=True, db_session=db_session
                )
                total_count += result.get("count", 0)
            except Exception as e:
                logger.error(f"{symbol}のデータ収集中にエラーが発生しました: {e}")
        logger.info(f"一括データ収集完了。合計{total_count}件のデータを保存しました")
        return {
            "status": "success",
            "message": "一括データ収集完了",
            "total_count": total_count,
        }

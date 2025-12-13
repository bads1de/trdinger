"""
ロング/ショート比率データ収集オーケストレーションサービス

このモジュールは、Bybitからのロング/ショート比率データの収集、
処理、永続化を調整する役割を担います。
"""

import logging
from typing import List, Optional

from fastapi import Depends
from sqlalchemy.orm import Session

from app.services.data_collection.bybit.long_short_ratio_service import (
    BybitLongShortRatioService,
)
from app.services.data_collection.orchestration.base_orchestration_service import (
    BaseDataCollectionOrchestrationService,
)
from database.models import LongShortRatioData
from database.repositories.long_short_ratio_repository import LongShortRatioRepository

logger = logging.getLogger(__name__)


class LongShortRatioOrchestrationService(BaseDataCollectionOrchestrationService):
    """
    ロング/ショート比率データの収集と管理を統括するサービスクラス
    """

    def __init__(
        self,
        bybit_service: BybitLongShortRatioService = Depends(BybitLongShortRatioService),
    ):
        """
        サービスの初期化

        Args:
            bybit_service: Bybitロング/ショート比率サービス
        """
        self.bybit_service = bybit_service

    async def get_long_short_ratio_data(
        self,
        symbol: str,
        period: str,
        limit: int,
        start_date: Optional[str],
        end_date: Optional[str],
        db_session: Session,
    ) -> List[LongShortRatioData]:
        """
        指定された条件でロング/ショート比率データを取得する

        Args:
            symbol: 通貨ペアシンボル
            period: 期間（例: 5min, 1h, 1d）
            limit: 取得件数
            start_date: 開始日
            end_date: 終了日
            db_session: データベースセッション

        Returns:
            ロング/ショート比率データのリスト
        """
        logger.info(f"{symbol} ({period}) のLS比率データを{limit}件取得します")

        # 文字列の日付をdatetimeオブジェクトに変換
        start_datetime = self._parse_datetime(start_date)
        end_datetime = self._parse_datetime(end_date)

        repo = LongShortRatioRepository(db_session)
        return repo.get_long_short_ratio_data(
            symbol=symbol,
            period=period,
            limit=limit,
            start_time=start_datetime,
            end_time=end_datetime,
        )

    async def collect_long_short_ratio_data(
        self,
        symbol: str,
        period: str,
        fetch_all: bool,
        db_session: Session,
    ) -> dict:
        """
        単一シンボルのロング/ショート比率データを収集・保存する

        Args:
            symbol: 通貨ペアシンボル
            period: 期間
            fetch_all: 全期間取得フラグ (Trueなら履歴収集、Falseなら差分更新)
            db_session: データベースセッション

        Returns:
            収集結果
        """
        logger.info(
            f"{symbol} ({period}) のLS比率データ収集を開始します (fetch_all={fetch_all})"
        )
        repo = LongShortRatioRepository(db_session)

        try:
            if fetch_all:
                # 履歴データの一括収集
                saved_count = await self.bybit_service.collect_historical_long_short_ratio_data(
                    symbol=symbol,
                    period=period,
                    repository=repo,
                )
            else:
                # 差分更新
                result = await self.bybit_service.fetch_incremental_long_short_ratio_data(
                    symbol=symbol,
                    period=period,
                    repository=repo,
                )
                saved_count = result.get("saved_count", 0)

            logger.info(f"{symbol} ({period}) のLS比率データ収集完了: {saved_count}件保存")
            
            return self._create_success_response(
                "データ収集完了", data={"count": saved_count}
            )

        except Exception as e:
            logger.error(f"{symbol} ({period}) のLS比率データ収集中にエラーが発生しました: {e}")
            return self._create_error_response(
                f"データ収集中にエラーが発生しました: {str(e)}"
            )

    async def collect_bulk_long_short_ratio_data(
        self,
        symbols: List[str],
        period: str,
        fetch_all: bool,
        db_session: Session,
    ) -> dict:
        """
        複数シンボルのロング/ショート比率データを一括で収集・保存する

        Args:
            symbols: 通貨ペアシンボルのリスト
            period: 期間
            fetch_all: 全期間取得フラグ
            db_session: データベースセッション

        Returns:
            収集結果
        """
        logger.info(f"{len(symbols)}シンボル ({period}) の一括データ収集を開始します")
        total_count = 0
        
        for symbol in symbols:
            try:
                result = await self.collect_long_short_ratio_data(
                    symbol=symbol,
                    period=period,
                    fetch_all=fetch_all,
                    db_session=db_session,
                )
                
                if result.get("success") and result.get("data"):
                    total_count += result["data"].get("count", 0)
                    
            except Exception as e:
                logger.error(f"{symbol}のデータ収集中にエラーが発生しました: {e}")
        
        logger.info(f"一括データ収集完了。合計{total_count}件のデータを保存しました")

        return self._create_success_response(
            "一括データ収集完了", data={"total_count": total_count}
        )

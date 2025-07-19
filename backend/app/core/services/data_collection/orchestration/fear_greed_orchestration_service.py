"""
Fear & Greed Index データ収集統合管理サービス

APIルーター内に散在していたFear & Greed Index関連のビジネスロジックを統合管理します。
責務の分離とSOLID原則に基づいた設計を実現します。
"""

import logging
from typing import Dict, Any
from sqlalchemy.orm import Session

from data_collector.external_market_collector import ExternalMarketDataCollector
from app.core.utils.api_utils import APIResponseHelper

logger = logging.getLogger(__name__)


class FearGreedOrchestrationService:
    """
    Fear & Greed Index データ収集統合管理サービス

    Fear & Greed Index データの収集、履歴データ収集、
    差分更新、クリーンアップ等の統一的な処理を担当します。
    """

    def __init__(self):
        """初期化"""
        pass

    async def collect_fear_greed_data(self, limit: int, db: Session) -> Dict[str, Any]:
        """
        Fear & Greed Index データを収集

        Args:
            limit: 取得するデータ数
            db: データベースセッション

        Returns:
            収集結果
        """
        try:
            async with ExternalMarketDataCollector() as collector:
                result = await collector.collect_fear_greed_data(
                    limit=limit, db_session=db
                )

            if result["success"]:
                return APIResponseHelper.api_response(
                    success=True,
                    message=result["message"],
                    data=result,
                )
            else:
                raise Exception(
                    f"データ収集に失敗しました: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(f"Fear & Greed Index データ収集エラー: {e}")
            raise

    async def collect_incremental_fear_greed_data(self, db: Session) -> Dict[str, Any]:
        """
        Fear & Greed Index の差分データを収集

        Args:
            db: データベースセッション

        Returns:
            差分収集結果
        """
        try:
            async with ExternalMarketDataCollector() as collector:
                result = await collector.collect_incremental_fear_greed_data(
                    db_session=db
                )

            if result["success"]:
                return APIResponseHelper.api_response(
                    success=True,
                    message=result["message"],
                    data=result,
                )
            else:
                raise Exception(
                    f"差分データ収集に失敗しました: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(f"Fear & Greed Index 差分データ収集エラー: {e}")
            raise

    async def collect_historical_fear_greed_data(
        self, limit: int, db: Session
    ) -> Dict[str, Any]:
        """
        Fear & Greed Index の履歴データを収集

        Args:
            limit: 取得するデータ数の上限
            db: データベースセッション

        Returns:
            履歴収集結果
        """
        try:
            async with ExternalMarketDataCollector() as collector:
                result = await collector.collect_historical_fear_greed_data(
                    limit=limit, db_session=db
                )

            if result["success"]:
                return APIResponseHelper.api_response(
                    success=True,
                    message=result["message"],
                    data=result,
                )
            else:
                raise Exception(
                    f"履歴データ収集に失敗しました: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(f"Fear & Greed Index 履歴データ収集エラー: {e}")
            raise

    async def cleanup_old_fear_greed_data(
        self, days_to_keep: int, db: Session
    ) -> Dict[str, Any]:
        """
        古いFear & Greed Index データをクリーンアップ

        Args:
            days_to_keep: 保持する日数
            db: データベースセッション

        Returns:
            クリーンアップ結果
        """
        try:
            async with ExternalMarketDataCollector() as collector:
                result = await collector.cleanup_old_data(
                    days_to_keep=days_to_keep, db_session=db
                )

            if result["success"]:
                return APIResponseHelper.api_response(
                    success=True,
                    message=result["message"],
                    data=result,
                )
            else:
                raise Exception(
                    f"データクリーンアップに失敗しました: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(f"Fear & Greed Index データクリーンアップエラー: {e}")
            raise

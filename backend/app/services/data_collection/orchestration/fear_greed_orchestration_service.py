"""
Fear & Greed Index データ収集統合管理サービス

APIルーター内に散在していたFear & Greed Index関連のビジネスロジックを統合管理します。
責務の分離とSOLID原則に基づいた設計を実現します。
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy.orm import Session

from app.services.data_collection.fear_greed.fear_greed_service import (
    FearGreedIndexService,
)
from app.utils.response import api_response
from database.repositories.fear_greed_repository import FearGreedIndexRepository

logger = logging.getLogger(__name__)


class FearGreedOrchestrationService:
    """
    Fear & Greed Index データ収集統合管理サービス

    Fear & Greed Index データの収集、履歴データ収集、
    差分更新、クリーンアップ等の統一的な処理を担当します。
    """

    def __init__(self):
        """初期化"""

    async def get_fear_greed_data_status(self, db: Session) -> Dict[str, Any]:
        """
        Fear & Greed Index データの状態を取得します。

        Args:
            db: データベースセッション

        Returns:
            データ状態情報
        """
        try:
            repository = FearGreedIndexRepository(db)
            data_range = repository.get_data_range()
            latest_timestamp = repository.get_latest_data_timestamp()

            status_data = {
                "success": True,
                "data_range": data_range,
                "latest_timestamp": (
                    latest_timestamp.isoformat() if latest_timestamp else None
                ),
                "current_time": datetime.now(timezone.utc).isoformat(),
            }
            return api_response(
                success=True,
                message="Fear & Greed Index データ状態を取得しました",
                data=status_data,
            )
        except Exception as e:
            logger.error(f"Fear & Greed Index データ状態取得エラー: {e}")
            raise

    async def collect_fear_greed_data(self, limit: int, db: Session) -> Dict[str, Any]:
        """
        Fear & Greed Index データを収集
        """
        try:
            async with FearGreedIndexService() as service:
                repository = FearGreedIndexRepository(db)
                result = await service.fetch_and_save_fear_greed_data(
                    limit=limit, repository=repository
                )

            if result["success"]:
                return api_response(
                    success=True,
                    message=result.get(
                        "message",
                        f"{result.get('inserted_count', 0)}件のデータを保存しました。",
                    ),
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
        """
        try:
            repository = FearGreedIndexRepository(db)
            latest_timestamp = repository.get_latest_data_timestamp()
            limit = 7 if latest_timestamp else 365

            async with FearGreedIndexService() as service:
                result = await service.fetch_and_save_fear_greed_data(
                    limit=limit, repository=repository
                )

            result["collection_type"] = "incremental"
            result["latest_timestamp_before"] = (
                latest_timestamp.isoformat() if latest_timestamp else None
            )

            if result["success"]:
                return api_response(
                    success=True,
                    message=result.get(
                        "message",
                        f"{result.get('inserted_count', 0)}件のデータを保存しました。",
                    ),
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
        """
        try:
            repository = FearGreedIndexRepository(db)
            data_range = repository.get_data_range()

            async with FearGreedIndexService() as service:
                result = await service.fetch_and_save_fear_greed_data(
                    limit=limit, repository=repository
                )

            result["collection_type"] = "historical"
            result["existing_data_range"] = data_range

            if result["success"]:
                return api_response(
                    success=True,
                    message=result.get(
                        "message",
                        f"{result.get('inserted_count', 0)}件のデータを保存しました。",
                    ),
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
        """
        try:
            repository = FearGreedIndexRepository(db)
            from datetime import timedelta

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            deleted_count = repository.delete_old_data(
                timestamp_column="data_timestamp",
                before_date=cutoff_date,
                additional_filters=None,
            )

            result = {
                "success": True,
                "deleted_count": deleted_count,
                "cutoff_date": cutoff_date.isoformat(),
                "days_kept": days_to_keep,
            }

            return api_response(
                success=True,
                message=f"{deleted_count}件の古いデータを削除しました。",
                data=result,
            )
        except Exception as e:
            logger.error(f"Fear & Greed Index データクリーンアップエラー: {e}")
            raise

    async def get_fear_greed_data(
        self,
        db: Session,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = 30,
    ) -> Dict[str, Any]:
        repository = FearGreedIndexRepository(db)

        start_time = None
        end_time = None

        if start_date:
            try:
                start_time = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            except ValueError:
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=400, detail=f"無効な開始日時形式: {start_date}"
                )

        if end_date:
            try:
                end_time = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            except ValueError:
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=400, detail=f"無効な終了日時形式: {end_date}"
                )

        data = repository.get_fear_greed_data(
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

        result_data = [item.to_dict() for item in data]

        return api_response(
            success=True,
            message=f"Fear & Greed Index データを {len(result_data)} 件取得しました",
            data={
                "data": result_data,
                "metadata": {
                    "count": len(result_data),
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": limit,
                },
            },
        )

    async def get_latest_fear_greed_data(
        self, db: Session, limit: int = 30
    ) -> Dict[str, Any]:
        repository = FearGreedIndexRepository(db)
        data = repository.get_latest_fear_greed_data(limit=limit)

        result_data = [item.to_dict() for item in data]

        return api_response(
            success=True,
            message=f"最新のFear & Greed Index データを {len(result_data)} 件取得しました",
            data={
                "data": result_data,
                "metadata": {
                    "count": len(result_data),
                    "limit": limit,
                },
            },
        )

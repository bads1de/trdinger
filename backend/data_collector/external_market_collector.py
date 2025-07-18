"""
外部市場データ収集器

Fear & Greed Index などの外部データソースからデータを収集し、
データベースに保存する機能を提供します。
"""

import logging
import asyncio
from typing import Dict, Optional
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from database.connection import SessionLocal
from database.repositories.fear_greed_repository import FearGreedIndexRepository
from database.repositories.external_market_repository import ExternalMarketRepository
from app.core.services.data_collection.fear_greed_service import FearGreedIndexService
from app.core.services.data_collection.external_market_service import (
    ExternalMarketService,
)

logger = logging.getLogger(__name__)


class ExternalMarketDataCollector:
    """外部市場データ収集器"""

    def __init__(self):
        """収集器を初期化"""
        self.fear_greed_service = None
        self.external_market_service = None

    async def __aenter__(self):
        """非同期コンテキストマネージャーの開始"""
        self.fear_greed_service = FearGreedIndexService()
        await self.fear_greed_service.__aenter__()

        self.external_market_service = ExternalMarketService()
        await self.external_market_service.__aenter__()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャーの終了"""
        if self.fear_greed_service:
            await self.fear_greed_service.__aexit__(exc_type, exc_val, exc_tb)

        if self.external_market_service:
            await self.external_market_service.__aexit__(exc_type, exc_val, exc_tb)

    async def collect_fear_greed_data(
        self,
        limit: int = 30,
        db_session: Optional[Session] = None,
    ) -> Dict:
        """
        Fear & Greed Index データを収集

        Args:
            limit: 取得するデータ数
            db_session: データベースセッション（オプション）

        Returns:
            収集結果を含む辞書
        """
        session = db_session or SessionLocal()

        try:
            repository = FearGreedIndexRepository(session)

            if not self.fear_greed_service:
                self.fear_greed_service = FearGreedIndexService()
                await self.fear_greed_service.__aenter__()

            result = await self.fear_greed_service.fetch_and_save_fear_greed_data(
                limit=limit, repository=repository
            )

            return result

        except Exception as e:
            logger.error(f"Fear & Greed Index データ収集エラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "fetched_count": 0,
                "inserted_count": 0,
            }
        finally:
            if not db_session:  # 外部から提供されていない場合のみクローズ
                session.close()

    async def collect_incremental_fear_greed_data(
        self,
        db_session: Optional[Session] = None,
    ) -> Dict:
        """
        Fear & Greed Index の差分データを収集

        Args:
            db_session: データベースセッション（オプション）

        Returns:
            収集結果を含む辞書
        """
        session = db_session or SessionLocal()

        try:
            repository = FearGreedIndexRepository(session)

            # 最新データの確認
            latest_timestamp = repository.get_latest_data_timestamp()

            if latest_timestamp:
                logger.info(f"最新のFear & Greed Indexデータ: {latest_timestamp}")
                # 通常は1日1回更新なので、少数のデータを取得
                limit = 7  # 1週間分
            else:
                logger.info(
                    "Fear & Greed Indexデータが存在しません。初回収集を実行します。"
                )
                # 初回収集時はより多くのデータを取得
                limit = 365  # 1年分

            result = await self.collect_fear_greed_data(limit=limit, db_session=session)

            # 結果にメタデータを追加
            result["collection_type"] = "incremental"
            result["latest_timestamp_before"] = (
                latest_timestamp.isoformat() if latest_timestamp else None
            )

            return result

        except Exception as e:
            logger.error(f"Fear & Greed Index 差分データ収集エラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "collection_type": "incremental",
                "fetched_count": 0,
                "inserted_count": 0,
            }
        finally:
            if not db_session:
                session.close()

    async def collect_historical_fear_greed_data(
        self,
        limit: int = 1000,
        db_session: Optional[Session] = None,
    ) -> Dict:
        """
        Fear & Greed Index の履歴データを収集

        Args:
            limit: 取得するデータ数（最大値）
            db_session: データベースセッション（オプション）

        Returns:
            収集結果を含む辞書
        """
        session = db_session or SessionLocal()

        try:
            repository = FearGreedIndexRepository(session)

            # 既存データの確認
            data_range = repository.get_data_range()
            logger.info(f"既存のFear & Greed Indexデータ範囲: {data_range}")

            result = await self.collect_fear_greed_data(limit=limit, db_session=session)

            # 結果にメタデータを追加
            result["collection_type"] = "historical"
            result["existing_data_range"] = data_range

            return result

        except Exception as e:
            logger.error(f"Fear & Greed Index 履歴データ収集エラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "collection_type": "historical",
                "fetched_count": 0,
                "inserted_count": 0,
            }
        finally:
            if not db_session:
                session.close()

    async def get_data_status(
        self,
        db_session: Optional[Session] = None,
    ) -> Dict:
        """
        データの状態を取得

        Args:
            db_session: データベースセッション（オプション）

        Returns:
            データ状態情報
        """
        session = db_session or SessionLocal()

        try:
            repository = FearGreedIndexRepository(session)

            data_range = repository.get_data_range()
            latest_timestamp = repository.get_latest_data_timestamp()

            return {
                "success": True,
                "data_range": data_range,
                "latest_timestamp": (
                    latest_timestamp.isoformat() if latest_timestamp else None
                ),
                "current_time": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Fear & Greed Index データ状態取得エラー: {e}")
            return {
                "success": False,
                "error": str(e),
            }
        finally:
            if not db_session:
                session.close()

    async def collect_external_market_data(
        self,
        symbols: Optional[list] = None,
        period: str = "1mo",
        db_session: Optional[Session] = None,
    ) -> Dict:
        """
        外部市場データを収集

        Args:
            symbols: 取得するシンボルのリスト（デフォルト: 全シンボル）
            period: 取得期間
            db_session: データベースセッション（オプション）

        Returns:
            収集結果を含む辞書
        """
        session = db_session or SessionLocal()

        try:
            repository = ExternalMarketRepository(session)

            if not self.external_market_service:
                self.external_market_service = ExternalMarketService()
                await self.external_market_service.__aenter__()

            result = (
                await self.external_market_service.fetch_and_save_external_market_data(
                    symbols=symbols, period=period, repository=repository
                )
            )

            return result

        except Exception as e:
            logger.error(f"外部市場データ収集エラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "fetched_count": 0,
                "inserted_count": 0,
            }
        finally:
            if not db_session:
                session.close()

    async def collect_incremental_external_market_data(
        self,
        symbols: Optional[list] = None,
        db_session: Optional[Session] = None,
    ) -> Dict:
        """
        外部市場データの差分データを収集

        Args:
            symbols: 取得するシンボルのリスト
            db_session: データベースセッション（オプション）

        Returns:
            収集結果を含む辞書
        """
        session = db_session or SessionLocal()

        try:
            repository = ExternalMarketRepository(session)

            # 最新データの確認
            latest_timestamp = repository.get_latest_data_timestamp()

            if latest_timestamp:
                logger.info(f"最新の外部市場データ: {latest_timestamp}")

                # 最新データの翌日から今日までを取得
                from datetime import datetime, timedelta

                start_date = (latest_timestamp + timedelta(days=1)).strftime("%Y-%m-%d")
                end_date = datetime.now().strftime("%Y-%m-%d")

                logger.info(f"差分収集期間: {start_date} ～ {end_date}")

                # 日付指定で差分データを取得
                if not self.external_market_service:
                    self.external_market_service = ExternalMarketService()
                    await self.external_market_service.__aenter__()

                result = await self.external_market_service.fetch_and_save_external_market_data(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    repository=repository,
                )
            else:
                logger.info("外部市場データが存在しません。初回収集を実行します。")
                # 初回収集時は2020年からのデータを取得
                from datetime import datetime

                start_date = "2020-01-01"
                end_date = datetime.now().strftime("%Y-%m-%d")

                if not self.external_market_service:
                    self.external_market_service = ExternalMarketService()
                    await self.external_market_service.__aenter__()

                result = await self.external_market_service.fetch_and_save_external_market_data(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    repository=repository,
                )

            # 結果にメタデータを追加
            result["collection_type"] = "incremental"
            result["latest_timestamp_before"] = (
                latest_timestamp.isoformat() if latest_timestamp else None
            )

            return result

        except Exception as e:
            logger.error(f"外部市場データ差分収集エラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "collection_type": "incremental",
                "fetched_count": 0,
                "inserted_count": 0,
            }
        finally:
            if not db_session:
                session.close()

    async def collect_historical_external_market_data(
        self,
        symbols: Optional[list] = None,
        period: str = "5y",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        db_session: Optional[Session] = None,
    ) -> Dict:
        """
        外部市場データの履歴データを収集

        Args:
            symbols: 取得するシンボルのリスト
            period: 取得期間
            start_date: 開始日（YYYY-MM-DD形式、periodより優先）
            end_date: 終了日（YYYY-MM-DD形式、periodより優先）
            db_session: データベースセッション（オプション）

        Returns:
            収集結果を含む辞書
        """
        session = db_session or SessionLocal()

        try:
            repository = ExternalMarketRepository(session)

            # 既存データの確認
            data_range = repository.get_data_range()
            logger.info(f"既存の外部市場データ範囲: {data_range}")

            # 日付指定がある場合は、external_market_serviceに直接渡す
            if start_date or end_date:
                if not self.external_market_service:
                    self.external_market_service = ExternalMarketService()
                    await self.external_market_service.__aenter__()

                result = await self.external_market_service.fetch_and_save_external_market_data(
                    symbols=symbols,
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                    repository=repository,
                )
            else:
                result = await self.collect_external_market_data(
                    symbols=symbols, period=period, db_session=session
                )

            # 結果にメタデータを追加
            result["collection_type"] = "historical"
            result["existing_data_range"] = data_range
            if start_date:
                result["start_date"] = start_date
            if end_date:
                result["end_date"] = end_date

            return result

        except Exception as e:
            logger.error(f"外部市場データ履歴収集エラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "collection_type": "historical",
                "fetched_count": 0,
                "inserted_count": 0,
            }
        finally:
            if not db_session:
                session.close()

    async def get_external_market_data_status(
        self,
        db_session: Optional[Session] = None,
    ) -> Dict:
        """
        外部市場データの状態を取得

        Args:
            db_session: データベースセッション（オプション）

        Returns:
            データ状態情報
        """
        session = db_session or SessionLocal()

        try:
            repository = ExternalMarketRepository(session)

            statistics = repository.get_data_statistics()
            latest_timestamp = repository.get_latest_data_timestamp()

            return {
                "success": True,
                "statistics": statistics,
                "latest_timestamp": (
                    latest_timestamp.isoformat() if latest_timestamp else None
                ),
                "current_time": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"外部市場データ状態取得エラー: {e}")
            return {
                "success": False,
                "error": str(e),
            }
        finally:
            if not db_session:
                session.close()

    async def cleanup_old_data(
        self,
        days_to_keep: int = 365,
        db_session: Optional[Session] = None,
    ) -> Dict:
        """
        古いデータをクリーンアップ

        Args:
            days_to_keep: 保持する日数
            db_session: データベースセッション（オプション）

        Returns:
            クリーンアップ結果
        """
        session = db_session or SessionLocal()

        try:
            repository = FearGreedIndexRepository(session)

            from datetime import timedelta

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

            deleted_count = repository.delete_old_data(cutoff_date)

            return {
                "success": True,
                "deleted_count": deleted_count,
                "cutoff_date": cutoff_date.isoformat(),
                "days_kept": days_to_keep,
            }

        except Exception as e:
            logger.error(f"Fear & Greed Index データクリーンアップエラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "deleted_count": 0,
            }
        finally:
            if not db_session:
                session.close()


# スタンドアロン実行用の関数
async def main():
    """メイン関数（テスト用）"""
    logging.basicConfig(level=logging.INFO)

    async with ExternalMarketDataCollector() as collector:
        # Fear & Greed Index データ状態の確認
        fear_greed_status = await collector.get_data_status()
        print(f"Fear & Greed Index データ状態: {fear_greed_status}")

        # 外部市場データ状態の確認
        external_market_status = await collector.get_external_market_data_status()
        print(f"外部市場データ状態: {external_market_status}")

        # Fear & Greed Index 差分データ収集
        fear_greed_result = await collector.collect_incremental_fear_greed_data()
        print(f"Fear & Greed Index 収集結果: {fear_greed_result}")

        # 外部市場データ差分収集
        external_market_result = (
            await collector.collect_incremental_external_market_data()
        )
        print(f"外部市場データ収集結果: {external_market_result}")


if __name__ == "__main__":
    asyncio.run(main())

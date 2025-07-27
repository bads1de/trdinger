"""
Fear & Greed Index サービス統合テスト
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.connection import Base
from database.repositories.fear_greed_repository import FearGreedIndexRepository
from app.core.services.data_collection.fear_greed.fear_greed_service import (
    FearGreedIndexService,
)
from app.core.services.data_collection.orchestration.fear_greed_orchestration_service import (
    FearGreedOrchestrationService,
)


class TestFearGreedServiceIntegration:
    """Fear & Greed Index サービス統合テストクラス"""

    @pytest.fixture(scope="function")
    def db_session(self):
        """テスト用データベースセッション"""
        # インメモリSQLiteデータベースを使用
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)

        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()

        yield session

        session.close()

    @pytest.fixture
    def fear_greed_repository(self, db_session):
        """FearGreedIndexRepositoryのインスタンス"""
        return FearGreedIndexRepository(db_session)

    @pytest.fixture
    def mock_api_response(self):
        """モックAPIレスポンス"""
        return {
            "name": "Fear and Greed Index",
            "data": [
                {
                    "value": "25",
                    "value_classification": "Fear",
                    "timestamp": str(
                        int(
                            (datetime.now(timezone.utc) - timedelta(days=1)).timestamp()
                        )
                    ),
                    "time_until_update": "86400",
                },
                {
                    "value": "75",
                    "value_classification": "Greed",
                    "timestamp": str(int(datetime.now(timezone.utc).timestamp())),
                    "time_until_update": "86400",
                },
            ],
            "metadata": {"error": None},
        }

    @pytest.mark.asyncio
    async def test_fear_greed_service_data_conversion(self):
        """Fear & Greed Serviceのデータ変換テスト"""
        service = FearGreedIndexService()

        # APIレスポンスのサンプル
        api_data = [
            {
                "value": "25",
                "value_classification": "Fear",
                "timestamp": "1640995200",  # 2022-01-01 00:00:00 UTC
                "time_until_update": "86400",
            },
            {
                "value": "75",
                "value_classification": "Greed",
                "timestamp": "1640908800",  # 2021-12-31 00:00:00 UTC
                "time_until_update": "86400",
            },
        ]

        converted_data = service._convert_api_data_to_db_format(api_data)

        assert len(converted_data) == 2

        # 最初のレコードを確認
        first_record = converted_data[0]
        assert first_record["value"] == 25
        assert first_record["value_classification"] == "Fear"
        assert isinstance(first_record["data_timestamp"], datetime)
        assert isinstance(first_record["timestamp"], datetime)
        assert first_record["data_timestamp"].tzinfo == timezone.utc
        assert first_record["timestamp"].tzinfo == timezone.utc

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_fear_greed_service_fetch_data(self, mock_get, mock_api_response):
        """Fear & Greed Serviceのデータ取得テスト"""
        # モックレスポンスの設定
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_api_response
        mock_get.return_value.__aenter__.return_value = mock_response

        async with FearGreedIndexService() as service:
            data = await service.fetch_fear_greed_data(limit=30)

            assert len(data) == 2
            assert data[0]["value"] == 25
            assert data[0]["value_classification"] == "Fear"
            assert data[1]["value"] == 75
            assert data[1]["value_classification"] == "Greed"

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_fear_greed_service_fetch_and_save(
        self, mock_get, mock_api_response, fear_greed_repository
    ):
        """Fear & Greed Serviceのデータ取得・保存テスト"""
        # モックレスポンスの設定
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_api_response
        mock_get.return_value.__aenter__.return_value = mock_response

        async with FearGreedIndexService() as service:
            result = await service.fetch_and_save_fear_greed_data(
                limit=30, repository=fear_greed_repository
            )

            assert result["success"] is True
            assert result["fetched_count"] == 2
            assert result["inserted_count"] == 2

            # データベースに保存されたことを確認
            saved_data = fear_greed_repository.get_fear_greed_data()
            assert len(saved_data) == 2

    @pytest.mark.asyncio
    async def test_orchestration_service_collect_data(self, db_session):
        """オーケストレーションサービスのデータ収集テスト"""
        orchestration_service = FearGreedOrchestrationService()

        # FearGreedIndexServiceをモック
        with patch(
            "app.core.services.data_collection.orchestration.fear_greed_orchestration_service.FearGreedIndexService"
        ) as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value.__aenter__.return_value = mock_service

            mock_service.fetch_and_save_fear_greed_data.return_value = {
                "success": True,
                "fetched_count": 2,
                "inserted_count": 2,
                "message": "Fear & Greed Indexデータを 2 件保存しました",
            }

            result = await orchestration_service.collect_fear_greed_data(
                limit=30, db=db_session
            )

            assert result["success"] is True
            assert result["data"]["inserted_count"] == 2

    @pytest.mark.asyncio
    async def test_orchestration_service_incremental_collect(self, db_session):
        """オーケストレーションサービスの差分収集テスト"""
        orchestration_service = FearGreedOrchestrationService()

        # 既存データを挿入
        repository = FearGreedIndexRepository(db_session)
        base_time = datetime.now(timezone.utc)
        existing_data = [
            {
                "value": 50,
                "value_classification": "Neutral",
                "data_timestamp": base_time - timedelta(days=2),
                "timestamp": base_time - timedelta(hours=1),
            }
        ]
        repository.insert_fear_greed_data(existing_data)

        # FearGreedIndexServiceをモック
        with patch(
            "app.core.services.data_collection.orchestration.fear_greed_orchestration_service.FearGreedIndexService"
        ) as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value.__aenter__.return_value = mock_service

            mock_service.fetch_and_save_fear_greed_data.return_value = {
                "success": True,
                "fetched_count": 1,
                "inserted_count": 1,
                "message": "Fear & Greed Indexデータを 1 件保存しました",
            }

            result = await orchestration_service.collect_incremental_fear_greed_data(
                db=db_session
            )

            assert result["success"] is True
            assert result["data"]["collection_type"] == "incremental"
            assert result["data"]["latest_timestamp_before"] is not None

    @pytest.mark.asyncio
    async def test_orchestration_service_get_status(self, db_session):
        """オーケストレーションサービスのステータス取得テスト"""
        orchestration_service = FearGreedOrchestrationService()

        # テストデータを挿入
        repository = FearGreedIndexRepository(db_session)
        base_time = datetime.now(timezone.utc)
        test_data = [
            {
                "value": 30,
                "value_classification": "Fear",
                "data_timestamp": base_time - timedelta(days=1),
                "timestamp": base_time,
            },
            {
                "value": 70,
                "value_classification": "Greed",
                "data_timestamp": base_time,
                "timestamp": base_time,
            },
        ]
        repository.insert_fear_greed_data(test_data)

        result = await orchestration_service.get_fear_greed_data_status(db=db_session)

        assert result["success"] is True
        assert result["data"]["data_range"]["total_count"] == 2
        assert result["data"]["latest_timestamp"] is not None

    @pytest.mark.asyncio
    async def test_orchestration_service_cleanup(self, db_session):
        """オーケストレーションサービスのクリーンアップテスト"""
        orchestration_service = FearGreedOrchestrationService()

        # 古いテストデータを挿入
        repository = FearGreedIndexRepository(db_session)
        base_time = datetime.now(timezone.utc)
        old_data = [
            {
                "value": 20,
                "value_classification": "Extreme Fear",
                "data_timestamp": base_time - timedelta(days=400),  # 400日前
                "timestamp": base_time - timedelta(days=400),
            },
            {
                "value": 80,
                "value_classification": "Extreme Greed",
                "data_timestamp": base_time - timedelta(days=1),  # 1日前
                "timestamp": base_time,
            },
        ]
        repository.insert_fear_greed_data(old_data)

        result = await orchestration_service.cleanup_old_fear_greed_data(
            days_to_keep=365, db=db_session
        )

        assert result["success"] is True
        assert result["data"]["deleted_count"] == 1  # 400日前のデータが削除される

        # 残りのデータを確認
        remaining_data = repository.get_fear_greed_data()
        assert len(remaining_data) == 1
        assert remaining_data[0].value == 80

    @pytest.mark.asyncio
    async def test_data_update_scenario(self, db_session):
        """データ更新シナリオの統合テスト"""
        repository = FearGreedIndexRepository(db_session)

        # 1. 初期データの挿入
        base_time = datetime.now(timezone.utc)
        initial_data = [
            {
                "value": 40,
                "value_classification": "Fear",
                "data_timestamp": base_time - timedelta(days=2),
                "timestamp": base_time - timedelta(hours=2),
            },
            {
                "value": 60,
                "value_classification": "Greed",
                "data_timestamp": base_time - timedelta(days=1),
                "timestamp": base_time - timedelta(hours=1),
            },
        ]
        repository.insert_fear_greed_data(initial_data)

        # 初期状態の確認
        initial_count = repository.get_data_count()
        assert initial_count == 2

        initial_latest = repository.get_latest_data_timestamp()
        assert initial_latest == (base_time - timedelta(days=1))

        # 2. 新しいデータの追加（差分更新をシミュレート）
        new_data = [
            {
                "value": 80,
                "value_classification": "Extreme Greed",
                "data_timestamp": base_time,
                "timestamp": base_time,
            }
        ]
        inserted_count = repository.insert_fear_greed_data(new_data)
        assert inserted_count == 1

        # 更新後の状態確認
        updated_count = repository.get_data_count()
        assert updated_count == 3

        updated_latest = repository.get_latest_data_timestamp()
        assert updated_latest == base_time

        # 3. 重複データの挿入テスト（同じdata_timestampのデータ）
        duplicate_data = [
            {
                "value": 85,  # 異なる値
                "value_classification": "Extreme Greed",
                "data_timestamp": base_time,  # 同じタイムスタンプ
                "timestamp": base_time + timedelta(minutes=30),
            }
        ]
        duplicate_inserted = repository.insert_fear_greed_data(duplicate_data)
        assert duplicate_inserted == 0  # 重複により挿入されない

        # データ数が変わらないことを確認
        final_count = repository.get_data_count()
        assert final_count == 3

        # 4. データ範囲の確認
        data_range = repository.get_data_range()
        assert data_range["total_count"] == 3
        assert data_range["oldest_data"] is not None
        assert data_range["newest_data"] is not None

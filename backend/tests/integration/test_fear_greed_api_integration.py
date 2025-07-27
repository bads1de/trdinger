"""
Fear & Greed Index API統合テスト
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.connection import Base, get_db
from app.main import app


class TestFearGreedAPIIntegration:
    """Fear & Greed Index API統合テストクラス"""

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
    def client(self, db_session):
        """テスト用FastAPIクライアント"""

        def override_get_db():
            try:
                yield db_session
            finally:
                pass

        app.dependency_overrides[get_db] = override_get_db
        client = TestClient(app)
        yield client
        app.dependency_overrides.clear()

    @pytest.fixture
    def mock_fear_greed_service_data(self):
        """モックFear & Greed Serviceデータ"""
        base_time = datetime.now(timezone.utc)
        return [
            {
                "value": 25,
                "value_classification": "Fear",
                "data_timestamp": base_time - timedelta(days=1),
                "timestamp": base_time,
            },
            {
                "value": 75,
                "value_classification": "Greed",
                "data_timestamp": base_time,
                "timestamp": base_time,
            },
        ]

    def test_get_fear_greed_status_empty_db(self, client):
        """空のデータベースでのステータス取得テスト"""
        response = client.get("/api/fear-greed/status")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["data_range"]["total_count"] == 0
        assert data["data"]["latest_timestamp"] is None

    @patch(
        "app.core.services.data_collection.fear_greed.fear_greed_service.FearGreedIndexService.fetch_and_save_fear_greed_data"
    )
    def test_collect_fear_greed_data(
        self, mock_fetch_and_save, client, mock_fear_greed_service_data
    ):
        """Fear & Greed Indexデータ収集テスト"""
        # モックの設定
        mock_fetch_and_save.return_value = {
            "success": True,
            "fetched_count": 2,
            "inserted_count": 2,
            "message": "Fear & Greed Indexデータを 2 件保存しました",
        }

        response = client.post("/api/fear-greed/collect?limit=30")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["inserted_count"] == 2
        assert "保存しました" in data["message"]

    @patch(
        "app.core.services.data_collection.fear_greed.fear_greed_service.FearGreedIndexService.fetch_and_save_fear_greed_data"
    )
    def test_collect_incremental_fear_greed_data(self, mock_fetch_and_save, client):
        """Fear & Greed Index差分データ収集テスト"""
        # モックの設定
        mock_fetch_and_save.return_value = {
            "success": True,
            "fetched_count": 1,
            "inserted_count": 1,
            "message": "Fear & Greed Indexデータを 1 件保存しました",
        }

        response = client.post("/api/fear-greed/collect-incremental")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["collection_type"] == "incremental"

    def test_get_fear_greed_data_empty(self, client):
        """空のデータベースでのデータ取得テスト"""
        response = client.get("/api/fear-greed/data")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["data"] == []
        assert data["data"]["metadata"]["count"] == 0

    def test_get_latest_fear_greed_data_empty(self, client):
        """空のデータベースでの最新データ取得テスト"""
        response = client.get("/api/fear-greed/latest")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["data"] == []
        assert data["data"]["metadata"]["count"] == 0

    def test_cleanup_old_fear_greed_data(self, client):
        """古いデータクリーンアップテスト"""
        response = client.delete("/api/fear-greed/cleanup?days_to_keep=365")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["deleted_count"] == 0  # 空のDBなので0件削除

    def test_get_fear_greed_data_with_date_range(self, client):
        """日付範囲指定でのデータ取得テスト"""
        start_date = "2025-01-01T00:00:00Z"
        end_date = "2025-01-31T23:59:59Z"

        response = client.get(
            f"/api/fear-greed/data?start_date={start_date}&end_date={end_date}&limit=10"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["metadata"]["start_date"] == start_date
        assert data["data"]["metadata"]["end_date"] == end_date
        assert data["data"]["metadata"]["limit"] == 10

    def test_get_fear_greed_data_invalid_date_format(self, client):
        """無効な日付形式でのエラーテスト"""
        response = client.get("/api/fear-greed/data?start_date=invalid-date")

        assert response.status_code == 400
        data = response.json()
        assert "無効な開始日時形式" in data["detail"]

    @patch(
        "app.core.services.data_collection.fear_greed.fear_greed_service.FearGreedIndexService.fetch_and_save_fear_greed_data"
    )
    def test_collect_fear_greed_data_failure(self, mock_fetch_and_save, client):
        """データ収集失敗時のテスト"""
        # モックの設定（失敗）
        mock_fetch_and_save.return_value = {
            "success": False,
            "fetched_count": 0,
            "inserted_count": 0,
            "error": "API connection failed",
            "message": "データ収集に失敗しました",
        }

        response = client.post("/api/fear-greed/collect?limit=30")

        # UnifiedErrorHandlerによりエラーが適切に処理される
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False

    @patch(
        "app.core.services.data_collection.fear_greed.fear_greed_service.FearGreedIndexService"
    )
    def test_end_to_end_data_flow(self, mock_service_class, client, db_session):
        """エンドツーエンドのデータフロー統合テスト"""
        # モックサービスインスタンスの設定
        mock_service = AsyncMock()
        mock_service_class.return_value.__aenter__.return_value = mock_service

        # APIレスポンスのシミュレーション
        api_response_data = [
            {
                "value": "30",
                "value_classification": "Fear",
                "timestamp": str(
                    int((datetime.now(timezone.utc) - timedelta(days=1)).timestamp())
                ),
            },
            {
                "value": "70",
                "value_classification": "Greed",
                "timestamp": str(int(datetime.now(timezone.utc).timestamp())),
            },
        ]

        # サービスメソッドのモック設定
        mock_service.fetch_fear_greed_data.return_value = api_response_data
        mock_service._convert_api_data_to_db_format.return_value = [
            {
                "value": 30,
                "value_classification": "Fear",
                "data_timestamp": datetime.now(timezone.utc) - timedelta(days=1),
                "timestamp": datetime.now(timezone.utc),
            },
            {
                "value": 70,
                "value_classification": "Greed",
                "data_timestamp": datetime.now(timezone.utc),
                "timestamp": datetime.now(timezone.utc),
            },
        ]

        # fetch_and_save_fear_greed_dataの実際の実装をシミュレート
        async def mock_fetch_and_save(limit, repository):
            converted_data = mock_service._convert_api_data_to_db_format.return_value
            inserted_count = repository.insert_fear_greed_data(converted_data)
            return {
                "success": True,
                "fetched_count": len(converted_data),
                "inserted_count": inserted_count,
                "message": f"Fear & Greed Indexデータを {inserted_count} 件保存しました",
            }

        mock_service.fetch_and_save_fear_greed_data = mock_fetch_and_save

        # 1. データ収集
        response = client.post("/api/fear-greed/collect?limit=30")
        assert response.status_code == 200
        collect_data = response.json()
        assert collect_data["success"] is True
        assert collect_data["data"]["inserted_count"] == 2

        # 2. ステータス確認
        response = client.get("/api/fear-greed/status")
        assert response.status_code == 200
        status_data = response.json()
        assert status_data["success"] is True
        assert status_data["data"]["data_range"]["total_count"] == 2

        # 3. データ取得
        response = client.get("/api/fear-greed/data")
        assert response.status_code == 200
        data_response = response.json()
        assert data_response["success"] is True
        assert len(data_response["data"]["data"]) == 2

        # 4. 最新データ取得
        response = client.get("/api/fear-greed/latest?limit=1")
        assert response.status_code == 200
        latest_data = response.json()
        assert latest_data["success"] is True
        assert len(latest_data["data"]["data"]) == 1
        assert latest_data["data"]["data"][0]["value"] == 70  # 最新のデータ

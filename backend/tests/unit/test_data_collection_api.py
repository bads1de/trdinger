"""
データ収集APIのテスト

修正されたdata_collection.pyのエンドポイントをテストします。
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from app.main import app
from database.repository import OHLCVRepository


class TestDataCollectionAPI:
    """データ収集APIのテストクラス"""

    @pytest.fixture
    def client(self):
        """テストクライアント"""
        return TestClient(app)

    @pytest.fixture
    def mock_db(self):
        """モックデータベースセッション"""
        return Mock(spec=Session)

    @pytest.fixture
    def mock_repository(self):
        """モックOHLCVRepository"""
        return Mock(spec=OHLCVRepository)

    def test_get_collection_status_with_normalization(
        self, client, mock_db, mock_repository
    ):
        """
        シンボル正規化が正しく動作することをテスト
        """
        # Given: データが存在する場合
        mock_repository.get_data_count.return_value = 100
        mock_repository.get_latest_timestamp.return_value = None
        mock_repository.get_oldest_timestamp.return_value = None

        with (
            patch("app.api.data_collection.get_db", return_value=mock_db),
            patch(
                "app.api.data_collection.OHLCVRepository",
                return_value=mock_repository,
            ),
        ):

            # When: SOL/USDTのステータスを確認
            response = client.get("/api/data-collection/status/SOL/USDT/1d")

            # Then: 正常にレスポンスが返される
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["symbol"] == "SOL/USDT"  # 正規化されたシンボル
            assert data["original_symbol"] == "SOL/USDT"
            assert data["timeframe"] == "1d"
            assert data["data_count"] == 100
            assert data["status"] == "data_exists"

    def test_get_collection_status_no_data_with_suggestion(
        self, client, mock_db, mock_repository
    ):
        """
        データが存在しない場合の提案機能をテスト
        """
        # Given: データが存在しない場合
        mock_repository.get_data_count.return_value = 0

        with (
            patch("app.api.data_collection.get_db", return_value=mock_db),
            patch(
                "app.api.data_collection.OHLCVRepository",
                return_value=mock_repository,
            ),
        ):

            # When: LTC/USDTのステータスを確認（auto_fetch=false）
            response = client.get("/api/data-collection/status/LTC/USDT/1h")

            # Then: データなしの提案レスポンスが返される
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["symbol"] == "LTC/USDT"
            assert data["original_symbol"] == "LTC/USDT"
            assert data["data_count"] == 0
            assert data["status"] == "no_data"
            assert "suggestion" in data
            assert "manual_fetch" in data["suggestion"]
            assert "auto_fetch" in data["suggestion"]

    def test_get_collection_status_auto_fetch(self, client, mock_db, mock_repository):
        """
        自動フェッチ機能をテスト
        """
        # Given: データが存在しない場合
        mock_repository.get_data_count.return_value = 0

        with (
            patch("app.api.data_collection.get_db", return_value=mock_db),
            patch(
                "app.api.data_collection.OHLCVRepository",
                return_value=mock_repository,
            ),
            patch(
                "app.api.data_collection._collect_historical_background"
            ) as mock_background,
        ):

            # When: UNI/USDTのステータスを確認（auto_fetch=true）
            response = client.get(
                "/api/data-collection/status/UNI/USDT/1d?auto_fetch=true"
            )

            # Then: 自動フェッチ開始レスポンスが返される
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["symbol"] == "UNI/USDT"
            assert data["status"] == "auto_fetch_started"
            assert "自動収集を開始しました" in data["message"]

    def test_get_collection_status_invalid_symbol(self, client):
        """
        無効なシンボルのテスト
        """
        # When: サポートされていないシンボルでリクエスト
        response = client.get("/api/data-collection/status/INVALID/SYMBOL/1d")

        # Then: 400エラーが返される
        assert response.status_code == 400
        data = response.json()
        assert "サポートされていないシンボル" in data["detail"]

    def test_get_collection_status_invalid_timeframe(self, client):
        """
        無効な時間軸のテスト
        """
        # When: サポートされていない時間軸でリクエスト
        response = client.get("/api/data-collection/status/BTC/USDT/invalid")

        # Then: 400エラーが返される
        assert response.status_code == 400
        data = response.json()
        assert "無効な時間軸" in data["detail"]

    def test_get_supported_symbols(self, client):
        """
        サポートされているシンボル一覧の取得をテスト
        """
        # When: サポートシンボル一覧を取得
        response = client.get("/api/data-collection/supported-symbols")

        # Then: 正常にレスポンスが返される
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "symbols" in data
        assert "timeframes" in data
        assert "BTC/USDT" in data["symbols"]
        assert "SOL/USDT" in data["symbols"]
        assert "LTC/USDT" in data["symbols"]
        assert "UNI/USDT" in data["symbols"]
        assert "ETH/USDT" in data["symbols"]
        assert "1d" in data["timeframes"]

    def test_ethusd_symbol_handling(self, client, mock_db, mock_repository):
        """
        ETHUSD（先物）シンボルの処理をテスト
        """
        # Given: ETHUSDデータが存在する場合
        mock_repository.get_data_count.return_value = 50
        mock_repository.get_latest_timestamp.return_value = None
        mock_repository.get_oldest_timestamp.return_value = None

        with (
            patch("app.api.data_collection.get_db", return_value=mock_db),
            patch(
                "app.api.data_collection.OHLCVRepository",
                return_value=mock_repository,
            ),
        ):

            # When: ETHUSDのステータスを確認
            response = client.get("/api/data-collection/status/ETHUSD/1d")

            # Then: 正常にレスポンスが返される
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["symbol"] == "ETHUSD"  # 正規化されたシンボル
            assert data["original_symbol"] == "ETHUSD"
            assert data["data_count"] == 50
            assert data["status"] == "data_exists"

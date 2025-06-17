"""
差分更新機能のテスト

このテストファイルは、差分更新APIエンドポイントとHistoricalDataServiceの
collect_incremental_dataメソッドの動作を検証します。
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from app.main import app
from app.core.services.historical_data_service import HistoricalDataService
from database.repositories.ohlcv_repository import OHLCVRepository


class TestIncrementalUpdateAPI:
    """差分更新APIのテストクラス"""

    def setup_method(self):
        """テストメソッドの前処理"""
        self.client = TestClient(app)
        self.test_symbol = "BTC/USDT:USDT"
        self.test_timeframe = "1h"

    @patch("app.api.data_collection.get_db")
    @patch("app.api.data_collection.HistoricalDataService")
    @patch("app.api.data_collection.OHLCVRepository")
    def test_incremental_update_success(
        self, mock_repo_class, mock_service_class, mock_get_db
    ):
        """差分更新が成功する場合のテスト"""
        # モックの設定
        mock_db = Mock(spec=Session)
        mock_get_db.return_value = mock_db

        mock_repo = Mock(spec=OHLCVRepository)
        mock_repo_class.return_value = mock_repo

        mock_service = Mock(spec=HistoricalDataService)
        mock_service.collect_incremental_data = AsyncMock(
            return_value={
                "success": True,
                "symbol": self.test_symbol,
                "timeframe": self.test_timeframe,
                "saved_count": 5,
            }
        )
        mock_service_class.return_value = mock_service

        # APIエンドポイントを呼び出し
        response = self.client.post(
            f"/api/data-collection/update?symbol={self.test_symbol}&timeframe={self.test_timeframe}"
        )

        # レスポンスの検証
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["symbol"] == self.test_symbol
        assert data["timeframe"] == self.test_timeframe
        assert data["saved_count"] == 5

        # モックが正しく呼び出されたことを確認
        mock_service.collect_incremental_data.assert_called_once_with(
            self.test_symbol, self.test_timeframe, mock_repo
        )

    @patch("app.api.data_collection.get_db")
    @patch("app.api.data_collection.HistoricalDataService")
    @patch("app.api.data_collection.OHLCVRepository")
    def test_incremental_update_no_new_data(
        self, mock_repo_class, mock_service_class, mock_get_db
    ):
        """新しいデータがない場合のテスト"""
        # モックの設定
        mock_db = Mock(spec=Session)
        mock_get_db.return_value = mock_db

        mock_repo = Mock(spec=OHLCVRepository)
        mock_repo_class.return_value = mock_repo

        mock_service = Mock(spec=HistoricalDataService)
        mock_service.collect_incremental_data = AsyncMock(
            return_value={
                "success": True,
                "message": "新しいデータはありません",
                "saved_count": 0,
            }
        )
        mock_service_class.return_value = mock_service

        # APIエンドポイントを呼び出し
        response = self.client.post(
            f"/api/data-collection/update?symbol={self.test_symbol}&timeframe={self.test_timeframe}"
        )

        # レスポンスの検証
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "新しいデータはありません"
        assert data["saved_count"] == 0

    @patch("app.api.data_collection.get_db")
    @patch("app.api.data_collection.HistoricalDataService")
    @patch("app.api.data_collection.OHLCVRepository")
    def test_incremental_update_service_error(
        self, mock_repo_class, mock_service_class, mock_get_db
    ):
        """サービスでエラーが発生する場合のテスト"""
        # モックの設定
        mock_db = Mock(spec=Session)
        mock_get_db.return_value = mock_db

        mock_repo = Mock(spec=OHLCVRepository)
        mock_repo_class.return_value = mock_repo

        mock_service = Mock(spec=HistoricalDataService)
        mock_service.collect_incremental_data = AsyncMock(
            side_effect=Exception("テストエラー")
        )
        mock_service_class.return_value = mock_service

        # APIエンドポイントを呼び出し
        response = self.client.post(
            f"/api/data-collection/update?symbol={self.test_symbol}&timeframe={self.test_timeframe}"
        )

        # レスポンスの検証
        assert response.status_code == 500
        data = response.json()
        assert "テストエラー" in data["detail"]

    def test_incremental_update_default_parameters(self):
        """デフォルトパラメータでの呼び出しテスト"""
        with (
            patch("app.api.data_collection.get_db") as mock_get_db,
            patch(
                "app.api.data_collection.HistoricalDataService"
            ) as mock_service_class,
            patch("app.api.data_collection.OHLCVRepository") as mock_repo_class,
        ):

            # モックの設定
            mock_db = Mock(spec=Session)
            mock_get_db.return_value = mock_db

            mock_repo = Mock(spec=OHLCVRepository)
            mock_repo_class.return_value = mock_repo

            mock_service = Mock(spec=HistoricalDataService)
            mock_service.collect_incremental_data = AsyncMock(
                return_value={
                    "success": True,
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "saved_count": 3,
                }
            )
            mock_service_class.return_value = mock_service

            # パラメータなしでAPIエンドポイントを呼び出し
            response = self.client.post("/api/data-collection/update")

            # レスポンスの検証
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

            # デフォルトパラメータで呼び出されたことを確認
            mock_service.collect_incremental_data.assert_called_once_with(
                "BTC/USDT", "1h", mock_repo
            )


class TestHistoricalDataServiceIncrementalUpdate:
    """HistoricalDataServiceの差分更新メソッドのテストクラス"""

    def setup_method(self):
        """テストメソッドの前処理"""
        self.service = HistoricalDataService()
        self.test_symbol = "BTC/USDT:USDT"
        self.test_timeframe = "1h"

    @patch("app.core.services.historical_data_service.OHLCVRepository")
    async def test_collect_incremental_data_no_repository(self, mock_repo_class):
        """リポジトリが提供されない場合のテスト"""
        result = await self.service.collect_incremental_data(
            self.test_symbol, self.test_timeframe, None
        )

        assert result["success"] is False
        assert result["message"] == "リポジトリが必要です"

    @patch("app.core.services.historical_data_service.logger")
    async def test_collect_incremental_data_with_existing_data(self, mock_logger):
        """既存データがある場合の差分更新テスト"""
        # モックリポジトリの設定
        mock_repo = Mock(spec=OHLCVRepository)
        latest_timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_repo.get_latest_timestamp.return_value = latest_timestamp

        # モックマーケットサービスの設定
        with patch.object(self.service, "market_service") as mock_market_service:
            mock_ohlcv_data = [
                [1704110400000, 45000, 45100, 44900, 45050, 100],  # 新しいデータ
                [1704114000000, 45050, 45200, 45000, 45150, 120],
            ]
            mock_market_service.fetch_ohlcv_data = AsyncMock(
                return_value=mock_ohlcv_data
            )
            mock_market_service._save_ohlcv_to_database = AsyncMock(return_value=2)

            # 差分更新を実行
            result = await self.service.collect_incremental_data(
                self.test_symbol, self.test_timeframe, mock_repo
            )

            # 結果の検証
            assert result["success"] is True
            assert result["symbol"] == self.test_symbol
            assert result["timeframe"] == self.test_timeframe
            assert result["saved_count"] == 2

            # 正しいタイムスタンプで呼び出されたことを確認
            expected_since_ms = int(latest_timestamp.timestamp() * 1000)
            mock_market_service.fetch_ohlcv_data.assert_called_once_with(
                self.test_symbol, self.test_timeframe, 1000, since=expected_since_ms
            )

    @patch("app.core.services.historical_data_service.logger")
    async def test_collect_incremental_data_no_existing_data(self, mock_logger):
        """既存データがない場合の初回データ収集テスト"""
        # モックリポジトリの設定
        mock_repo = Mock(spec=OHLCVRepository)
        mock_repo.get_latest_timestamp.return_value = None

        # モックマーケットサービスの設定
        with patch.object(self.service, "market_service") as mock_market_service:
            mock_ohlcv_data = [
                [1704110400000, 45000, 45100, 44900, 45050, 100],
            ]
            mock_market_service.fetch_ohlcv_data = AsyncMock(
                return_value=mock_ohlcv_data
            )
            mock_market_service._save_ohlcv_to_database = AsyncMock(return_value=1)

            # 差分更新を実行
            result = await self.service.collect_incremental_data(
                self.test_symbol, self.test_timeframe, mock_repo
            )

            # 結果の検証
            assert result["success"] is True
            assert result["saved_count"] == 1

            # since=Noneで呼び出されたことを確認
            mock_market_service.fetch_ohlcv_data.assert_called_once_with(
                self.test_symbol, self.test_timeframe, 1000, since=None
            )

    async def test_collect_incremental_data_no_new_data(self):
        """新しいデータがない場合のテスト"""
        # モックリポジトリの設定
        mock_repo = Mock(spec=OHLCVRepository)
        latest_timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_repo.get_latest_timestamp.return_value = latest_timestamp

        # モックマーケットサービスの設定（データなし）
        with patch.object(self.service, "market_service") as mock_market_service:
            mock_market_service.fetch_ohlcv_data = AsyncMock(return_value=None)

            # 差分更新を実行
            result = await self.service.collect_incremental_data(
                self.test_symbol, self.test_timeframe, mock_repo
            )

            # 結果の検証
            assert result["success"] is True
            assert result["message"] == "新しいデータはありません"
            assert result["saved_count"] == 0

            # _save_ohlcv_to_databaseが呼び出されないことを確認
            mock_market_service._save_ohlcv_to_database.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

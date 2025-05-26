"""
データ管理APIの単体テスト

TDDアプローチでOHLCVデータ保存機能をテストします。
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime
import json

from app.main import create_app
from app.core.services.market_data_service import BybitMarketDataService


class TestDataManagementAPI:
    """データ管理APIのテストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        app = create_app()
        self.client = TestClient(app)
        self.test_symbol = "BTC/USD:BTC"
        self.test_timeframe = "1h"
        self.test_limit = 100

        # モックOHLCVデータ
        self.mock_ohlcv_data = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 1000.0],
            [1640998800000, 47200.0, 47600.0, 47000.0, 47400.0, 1200.0],
            [1641002400000, 47400.0, 47800.0, 47300.0, 47600.0, 900.0],
        ]

    def test_save_ohlcv_endpoint_exists(self):
        """保存エンドポイントが存在することをテスト（最初は失敗する）"""
        response = self.client.post("/api/v1/market-data/save-ohlcv")
        # 最初は404が返される（エンドポイントが存在しないため）
        assert response.status_code != 404, "保存エンドポイントが存在しません"

    @patch('app.api.v1.data_management.get_market_data_service')
    def test_save_ohlcv_success(self, mock_get_service):
        """OHLCVデータ保存成功のテスト"""
        # モックサービスの設定
        mock_service = Mock(spec=BybitMarketDataService)
        mock_service.fetch_ohlcv_data = AsyncMock(return_value=self.mock_ohlcv_data)
        mock_service.save_ohlcv_to_database = AsyncMock(return_value=3)  # 3件保存
        mock_service.normalize_symbol.return_value = self.test_symbol
        mock_get_service.return_value = mock_service

        # リクエストデータ
        request_data = {
            "symbol": self.test_symbol,
            "timeframe": self.test_timeframe,
            "limit": self.test_limit
        }

        # APIコール
        response = self.client.post(
            "/api/v1/market-data/save-ohlcv",
            json=request_data
        )

        # レスポンス検証
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["records_saved"] == 3
        assert data["symbol"] == self.test_symbol
        assert data["timeframe"] == self.test_timeframe
        assert "message" in data
        assert "timestamp" in data

        # サービスメソッドが正しく呼ばれたことを確認
        mock_service.fetch_ohlcv_data.assert_called_once_with(
            self.test_symbol, self.test_timeframe, self.test_limit
        )
        mock_service.save_ohlcv_to_database.assert_called_once()

    @patch('app.api.v1.data_management.get_market_data_service')
    def test_save_ohlcv_invalid_symbol(self, mock_get_service):
        """無効なシンボルでのエラーテスト"""
        # モックサービスの設定（無効なシンボルエラー）
        mock_service = Mock(spec=BybitMarketDataService)
        mock_service.fetch_ohlcv_data = AsyncMock(
            side_effect=ValueError("無効なシンボルです: INVALID")
        )
        mock_get_service.return_value = mock_service

        # リクエストデータ
        request_data = {
            "symbol": "INVALID",
            "timeframe": self.test_timeframe,
            "limit": self.test_limit
        }

        # APIコール
        response = self.client.post(
            "/api/v1/market-data/save-ohlcv",
            json=request_data
        )

        # レスポンス検証
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["success"] is False
        assert "無効なシンボル" in data["detail"]["message"]

    @patch('app.api.v1.data_management.get_market_data_service')
    def test_save_ohlcv_database_error(self, mock_get_service):
        """データベースエラーのテスト"""
        # モックサービスの設定（データベースエラー）
        mock_service = Mock(spec=BybitMarketDataService)
        mock_service.fetch_ohlcv_data = AsyncMock(return_value=self.mock_ohlcv_data)
        mock_service.save_ohlcv_to_database = AsyncMock(
            side_effect=Exception("データベース接続エラー")
        )
        mock_service.normalize_symbol.return_value = self.test_symbol
        mock_get_service.return_value = mock_service

        # リクエストデータ
        request_data = {
            "symbol": self.test_symbol,
            "timeframe": self.test_timeframe,
            "limit": self.test_limit
        }

        # APIコール
        response = self.client.post(
            "/api/v1/market-data/save-ohlcv",
            json=request_data
        )

        # レスポンス検証
        assert response.status_code == 500
        data = response.json()
        assert data["detail"]["success"] is False
        assert "内部サーバーエラー" in data["detail"]["message"]

    def test_save_ohlcv_missing_parameters(self):
        """必須パラメータ不足のテスト"""
        # symbolパラメータなし
        request_data = {
            "timeframe": self.test_timeframe,
            "limit": self.test_limit
        }

        response = self.client.post(
            "/api/v1/market-data/save-ohlcv",
            json=request_data
        )

        # バリデーションエラーが返される
        assert response.status_code == 422

    def test_save_ohlcv_invalid_limit(self):
        """無効なlimitパラメータのテスト"""
        # limitが範囲外
        request_data = {
            "symbol": self.test_symbol,
            "timeframe": self.test_timeframe,
            "limit": 2000  # 上限を超える値
        }

        response = self.client.post(
            "/api/v1/market-data/save-ohlcv",
            json=request_data
        )

        # バリデーションエラーが返される
        assert response.status_code == 422

    @patch('app.api.v1.data_management.get_market_data_service')
    def test_save_ohlcv_duplicate_data(self, mock_get_service):
        """重複データ処理のテスト"""
        # モックサービスの設定（重複により0件保存）
        mock_service = Mock(spec=BybitMarketDataService)
        mock_service.fetch_ohlcv_data = AsyncMock(return_value=self.mock_ohlcv_data)
        mock_service.save_ohlcv_to_database = AsyncMock(return_value=0)  # 重複により0件
        mock_service.normalize_symbol.return_value = self.test_symbol
        mock_get_service.return_value = mock_service

        # リクエストデータ
        request_data = {
            "symbol": self.test_symbol,
            "timeframe": self.test_timeframe,
            "limit": self.test_limit
        }

        # APIコール
        response = self.client.post(
            "/api/v1/market-data/save-ohlcv",
            json=request_data
        )

        # レスポンス検証
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["records_saved"] == 0
        assert "重複" in data["message"] or "既に存在" in data["message"]


class TestMarketDataServiceSaveMethod:
    """MarketDataServiceの保存メソッドのテスト"""

    @patch('database.repository.OHLCVRepository')
    @patch('database.connection.get_db')
    def test_save_ohlcv_to_database_method_exists(self, mock_get_db, mock_repo_class):
        """save_ohlcv_to_databaseメソッドが存在することをテスト"""
        from app.core.services.market_data_service import BybitMarketDataService

        # モックの設定
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo

        service = BybitMarketDataService()

        # メソッドが存在することを確認（最初は失敗する）
        assert hasattr(service, 'save_ohlcv_to_database'), \
            "save_ohlcv_to_databaseメソッドが存在しません"

    @patch('app.core.services.market_data_service.OHLCVRepository')
    @patch('app.core.services.market_data_service.get_db')
    @pytest.mark.asyncio
    async def test_save_ohlcv_to_database_conversion(self, mock_get_db, mock_repo_class):
        """OHLCVデータの変換と保存のテスト"""
        from app.core.services.market_data_service import BybitMarketDataService

        # モックの設定
        mock_db = Mock()
        mock_get_db.return_value = iter([mock_db])  # generatorを返す
        mock_repo = Mock()
        mock_repo.insert_ohlcv_data.return_value = 3
        mock_repo_class.return_value = mock_repo

        service = BybitMarketDataService()

        # テストデータ
        ohlcv_data = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 1000.0],
            [1640998800000, 47200.0, 47600.0, 47000.0, 47400.0, 1200.0],
        ]

        # メソッド実行
        result = await service.save_ohlcv_to_database(
            ohlcv_data, "BTC/USD:BTC", "1h"
        )

        # 結果検証
        assert result == 3

        # リポジトリメソッドが正しいデータで呼ばれたことを確認
        mock_repo.insert_ohlcv_data.assert_called_once()
        call_args = mock_repo.insert_ohlcv_data.call_args[0][0]

        # 変換されたデータの構造を確認
        assert len(call_args) == 2
        assert all('symbol' in record for record in call_args)
        assert all('timeframe' in record for record in call_args)
        assert all('timestamp' in record for record in call_args)
        assert all('open' in record for record in call_args)
        assert all('high' in record for record in call_args)
        assert all('low' in record for record in call_args)
        assert all('close' in record for record in call_args)
        assert all('volume' in record for record in call_args)

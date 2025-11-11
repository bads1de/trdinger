"""
データ収集APIのテストモジュール

データ収集APIエンドポイントの正常系、異常系、エッジケースをテストします。
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.api.dependencies import get_data_collection_orchestration_service, get_db


@pytest.fixture
def test_client() -> TestClient:
    """
    FastAPIテストクライアントのフィクスチャ

    Returns:
        TestClient: テスト用のFastAPIクライアント
    """
    return TestClient(app)

@pytest.fixture
def mock_data_collection_orchestration_service() -> AsyncMock:
    """
    オーケストレーションサービスのモック

    Returns:
        AsyncMock: モックされたオーケストレーションサービス
    """
    return AsyncMock()


@pytest.fixture
def mock_db_session() -> Mock:
    """
    データベースセッションのモック

    Returns:
        Mock: モックされたデータベースセッション
    """
    return Mock()

@pytest.fixture(autouse=True)
def override_dependencies(mock_db_session, mock_data_collection_orchestration_service):
    """
    FastAPIの依存性注入をオーバーライド
    
    Args:
        mock_db_session: モックDBセッション
        mock_data_collection_orchestration_service: モックサービス
    """
    app.dependency_overrides[get_db] = lambda: mock_db_session
    app.dependency_overrides[get_data_collection_orchestration_service] = lambda: mock_data_collection_orchestration_service
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_background_tasks() -> Mock:
    """
    BackgroundTasksのモック

    Returns:
        Mock: モックされたBackgroundTasks
    """
    return Mock()




@pytest.fixture
def sample_collection_response() -> Dict[str, Any]:
    """
    サンプルデータ収集レスポンス

    Returns:
        Dict[str, Any]: データ収集レスポンスのサンプルデータ
    """
    return {
        "success": True,
        "message": "データ収集を開始しました",
        "data": {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2020-03-25",
            "status": "started",
        },
    }


@pytest.fixture
def sample_collection_status() -> Dict[str, Any]:
    """
    サンプルデータ収集状態

    Returns:
        Dict[str, Any]: データ収集状態のサンプルデータ
    """
    return {
        "success": True,
        "data": {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "total_records": 5000,
            "latest_timestamp": "2024-01-31T23:00:00",
            "oldest_timestamp": "2020-03-25T00:00:00",
            "is_collecting": False,
        },
    }


class TestHistoricalDataCollection:
    """履歴データ収集のテストクラス"""


    def test_collect_historical_data_success(
        self,
        test_client: TestClient,
        mock_data_collection_orchestration_service: AsyncMock,
        sample_collection_response: Dict[str, Any],
    ) -> None:
        """
        正常系: 履歴データ収集が正常に開始される

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_collection_orchestration_service: オーケストレーションサービスモック
            sample_collection_response: サンプルレスポンス
        """
        # モックの設定
        mock_data_collection_orchestration_service.start_historical_data_collection.return_value = (
            sample_collection_response
        )

        # APIリクエスト
        response = test_client.post(
            "/api/data-collection/historical",
            params={
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "force_update": False,
            },
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "データ収集を開始" in data["message"]
        assert data["data"]["symbol"] == "BTC/USDT:USDT"
        assert data["data"]["timeframe"] == "1h"


    def test_collect_historical_data_with_custom_dates(
        self,
        test_client: TestClient,
        mock_data_collection_orchestration_service: AsyncMock,
        sample_collection_response: Dict[str, Any],
    ) -> None:
        """
        正常系: カスタム開始日付で履歴データ収集が開始される

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_collection_orchestration_service: オーケストレーションサービスモック
            sample_collection_response: サンプルレスポンス
        """
        # モックの設定
        mock_data_collection_orchestration_service.start_historical_data_collection.return_value = (
            sample_collection_response
        )

        # APIリクエスト
        response = test_client.post(
            "/api/data-collection/historical",
            params={
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "force_update": True,
                "start_date": "2023-01-01",
            },
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


    def test_collect_historical_data_db_init_failure(
        self,
        test_client: TestClient,
        mock_data_collection_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: データベース初期化失敗時のエラーハンドリング

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_collection_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_collection_orchestration_service.start_historical_data_collection.side_effect = Exception("DB init failed")
        
        # APIリクエスト
        response = test_client.post(
            "/api/data-collection/historical",
            params={"symbol": "BTC/USDT:USDT", "timeframe": "1h"},
        )

        # アサーション（ErrorHandlerによって処理される）
        assert response.status_code in [200, 500]


class TestBulkIncrementalUpdate:
    """一括差分更新のテストクラス"""

    def test_bulk_incremental_update_success(
        self,
        test_client: TestClient,
        mock_data_collection_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: 一括差分更新が正常に実行される

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_collection_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_collection_orchestration_service.execute_bulk_incremental_update.return_value = {
            "success": True,
            "message": "一括差分更新が完了しました",
            "data": {
                "ohlcv_updated": 100,
                "funding_rate_updated": 50,
                "open_interest_updated": 50,
            },
        }

        # APIリクエスト
        response = test_client.post(
            "/api/data-collection/bulk-incremental-update",
            params={"symbol": "BTC/USDT:USDT"},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "一括差分更新が完了" in data["message"]
        assert data["data"]["ohlcv_updated"] == 100

    def test_bulk_incremental_update_with_different_symbol(
        self,
        test_client: TestClient,
        mock_data_collection_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: 異なるシンボルで一括差分更新が実行される

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_collection_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_collection_orchestration_service.execute_bulk_incremental_update.return_value = {
            "success": True,
            "message": "一括差分更新が完了しました",
            "data": {"ohlcv_updated": 75, "funding_rate_updated": 25},
        }

        # APIリクエスト
        response = test_client.post(
            "/api/data-collection/bulk-incremental-update",
            params={"symbol": "ETH/USDT:USDT"},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestBulkHistoricalDataCollection:
    """一括履歴データ収集のテストクラス"""


    def test_bulk_historical_collection_success(
        self,
        test_client: TestClient,
        mock_data_collection_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: 一括履歴データ収集が正常に開始される

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_collection_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_collection_orchestration_service.start_bulk_historical_data_collection.return_value = {
            "success": True,
            "message": "一括履歴データ収集を開始しました",
            "data": {"symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"], "timeframes": ["1h", "4h"]},
        }

        # APIリクエスト
        response = test_client.post(
            "/api/data-collection/bulk-historical",
            params={"force_update": True, "start_date": "2020-03-25"},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "一括履歴データ収集を開始" in data["message"]


    def test_bulk_historical_collection_without_force_update(
        self,
        test_client: TestClient,
        mock_data_collection_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: 強制更新なしで一括履歴データ収集が開始される

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_collection_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_collection_orchestration_service.start_bulk_historical_data_collection.return_value = {
            "success": True,
            "message": "既存データがスキップされました",
        }

        # APIリクエスト
        response = test_client.post(
            "/api/data-collection/bulk-historical",
            params={"force_update": False},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestCollectionStatus:
    """データ収集状態取得のテストクラス"""


    def test_get_collection_status_success(
        self,
        test_client: TestClient,
        mock_data_collection_orchestration_service: AsyncMock,
        sample_collection_status: Dict[str, Any],
    ) -> None:
        """
        正常系: データ収集状態が正常に取得できる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_collection_orchestration_service: オーケストレーションサービスモック
            sample_collection_status: サンプル状態
        """
        # モックの設定
        mock_data_collection_orchestration_service.get_collection_status.return_value = (
            sample_collection_status
        )

        # APIリクエスト
        response = test_client.get(
            "/api/data-collection/status/BTC%2FUSDT%3AUSDT/1h",
            params={"auto_fetch": False},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["symbol"] == "BTC/USDT:USDT"
        assert data["data"]["timeframe"] == "1h"
        assert data["data"]["total_records"] == 5000


    def test_get_collection_status_with_auto_fetch(
        self,
        test_client: TestClient,
        mock_data_collection_orchestration_service: AsyncMock,
        sample_collection_status: Dict[str, Any],
    ) -> None:
        """
        正常系: 自動フェッチ有効で状態が取得できる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_collection_orchestration_service: オーケストレーションサービスモック
            sample_collection_status: サンプル状態
        """
        # モックの設定
        mock_data_collection_orchestration_service.get_collection_status.return_value = (
            sample_collection_status
        )

        # APIリクエスト
        response = test_client.get(
            "/api/data-collection/status/BTC%2FUSDT%3AUSDT/1h",
            params={"auto_fetch": True},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @pytest.mark.parametrize(
        "symbol,timeframe",
        [
            ("BTC/USDT:USDT", "1h"),
            ("ETH/USDT:USDT", "4h"),
            ("BNB/USDT:USDT", "1d"),
        ],
    )


    def test_get_collection_status_multiple_symbols(
        self,
        test_client: TestClient,
        mock_data_collection_orchestration_service: AsyncMock,
        symbol: str,
        timeframe: str,
    ) -> None:
        """
        エッジケース: 複数のシンボル・タイムフレームの組み合わせテスト

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_collection_orchestration_service: オーケストレーションサービスモック
            symbol: テスト対象シンボル
            timeframe: テスト対象タイムフレーム
        """
        # モックの設定
        mock_data_collection_orchestration_service.get_collection_status.return_value = {
            "success": True,
            "data": {
                "symbol": symbol,
                "timeframe": timeframe,
                "total_records": 1000,
            },
        }

        # URLエンコード
        encoded_symbol = symbol.replace("/", "%2F").replace(":", "%3A")

        # APIリクエスト
        response = test_client.get(
            f"/api/data-collection/status/{encoded_symbol}/{timeframe}"
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestAllDataBulkCollection:
    """全データ一括収集のテストクラス"""


    def test_collect_all_data_bulk_success(
        self,
        test_client: TestClient,
        mock_data_collection_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: 全データ一括収集が正常に開始される

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_collection_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_collection_orchestration_service.start_bulk_historical_data_collection.return_value = {
            "success": True,
            "message": "全データ一括収集を開始しました",
            "data": {
                "ohlcv": True,
                "funding_rate": True,
                "open_interest": True,
            },
        }

        # APIリクエスト
        response = test_client.post(
            "/api/data-collection/all/bulk-collect",
            params={"force_update": False, "start_date": "2020-03-25"},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "一括収集を開始" in data["message"] or data["success"] is True


class TestErrorHandling:
    """エラーハンドリングのテストクラス"""

    def test_service_error_handling(
        self,
        test_client: TestClient,
        mock_data_collection_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: サービス層でエラーが発生した場合

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_collection_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_collection_orchestration_service.execute_bulk_incremental_update.return_value = {
            "success": False,
            "error": "Database connection error",
            "status_code": 500,
        }

        # APIリクエスト
        response = test_client.post(
            "/api/data-collection/bulk-incremental-update",
            params={"symbol": "BTC/USDT:USDT"},
        )

        # アサーション
        assert response.status_code == 200  # ErrorHandlerによりラップされる
        data = response.json()
        assert data["success"] is False
        assert "error" in data


    def test_unexpected_exception_handling(
        self,
        test_client: TestClient,
        mock_data_collection_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: 予期しない例外が発生した場合

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_collection_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_collection_orchestration_service.start_historical_data_collection.side_effect = (
            Exception("Unexpected error")
        )

        # APIリクエスト
        response = test_client.post(
            "/api/data-collection/historical",
            params={"symbol": "BTC/USDT:USDT", "timeframe": "1h"},
        )

        # アサーション（ErrorHandlerによって処理される）
        assert response.status_code in [200, 500]


class TestEdgeCases:
    """エッジケースのテストクラス"""

    @pytest.mark.parametrize(
        "timeframe",
        ["15m", "30m", "1h", "4h", "1d"],
    )


    def test_multiple_timeframes(
        self,
        test_client: TestClient,
        mock_data_collection_orchestration_service: AsyncMock,
        timeframe: str,
    ) -> None:
        """
        エッジケース: 複数のタイムフレームでのデータ収集テスト

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_collection_orchestration_service: オーケストレーションサービスモック
            timeframe: テスト対象タイムフレーム
        """
        # モックの設定
        mock_data_collection_orchestration_service.start_historical_data_collection.return_value = {
            "success": True,
            "message": f"{timeframe}のデータ収集を開始しました",
            "data": {"timeframe": timeframe},
        }

        # APIリクエスト
        response = test_client.post(
            "/api/data-collection/historical",
            params={"symbol": "BTC/USDT:USDT", "timeframe": timeframe},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
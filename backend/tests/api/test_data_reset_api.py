"""
データリセットAPIのテストモジュール

データリセットAPIエンドポイントの正常系、異常系、エッジケースをテストします。
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.dependencies import get_data_management_orchestration_service, get_db


@pytest.fixture
def test_client() -> TestClient:
    """
    FastAPIテストクライアントのフィクスチャ

    Returns:
        TestClient: テスト用のFastAPIクライアント
    """
    return TestClient(app)


@pytest.fixture
def mock_data_management_orchestration_service() -> AsyncMock:
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
def override_dependencies(mock_db_session, mock_data_management_orchestration_service):
    """
    FastAPIの依存性注入をオーバーライド

    Args:
        mock_db_session: モックDBセッション
        mock_data_management_orchestration_service: モックサービス
    """
    app.dependency_overrides[get_db] = lambda: mock_db_session
    app.dependency_overrides[get_data_management_orchestration_service] = (
        lambda: mock_data_management_orchestration_service
    )
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def sample_reset_result() -> Dict[str, Any]:
    """
    サンプルリセット結果

    Returns:
        Dict[str, Any]: リセット結果のサンプル
    """
    return {
        "success": True,
        "message": "データをリセットしました",
        "deleted_counts": {
            "ohlcv": 1000,
            "funding_rates": 500,
            "open_interest": 300,
        },
    }


class TestResetAllData:
    """全データリセットのテストクラス"""

    def test_reset_all_data_success(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_data_management_orchestration_service: AsyncMock,
        sample_reset_result: Dict[str, Any],
    ) -> None:
        """
        正常系: 全データが正常にリセットされる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_management_orchestration_service: オーケストレーションサービスモック
            sample_reset_result: サンプルリセット結果
        """
        # モックの設定
        mock_data_management_orchestration_service.reset_all_data.return_value = (
            sample_reset_result
        )

        # APIリクエスト
        response = test_client.delete("/api/data-reset/all")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deleted_counts" in data
        assert data["deleted_counts"]["ohlcv"] == 1000

    def test_reset_all_data_empty(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_data_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        エッジケース: データが空の状態でリセット

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_management_orchestration_service.reset_all_data.return_value = {
            "success": True,
            "message": "データはすでに空です",
            "deleted_counts": {"ohlcv": 0, "funding_rates": 0, "open_interest": 0},
        }

        # APIリクエスト
        response = test_client.delete("/api/data-reset/all")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deleted_counts"]["ohlcv"] == 0


class TestResetOHLCVData:
    """OHLCVデータリセットのテストクラス"""

    def test_reset_ohlcv_success(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_data_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: OHLCVデータが正常にリセットされる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_management_orchestration_service.reset_ohlcv_data.return_value = {
            "success": True,
            "message": "OHLCVデータをリセットしました",
            "deleted_count": 1000,
        }

        # APIリクエスト
        response = test_client.delete("/api/data-reset/ohlcv")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deleted_count"] == 1000


class TestResetFundingRateData:
    """ファンディングレートデータリセットのテストクラス"""

    def test_reset_funding_rate_success(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_data_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: ファンディングレートデータが正常にリセットされる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_management_orchestration_service.reset_funding_rate_data.return_value = {
            "success": True,
            "message": "ファンディングレートデータをリセットしました",
            "deleted_count": 500,
        }

        # APIリクエスト
        response = test_client.delete("/api/data-reset/funding-rates")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deleted_count"] == 500


class TestResetOpenInterestData:
    """オープンインタレストデータリセットのテストクラス"""

    def test_reset_open_interest_success(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_data_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: オープンインタレストデータが正常にリセットされる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_management_orchestration_service.reset_open_interest_data.return_value = {
            "success": True,
            "message": "オープンインタレストデータをリセットしました",
            "deleted_count": 300,
        }

        # APIリクエスト
        response = test_client.delete("/api/data-reset/open-interest")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deleted_count"] == 300


class TestResetDataBySymbol:
    """シンボル別データリセットのテストクラス"""

    def test_reset_by_symbol_success(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_data_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: シンボル別データが正常にリセットされる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_management_orchestration_service.reset_data_by_symbol.return_value = {
            "success": True,
            "message": "BTC/USDT:USDTのデータをリセットしました",
            "deleted_counts": {
                "ohlcv": 500,
                "funding_rates": 250,
                "open_interest": 150,
            },
        }

        # APIリクエスト
        response = test_client.delete("/api/data-reset/symbol/BTC/USDT:USDT")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deleted_counts" in data

    @pytest.mark.parametrize(
        "symbol",
        ["BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT"],
    )
    def test_reset_by_different_symbols(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_data_management_orchestration_service: AsyncMock,
        symbol: str,
    ) -> None:
        """
        正常系: 異なるシンボルでデータがリセットされる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_management_orchestration_service: オーケストレーションサービスモック
            symbol: シンボル
        """
        # モックの設定
        mock_data_management_orchestration_service.reset_data_by_symbol.return_value = {
            "success": True,
            "message": f"{symbol}のデータをリセットしました",
            "deleted_counts": {"ohlcv": 100, "funding_rates": 50, "open_interest": 30},
        }

        # APIリクエスト
        response = test_client.delete(f"/api/data-reset/symbol/{symbol}")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestGetDataStatus:
    """データステータス取得のテストクラス"""

    def test_get_data_status_success(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_data_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: データステータスが正常に取得できる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_management_orchestration_service.get_data_status.return_value = {
            "success": True,
            "data": {
                "ohlcv": {
                    "count": 1000,
                    "oldest": "2024-01-01T00:00:00",
                    "newest": "2024-01-31T23:59:59",
                },
                "funding_rates": {
                    "count": 500,
                    "oldest": "2024-01-01T00:00:00",
                    "newest": "2024-01-31T23:59:59",
                },
                "open_interest": {
                    "count": 300,
                    "oldest": "2024-01-01T00:00:00",
                    "newest": "2024-01-31T23:59:59",
                },
            },
        }

        # APIリクエスト
        response = test_client.get("/api/data-reset/status")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "ohlcv" in data["data"]
        assert data["data"]["ohlcv"]["count"] == 1000

    def test_get_data_status_empty(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_data_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        エッジケース: データが空の場合

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_management_orchestration_service.get_data_status.return_value = {
            "success": True,
            "data": {
                "ohlcv": {"count": 0, "oldest": None, "newest": None},
                "funding_rates": {"count": 0, "oldest": None, "newest": None},
                "open_interest": {"count": 0, "oldest": None, "newest": None},
            },
        }

        # APIリクエスト
        response = test_client.get("/api/data-reset/status")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["ohlcv"]["count"] == 0


class TestErrorHandling:
    """エラーハンドリングのテストクラス"""

    def test_reset_all_data_error(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_data_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: リセット時にエラーが発生する

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_management_orchestration_service.reset_all_data.side_effect = (
            Exception("Database error")
        )

        # APIリクエスト
        response = test_client.delete("/api/data-reset/all")

        # アサーション（ErrorHandlerによって処理される）
        assert response.status_code in [200, 500]

    def test_reset_by_symbol_not_found(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_data_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        エッジケース: 存在しないシンボルのリセット

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_management_orchestration_service.reset_data_by_symbol.return_value = {
            "success": True,
            "message": "該当するデータがありません",
            "deleted_counts": {"ohlcv": 0, "funding_rates": 0, "open_interest": 0},
        }

        # APIリクエスト
        response = test_client.delete("/api/data-reset/symbol/INVALID/SYMBOL")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deleted_counts"]["ohlcv"] == 0

    def test_get_status_error(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_data_management_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: ステータス取得時にエラーが発生する

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_data_management_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_data_management_orchestration_service.get_data_status.side_effect = (
            Exception("Database error")
        )

        # APIリクエスト
        response = test_client.get("/api/data-reset/status")

        # アサーション（ErrorHandlerによって処理される）
        assert response.status_code in [200, 500]

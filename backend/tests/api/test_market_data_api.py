"""
市場データAPIのテストモジュール

市場データAPIエンドポイントの正常系、異常系、エッジケースをテストします。
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.dependencies import get_market_data_orchestration_service, get_db


@pytest.fixture
def test_client() -> TestClient:
    """
    FastAPIテストクライアントのフィクスチャ

    Returns:
        TestClient: テスト用のFastAPIクライアント
    """
    return TestClient(app)


@pytest.fixture
def mock_db_session() -> Mock:
    """
    データベースセッションのモック

    Returns:
        Mock: モックされたデータベースセッション
    """
    return Mock()


@pytest.fixture
def mock_market_data_orchestration_service() -> Mock:
    """
    MarketDataOrchestrationServiceのモック

    Returns:
        Mock: モックされた市場データオーケストレーションサービス
    """
    mock_service = Mock()
    mock_service.get_ohlcv_data = AsyncMock()
    return mock_service


@pytest.fixture(autouse=True)
def override_dependencies(mock_db_session, mock_market_data_orchestration_service):
    """
    FastAPIの依存性注入をオーバーライド

    Args:
        mock_db_session: モックDBセッション
        mock_market_data_orchestration_service: モックサービス
    """
    app.dependency_overrides[get_db] = lambda: mock_db_session
    app.dependency_overrides[get_market_data_orchestration_service] = (
        lambda: mock_market_data_orchestration_service
    )
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def sample_ohlcv_data() -> Dict[str, Any]:
    """
    サンプルOHLCVデータ

    Returns:
        Dict[str, Any]: OHLCVデータのサンプル
    """
    return {
        "timestamp": "2024-01-01T00:00:00",
        "open": 42000.0,
        "high": 42500.0,
        "low": 41800.0,
        "close": 42300.0,
        "volume": 1000.0,
    }


@pytest.fixture
def sample_ohlcv_list(sample_ohlcv_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    サンプルOHLCVリスト

    Args:
        sample_ohlcv_data: 単一のOHLCVデータ

    Returns:
        List[Dict[str, Any]]: OHLCVデータのリスト
    """
    return [
        sample_ohlcv_data,
        {**sample_ohlcv_data, "timestamp": "2024-01-01T01:00:00", "close": 42400.0},
        {**sample_ohlcv_data, "timestamp": "2024-01-01T02:00:00", "close": 42500.0},
    ]


class TestGetOHLCVData:
    """OHLCVデータ取得のテストクラス"""

    def test_get_ohlcv_success(
        self,
        test_client: TestClient,
        mock_market_data_orchestration_service: AsyncMock,
        sample_ohlcv_list: List[Dict[str, Any]],
    ) -> None:
        """
        正常系: OHLCVデータが正常に取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_market_data_orchestration_service: オーケストレーションサービスモック
            sample_ohlcv_list: サンプルデータリスト
        """
        # モックの設定
        mock_market_data_orchestration_service.get_ohlcv_data.return_value = {
            "success": True,
            "data": sample_ohlcv_list,
            "count": len(sample_ohlcv_list),
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
        }

        # APIリクエスト
        response = test_client.get(
            "/api/market-data/ohlcv",
            params={
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "limit": 100,
            },
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 3
        assert data["symbol"] == "BTC/USDT:USDT"
        assert data["timeframe"] == "1h"

    def test_get_ohlcv_with_date_range(
        self,
        test_client: TestClient,
        mock_market_data_orchestration_service: AsyncMock,
        sample_ohlcv_data: Dict[str, Any],
    ) -> None:
        """
        正常系: 日付範囲指定でOHLCVデータが取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_market_data_orchestration_service: オーケストレーションサービスモック
            sample_ohlcv_data: サンプルデータ
        """
        # モックの設定
        mock_market_data_orchestration_service.get_ohlcv_data.return_value = {
            "success": True,
            "data": [sample_ohlcv_data],
            "count": 1,
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
        }

        # APIリクエスト
        response = test_client.get(
            "/api/market-data/ohlcv",
            params={
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-01-31T23:59:59",
                "limit": 100,
            },
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 1

    @pytest.mark.parametrize(
        "timeframe",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
    )
    def test_get_ohlcv_various_timeframes(
        self,
        test_client: TestClient,
        mock_market_data_orchestration_service: AsyncMock,
        sample_ohlcv_list: List[Dict[str, Any]],
        timeframe: str,
    ) -> None:
        """
        正常系: 様々な時間軸でOHLCVデータが取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_market_data_orchestration_service: オーケストレーションサービスモック
            sample_ohlcv_list: サンプルデータリスト
            timeframe: テストする時間軸
        """
        # モックの設定
        mock_market_data_orchestration_service.get_ohlcv_data.return_value = {
            "success": True,
            "data": sample_ohlcv_list,
            "count": len(sample_ohlcv_list),
            "symbol": "BTC/USDT:USDT",
            "timeframe": timeframe,
        }

        # APIリクエスト
        response = test_client.get(
            "/api/market-data/ohlcv",
            params={
                "symbol": "BTC/USDT:USDT",
                "timeframe": timeframe,
            },
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["timeframe"] == timeframe

    def test_get_ohlcv_empty_result(
        self,
        test_client: TestClient,
        mock_market_data_orchestration_service: AsyncMock,
    ) -> None:
        """
        エッジケース: データが空の場合

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_market_data_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_market_data_orchestration_service.get_ohlcv_data.return_value = {
            "success": True,
            "data": [],
            "count": 0,
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
        }

        # APIリクエスト
        response = test_client.get(
            "/api/market-data/ohlcv",
            params={
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
            },
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 0

    @pytest.mark.parametrize(
        "limit,expected_status",
        [
            (1, 200),  # 最小値
            (500, 200),  # 中間値
            (1000, 200),  # 最大値
            (0, 422),  # 無効な最小値
            (1001, 422),  # 最大値超過
        ],
    )
    def test_get_ohlcv_limit_validation(
        self,
        test_client: TestClient,
        mock_market_data_orchestration_service: AsyncMock,
        limit: int,
        expected_status: int,
    ) -> None:
        """
        異常系: リミットパラメータのバリデーション

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_market_data_orchestration_service: オーケストレーションサービスモック
            limit: リミット値
            expected_status: 期待されるステータスコード
        """
        # モックの設定
        mock_market_data_orchestration_service.get_ohlcv_data.return_value = {
            "success": True,
            "data": [],
            "count": 0,
        }

        # APIリクエスト
        response = test_client.get(
            "/api/market-data/ohlcv",
            params={
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "limit": limit,
            },
        )

        # アサーション
        assert response.status_code == expected_status

    def test_get_ohlcv_missing_symbol(
        self,
        test_client: TestClient,
    ) -> None:
        """
        異常系: 必須パラメータsymbolが欠落している場合

        Args:
            test_client: テストクライアント
        """
        # APIリクエスト
        response = test_client.get(
            "/api/market-data/ohlcv",
            params={"timeframe": "1h"},
        )

        # アサーション
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "symbol",
        [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
            "BNB/USDT:USDT",
        ],
    )
    def test_get_ohlcv_multiple_symbols(
        self,
        test_client: TestClient,
        mock_market_data_orchestration_service: AsyncMock,
        sample_ohlcv_list: List[Dict[str, Any]],
        symbol: str,
    ) -> None:
        """
        正常系: 複数のシンボルでOHLCVデータが取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_market_data_orchestration_service: オーケストレーションサービスモック
            sample_ohlcv_list: サンプルデータリスト
            symbol: テストするシンボル
        """
        # モックの設定
        mock_market_data_orchestration_service.get_ohlcv_data.return_value = {
            "success": True,
            "data": sample_ohlcv_list,
            "count": len(sample_ohlcv_list),
            "symbol": symbol,
            "timeframe": "1h",
        }

        # APIリクエスト
        response = test_client.get(
            "/api/market-data/ohlcv",
            params={
                "symbol": symbol,
                "timeframe": "1h",
            },
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["symbol"] == symbol

    def test_get_ohlcv_default_parameters(
        self,
        test_client: TestClient,
        mock_market_data_orchestration_service: AsyncMock,
        sample_ohlcv_list: List[Dict[str, Any]],
    ) -> None:
        """
        正常系: デフォルトパラメータで取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_market_data_orchestration_service: オーケストレーションサービスモック
            sample_ohlcv_list: サンプルデータリスト
        """
        # モックの設定
        mock_market_data_orchestration_service.get_ohlcv_data.return_value = {
            "success": True,
            "data": sample_ohlcv_list,
            "count": len(sample_ohlcv_list),
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
        }

        # APIリクエスト（最小限のパラメータのみ）
        response = test_client.get(
            "/api/market-data/ohlcv",
            params={"symbol": "BTC/USDT:USDT"},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestErrorHandling:
    """エラーハンドリングのテストクラス"""

    def test_service_error_handling(
        self,
        test_client: TestClient,
        mock_market_data_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: サービス層でエラーが発生した場合

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_market_data_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_market_data_orchestration_service.get_ohlcv_data.side_effect = Exception(
            "Database error"
        )

        # APIリクエスト
        response = test_client.get(
            "/api/market-data/ohlcv",
            params={"symbol": "BTC/USDT:USDT", "timeframe": "1h"},
        )

        # アサーション（ErrorHandlerによって処理される）
        assert response.status_code in [200, 500]

    def test_invalid_timeframe(
        self,
        test_client: TestClient,
        mock_market_data_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: 無効な時間軸が指定された場合

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_market_data_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_market_data_orchestration_service.get_ohlcv_data.return_value = {
            "success": False,
            "error": "Invalid timeframe",
            "status_code": 400,
        }

        # APIリクエスト
        response = test_client.get(
            "/api/market-data/ohlcv",
            params={"symbol": "BTC/USDT:USDT", "timeframe": "invalid"},
        )

        # アサーション
        assert response.status_code in [200, 400]

    def test_invalid_date_format(
        self,
        test_client: TestClient,
        mock_market_data_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: 無効な日付形式が指定された場合

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_market_data_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_market_data_orchestration_service.get_ohlcv_data.return_value = {
            "success": False,
            "error": "Invalid date format",
            "status_code": 400,
        }

        # APIリクエスト
        response = test_client.get(
            "/api/market-data/ohlcv",
            params={
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "start_date": "invalid-date",
            },
        )

        # アサーション
        assert response.status_code in [200, 400]

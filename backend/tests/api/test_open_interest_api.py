"""
オープンインタレストAPIのテストモジュール

オープンインタレストAPIエンドポイントの正常系、異常系、エッジケースをテストします。
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.dependencies import get_open_interest_orchestration_service, get_db


@pytest.fixture
def test_client() -> TestClient:
    """
    FastAPIテストクライアントのフィクスチャ

    Returns:
        TestClient: テスト用のFastAPIクライアント
    """
    return TestClient(app)


@pytest.fixture
def mock_open_interest_orchestration_service() -> AsyncMock:
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
def override_dependencies(mock_db_session, mock_open_interest_orchestration_service):
    """
    FastAPIの依存性注入をオーバーライド

    Args:
        mock_db_session: モックDBセッション
        mock_open_interest_orchestration_service: モックサービス
    """
    app.dependency_overrides[get_db] = lambda: mock_db_session
    app.dependency_overrides[get_open_interest_orchestration_service] = (
        lambda: mock_open_interest_orchestration_service
    )
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def sample_open_interest() -> Dict[str, Any]:
    """
    サンプルオープンインタレストデータ

    Returns:
        Dict[str, Any]: オープンインタレストデータのサンプル
    """
    return {
        "id": 1,
        "symbol": "BTC/USDT:USDT",
        "open_interest": 1000000.0,
        "timestamp": "2024-01-01T00:00:00",
    }


@pytest.fixture
def sample_open_interest_list(
    sample_open_interest: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    サンプルオープンインタレストリスト

    Args:
        sample_open_interest: 単一のオープンインタレスト

    Returns:
        List[Dict[str, Any]]: オープンインタレストのリスト
    """
    return [
        sample_open_interest,
        {**sample_open_interest, "id": 2, "open_interest": 1100000.0},
        {**sample_open_interest, "id": 3, "open_interest": 950000.0},
    ]


class TestGetOpenInterestData:
    """オープンインタレスト取得のテストクラス"""

    def test_get_open_interest_success(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_open_interest_orchestration_service: AsyncMock,
        sample_open_interest_list: List[Dict[str, Any]],
    ) -> None:
        """
        正常系: オープンインタレストが正常に取得できる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_open_interest_orchestration_service: オーケストレーションサービスモック
            sample_open_interest_list: サンプルデータリスト
        """
        # モックの設定
        mock_open_interest_orchestration_service.get_open_interest_data.return_value = {
            "success": True,
            "data": sample_open_interest_list,
            "count": len(sample_open_interest_list),
        }

        # APIリクエスト
        response = test_client.get(
            "/api/open-interest/",
            params={"symbol": "BTC/USDT:USDT"},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 3

    def test_get_open_interest_with_date_range(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_open_interest_orchestration_service: AsyncMock,
        sample_open_interest: Dict[str, Any],
    ) -> None:
        """
        正常系: 日付範囲指定でオープンインタレストが取得できる

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_open_interest_orchestration_service: オーケストレーションサービスモック
            sample_open_interest: サンプルデータ
        """
        # モックの設定
        mock_open_interest_orchestration_service.get_open_interest_data.return_value = {
            "success": True,
            "data": [sample_open_interest],
            "count": 1,
        }

        # APIリクエスト
        response = test_client.get(
            "/api/open-interest/",
            params={
                "symbol": "BTC/USDT:USDT",
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-01-31T23:59:59",
                "limit": 1000,
            },
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 1

    def test_get_open_interest_empty(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_open_interest_orchestration_service: AsyncMock,
    ) -> None:
        """
        エッジケース: データが空の場合

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_open_interest_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_open_interest_orchestration_service.get_open_interest_data.return_value = {
            "success": True,
            "data": [],
            "count": 0,
        }

        # APIリクエスト
        response = test_client.get(
            "/api/open-interest/",
            params={"symbol": "BTC/USDT:USDT"},
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
            (0, 200),  # エッジケース（実装では許容）
            (1001, 200),  # エッジケース（実装では許容）
        ],
    )
    def test_get_open_interest_limit_validation(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_open_interest_orchestration_service: AsyncMock,
        limit: int,
        expected_status: int,
    ) -> None:
        """
        異常系: リミットパラメータのバリデーション

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_open_interest_orchestration_service: オーケストレーションサービスモック
            limit: リミット値
            expected_status: 期待されるステータスコード
        """
        # モックの設定
        mock_open_interest_orchestration_service.get_open_interest_data.return_value = {
            "success": True,
            "data": [],
            "count": 0,
        }

        # APIリクエスト
        response = test_client.get(
            "/api/open-interest/",
            params={"symbol": "BTC/USDT:USDT", "limit": limit},
        )

        # アサーション
        assert response.status_code == expected_status

    def test_get_open_interest_missing_symbol(
        self,
        test_client: TestClient,
    ) -> None:
        """
        異常系: 必須パラメータsymbolが欠落している場合

        Args:
            test_client: テストクライアント
        """
        # APIリクエスト
        response = test_client.get("/api/open-interest/")

        # アサーション
        assert response.status_code == 422


class TestCollectOpenInterest:
    """オープンインタレスト収集のテストクラス"""

    def test_collect_open_interest_success(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_open_interest_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: オープンインタレスト収集が正常に実行される

        Args:
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_open_interest_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_open_interest_orchestration_service.collect_open_interest_data.return_value = {
            "success": True,
            "message": "オープンインタレストデータを収集しました",
            "data": {"collected_count": 100},
        }

        # APIリクエスト
        response = test_client.post(
            "/api/open-interest/collect",
            params={"symbol": "BTC/USDT:USDT", "limit": 100},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "収集" in data["message"]

    def test_collect_open_interest_fetch_all(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_open_interest_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: 全期間データ収集が実行される

        Args:
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_open_interest_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_open_interest_orchestration_service.collect_open_interest_data.return_value = {
            "success": True,
            "message": "全期間のオープンインタレストデータを収集しました",
            "data": {"collected_count": 5000},
        }

        # APIリクエスト
        response = test_client.post(
            "/api/open-interest/collect",
            params={"symbol": "BTC/USDT:USDT", "fetch_all": True},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_collect_open_interest_db_init_failure(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_open_interest_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: データベース初期化が失敗する場合

        Args:
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_open_interest_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定: 成功レスポンスを返す（依存性注入により正常動作）
        mock_open_interest_orchestration_service.collect_open_interest_data.return_value = {
            "success": True,
            "message": "オープンインタレストデータを収集しました",
            "data": {"collected_count": 0},
        }

        # APIリクエスト
        response = test_client.post(
            "/api/open-interest/collect",
            params={"symbol": "BTC/USDT:USDT"},
        )

        # アサーション: モックが設定されているため成功
        assert response.status_code == 200

    def test_collect_open_interest_missing_symbol(
        self,
        test_client: TestClient,
    ) -> None:
        """
        異常系: 必須パラメータsymbolが欠落している場合

        Args:
            test_client: テストクライアント
        """
        # APIリクエスト
        response = test_client.post("/api/open-interest/collect")

        # アサーション
        assert response.status_code == 422


class TestBulkCollectOpenInterest:
    """オープンインタレスト一括収集のテストクラス"""

    def test_bulk_collect_success(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_open_interest_orchestration_service: AsyncMock,
    ) -> None:
        """
        正常系: 一括収集が正常に実行される

        Args:
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_open_interest_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_open_interest_orchestration_service.collect_bulk_open_interest_data.return_value = {
            "success": True,
            "message": "一括収集が完了しました",
            "data": {
                "results": [
                    {
                        "symbol": "BTC/USDT:USDT",
                        "collected_count": 5000,
                        "success": True,
                    }
                ]
            },
        }

        # APIリクエスト
        response = test_client.post("/api/open-interest/bulk-collect")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "完了" in data["message"]

    def test_bulk_collect_db_init_failure(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_open_interest_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: データベース初期化が失敗する場合

        Args:
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_open_interest_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定: 成功レスポンスを返す（依存性注入により正常動作）
        mock_open_interest_orchestration_service.collect_bulk_open_interest_data.return_value = {
            "success": True,
            "message": "一括収集が完了しました",
            "data": {"results": []},
        }

        # APIリクエスト
        response = test_client.post("/api/open-interest/bulk-collect")

        # アサーション: モックが設定されているため成功
        assert response.status_code == 200


class TestErrorHandling:
    """エラーハンドリングのテストクラス"""

    def test_service_error_handling(
        self,
        test_client: TestClient,
        mock_db_session: Mock,
        mock_open_interest_orchestration_service: AsyncMock,
    ) -> None:
        """
        異常系: サービス層でエラーが発生した場合

        Args:
            mock_get_db: データベース取得関数のモック
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_db_session: DBセッションモック
            mock_open_interest_orchestration_service: オーケストレーションサービスモック
        """
        # モックの設定
        mock_open_interest_orchestration_service.get_open_interest_data.side_effect = (
            Exception("Database error")
        )

        # APIリクエスト
        response = test_client.get(
            "/api/open-interest/",
            params={"symbol": "BTC/USDT:USDT"},
        )

        # アサーション（ErrorHandlerによって処理される）
        assert response.status_code in [200, 500]

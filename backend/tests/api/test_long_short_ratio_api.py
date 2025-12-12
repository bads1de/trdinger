"""
ロング/ショート比率APIのテストモジュール

ロング/ショート比率APIエンドポイントの正常系、異常系、エッジケースをテストします。
"""

from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, Mock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_db,
    get_long_short_ratio_repository,
    get_long_short_ratio_service,
)
from app.main import app
from database.models import LongShortRatioData


@pytest.fixture
def test_client() -> TestClient:
    """
    FastAPIテストクライアントのフィクスチャ

    Returns:
        TestClient: テスト用のFastAPIクライアント
    """
    return TestClient(app)


@pytest.fixture
def mock_repository() -> Mock:
    """
    リポジトリのモック

    Returns:
        Mock: モックされたリポジトリ
    """
    mock = Mock()
    mock.to_dict = Mock(
        side_effect=lambda record: {
            "id": getattr(record, "id", 1),
            "symbol": getattr(record, "symbol", "BTC/USDT:USDT"),
            "period": getattr(record, "period", "1h"),
            "buy_ratio": getattr(record, "buy_ratio", 0.6),
            "sell_ratio": getattr(record, "sell_ratio", 0.4),
            "timestamp": getattr(
                record, "timestamp", datetime(2021, 1, 1, tzinfo=timezone.utc)
            ).isoformat(),
        }
    )
    return mock


@pytest.fixture
def mock_service() -> AsyncMock:
    """
    サービスのモック

    Returns:
        AsyncMock: モックされたサービス
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
def override_dependencies(mock_db_session, mock_repository, mock_service):
    """
    FastAPIの依存性注入をオーバーライド

    Args:
        mock_db_session: モックDBセッション
        mock_repository: モックリポジトリ
        mock_service: モックサービス
    """
    app.dependency_overrides[get_db] = lambda: mock_db_session
    app.dependency_overrides[get_long_short_ratio_repository] = lambda: mock_repository
    app.dependency_overrides[get_long_short_ratio_service] = lambda: mock_service
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def sample_ls_ratio_record() -> LongShortRatioData:
    """
    サンプルロング/ショート比率データレコード

    Returns:
        LongShortRatioData: サンプルデータ
    """
    record = MagicMock(spec=LongShortRatioData)
    record.id = 1
    record.symbol = "BTC/USDT:USDT"
    record.period = "1h"
    record.buy_ratio = 0.6
    record.sell_ratio = 0.4
    record.timestamp = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return record


@pytest.fixture
def sample_ls_ratio_list(sample_ls_ratio_record) -> List[LongShortRatioData]:
    """
    サンプルロング/ショート比率リスト

    Args:
        sample_ls_ratio_record: 単一のレコード

    Returns:
        List[LongShortRatioData]: レコードのリスト
    """
    record2 = MagicMock(spec=LongShortRatioData)
    record2.id = 2
    record2.symbol = "BTC/USDT:USDT"
    record2.period = "1h"
    record2.buy_ratio = 0.55
    record2.sell_ratio = 0.45
    record2.timestamp = datetime(2021, 1, 1, 1, 0, 0, tzinfo=timezone.utc)

    return [sample_ls_ratio_record, record2]


class TestGetLongShortRatioData:
    """ロング/ショート比率取得のテストクラス"""

    def test_get_long_short_ratio_success(
        self,
        test_client: TestClient,
        mock_repository: Mock,
        sample_ls_ratio_list: List[LongShortRatioData],
    ) -> None:
        """
        正常系: ロング/ショート比率が正常に取得できる
        """
        # モックの設定
        mock_repository.get_long_short_ratio_data.return_value = sample_ls_ratio_list

        # APIリクエスト
        response = test_client.get(
            "/api/long-short-ratio/",
            params={"symbol": "BTC/USDT:USDT", "period": "1h"},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["symbol"] == "BTC/USDT:USDT"
        assert data[0]["period"] == "1h"

    def test_get_long_short_ratio_with_date_range(
        self,
        test_client: TestClient,
        mock_repository: Mock,
        sample_ls_ratio_record: LongShortRatioData,
    ) -> None:
        """
        正常系: 日付範囲指定でロング/ショート比率が取得できる
        """
        # モックの設定
        mock_repository.get_long_short_ratio_data.return_value = [
            sample_ls_ratio_record
        ]

        # APIリクエスト
        response = test_client.get(
            "/api/long-short-ratio/",
            params={
                "symbol": "BTC/USDT:USDT",
                "period": "1h",
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-01-31T23:59:59",
                "limit": 100,
            },
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1

    def test_get_long_short_ratio_empty(
        self,
        test_client: TestClient,
        mock_repository: Mock,
    ) -> None:
        """
        エッジケース: データが空の場合
        """
        # モックの設定
        mock_repository.get_long_short_ratio_data.return_value = []

        # APIリクエスト
        response = test_client.get(
            "/api/long-short-ratio/",
            params={"symbol": "BTC/USDT:USDT", "period": "1h"},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0

    def test_get_long_short_ratio_missing_symbol(
        self,
        test_client: TestClient,
    ) -> None:
        """
        異常系: 必須パラメータsymbolが欠落している場合
        """
        response = test_client.get(
            "/api/long-short-ratio/",
            params={"period": "1h"},
        )
        assert response.status_code == 422

    def test_get_long_short_ratio_missing_period(
        self,
        test_client: TestClient,
    ) -> None:
        """
        異常系: 必須パラメータperiodが欠落している場合
        """
        response = test_client.get(
            "/api/long-short-ratio/",
            params={"symbol": "BTC/USDT:USDT"},
        )
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "limit,expected_status",
        [
            (1, 200),  # 最小値
            (100, 200),  # デフォルト値
            (500, 200),  # 中間値
            (1000, 200),  # 最大値
        ],
    )
    def test_get_long_short_ratio_limit_validation(
        self,
        test_client: TestClient,
        mock_repository: Mock,
        limit: int,
        expected_status: int,
    ) -> None:
        """
        リミットパラメータのバリデーション
        """
        mock_repository.get_long_short_ratio_data.return_value = []

        response = test_client.get(
            "/api/long-short-ratio/",
            params={"symbol": "BTC/USDT:USDT", "period": "1h", "limit": limit},
        )
        assert response.status_code == expected_status


class TestCollectLongShortRatioData:
    """ロング/ショート比率収集のテストクラス"""

    def test_collect_incremental_success(
        self,
        test_client: TestClient,
        mock_service: AsyncMock,
        mock_repository: Mock,
    ) -> None:
        """
        正常系: 差分データ収集が正常に実行される
        """
        # モックの設定
        mock_service.fetch_incremental_long_short_ratio_data.return_value = {
            "symbol": "BTC/USDT:USDT",
            "saved_count": 10,
            "success": True,
        }

        # APIリクエスト
        response = test_client.post(
            "/api/long-short-ratio/collect",
            params={"symbol": "BTC/USDT:USDT", "period": "1h", "mode": "incremental"},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "開始" in data["message"]
        assert data["symbol"] == "BTC/USDT:USDT"

    def test_collect_historical_success(
        self,
        test_client: TestClient,
        mock_service: AsyncMock,
        mock_repository: Mock,
    ) -> None:
        """
        正常系: 履歴データ収集が正常に実行される
        """
        # モックの設定
        mock_service.collect_historical_long_short_ratio_data.return_value = 100

        # APIリクエスト
        response = test_client.post(
            "/api/long-short-ratio/collect",
            params={"symbol": "BTC/USDT:USDT", "period": "1d", "mode": "historical"},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_collect_missing_symbol(
        self,
        test_client: TestClient,
    ) -> None:
        """
        異常系: 必須パラメータsymbolが欠落している場合
        """
        response = test_client.post(
            "/api/long-short-ratio/collect",
            params={"period": "1h"},
        )
        assert response.status_code == 422

    def test_collect_missing_period(
        self,
        test_client: TestClient,
    ) -> None:
        """
        異常系: 必須パラメータperiodが欠落している場合
        """
        response = test_client.post(
            "/api/long-short-ratio/collect",
            params={"symbol": "BTC/USDT:USDT"},
        )
        assert response.status_code == 422


class TestErrorHandling:
    """エラーハンドリングのテストクラス"""

    def test_repository_error_handling(
        self,
        test_client: TestClient,
        mock_repository: Mock,
    ) -> None:
        """
        異常系: リポジトリ層でエラーが発生した場合
        """
        # モックの設定
        mock_repository.get_long_short_ratio_data.side_effect = Exception(
            "Database error"
        )

        # APIリクエスト
        response = test_client.get(
            "/api/long-short-ratio/",
            params={"symbol": "BTC/USDT:USDT", "period": "1h"},
        )

        # アサーション（HTTPExceptionでハンドリングされる）
        assert response.status_code == 500
        data = response.json()
        assert "Database error" in data.get("detail", "")


class TestPeriodsValidation:
    """期間パラメータのバリデーションテスト"""

    @pytest.mark.parametrize(
        "period",
        ["5min", "1h", "1d"],
    )
    def test_valid_periods(
        self,
        test_client: TestClient,
        mock_repository: Mock,
        period: str,
    ) -> None:
        """
        正常系: 有効な期間パラメータ
        """
        mock_repository.get_long_short_ratio_data.return_value = []

        response = test_client.get(
            "/api/long-short-ratio/",
            params={"symbol": "BTC/USDT:USDT", "period": period},
        )
        assert response.status_code == 200

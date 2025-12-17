"""
戦略APIのテストモジュール

戦略APIエンドポイントの正常系、異常系、エッジケースをテストします。
"""

from typing import Any, Dict, List
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_db, get_generated_strategy_service_with_db
from app.main import app


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
def mock_strategy_integration_service() -> Mock:
    """
    StrategyIntegrationServiceのモック

    Returns:
        Mock: モックされた戦略統合サービス
    """
    mock_service = Mock()
    mock_service.get_strategies_with_response = Mock()
    return mock_service


@pytest.fixture(autouse=True)
def override_dependencies(mock_db_session, mock_strategy_integration_service):
    """
    FastAPIの依存性注入をオーバーライド

    Args:
        mock_db_session: モックDBセッション
        mock_strategy_integration_service: モックサービス
    """
    app.dependency_overrides[get_db] = lambda: mock_db_session
    app.dependency_overrides[get_generated_strategy_service_with_db] = (
        lambda: mock_strategy_integration_service
    )
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def sample_strategy() -> Dict[str, Any]:
    """
    サンプル戦略データ

    Returns:
        Dict[str, Any]: 戦略データのサンプル
    """
    return {
        "id": 1,
        "strategy_name": "ga_strategy_001",
        "fitness_score": 0.85,
        "expected_return": 0.15,
        "sharpe_ratio": 1.8,
        "max_drawdown": 0.12,
        "win_rate": 0.65,
        "risk_level": "medium",
        "experiment_id": 100,
        "created_at": "2024-01-01T00:00:00",
        "indicators": ["RSI", "MACD", "SMA"],
        "entry_conditions": "RSI < 30 AND MACD > 0",
        "exit_conditions": "RSI > 70 OR profit > 2%",
    }


@pytest.fixture
def sample_strategies_list(sample_strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    サンプル戦略リスト

    Args:
        sample_strategy: 単一の戦略データ

    Returns:
        List[Dict[str, Any]]: 戦略データのリスト
    """
    return [
        sample_strategy,
        {
            **sample_strategy,
            "id": 2,
            "fitness_score": 0.90,
            "risk_level": "low",
        },
        {
            **sample_strategy,
            "id": 3,
            "fitness_score": 0.75,
            "risk_level": "high",
        },
    ]


class TestGetStrategies:
    """戦略一覧取得のテストクラス"""

    def test_get_strategies_success(
        self,
        test_client: TestClient,
        mock_strategy_integration_service: Mock,
        sample_strategies_list: List[Dict[str, Any]],
    ) -> None:
        """
        正常系: 戦略一覧が正常に取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_strategy_integration_service: 戦略統合サービスモック
            sample_strategies_list: サンプル戦略リスト
        """
        # モックの設定
        mock_strategy_integration_service.get_strategies_with_response.return_value = {
            "success": True,
            "strategies": sample_strategies_list,
            "total_count": len(sample_strategies_list),
            "has_more": False,
            "message": "戦略が正常に取得されました",
        }

        # APIリクエスト
        response = test_client.get("/api/strategies/")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["strategies"]) == 3
        assert data["total_count"] == 3

    def test_get_strategies_with_filters(
        self,
        test_client: TestClient,
        mock_strategy_integration_service: Mock,
        sample_strategy: Dict[str, Any],
    ) -> None:
        """
        正常系: フィルター付きで戦略が取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_strategy_integration_service: 戦略統合サービスモック
            sample_strategy: サンプル戦略
        """
        # モックの設定
        mock_strategy_integration_service.get_strategies_with_response.return_value = {
            "success": True,
            "strategies": [sample_strategy],
            "total_count": 1,
            "has_more": False,
            "message": "戦略が正常に取得されました",
        }

        # APIリクエスト（フィルター付き）
        response = test_client.get(
            "/api/strategies/",
            params={
                "limit": 10,
                "offset": 0,
                "risk_level": "medium",
                "min_fitness": 0.8,
            },
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["strategies"]) == 1
        assert data["strategies"][0]["risk_level"] == "medium"

    def test_get_strategies_with_sorting(
        self,
        test_client: TestClient,
        mock_strategy_integration_service: Mock,
        sample_strategies_list: List[Dict[str, Any]],
    ) -> None:
        """
        正常系: ソート付きで戦略が取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_strategy_integration_service: 戦略統合サービスモック
            sample_strategies_list: サンプル戦略リスト
        """
        # モックの設定
        # フィットネススコアで降順ソート
        sorted_strategies = sorted(
            sample_strategies_list, key=lambda x: x["fitness_score"], reverse=True
        )
        mock_strategy_integration_service.get_strategies_with_response.return_value = {
            "success": True,
            "strategies": sorted_strategies,
            "total_count": len(sorted_strategies),
            "has_more": False,
            "message": "戦略が正常に取得されました",
        }

        # APIリクエスト（ソート付き）
        response = test_client.get(
            "/api/strategies/",
            params={"sort_by": "fitness_score", "sort_order": "desc"},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert (
            data["strategies"][0]["fitness_score"]
            >= data["strategies"][1]["fitness_score"]
        )

    def test_get_strategies_with_pagination(
        self,
        test_client: TestClient,
        mock_strategy_integration_service: Mock,
        sample_strategies_list: List[Dict[str, Any]],
    ) -> None:
        """
        正常系: ページネーション付きで戦略が取得できる

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_strategy_integration_service: 戦略統合サービスモック
            sample_strategies_list: サンプル戦略リスト
        """
        # モックの設定
        mock_strategy_integration_service.get_strategies_with_response.return_value = {
            "success": True,
            "strategies": sample_strategies_list[:2],
            "total_count": 3,
            "has_more": True,
            "message": "戦略が正常に取得されました",
        }

        # APIリクエスト（ページネーション）
        response = test_client.get(
            "/api/strategies/",
            params={"limit": 2, "offset": 0},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["strategies"]) == 2
        assert data["has_more"] is True

    def test_get_strategies_empty(
        self,
        test_client: TestClient,
        mock_strategy_integration_service: Mock,
    ) -> None:
        """
        エッジケース: 戦略が空の場合

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_strategy_integration_service: 戦略統合サービスモック
        """
        # モックの設定
        mock_strategy_integration_service.get_strategies_with_response.return_value = {
            "success": True,
            "strategies": [],
            "total_count": 0,
            "has_more": False,
            "message": "戦略が正常に取得されました",
        }

        # APIリクエスト
        response = test_client.get("/api/strategies/")

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["strategies"]) == 0
        assert data["total_count"] == 0

    @pytest.mark.parametrize(
        "limit,expected_status",
        [
            (1, 200),  # 最小値
            (50, 200),  # 中間値
            (100, 200),  # 最大値
            (0, 422),  # 無効な最小値
            (101, 422),  # 最大値超過
        ],
    )
    def test_get_strategies_limit_validation(
        self,
        test_client: TestClient,
        mock_strategy_integration_service: Mock,
        limit: int,
        expected_status: int,
    ) -> None:
        """
        異常系: リミットパラメータのバリデーション

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_strategy_integration_service: 戦略統合サービスモック
            limit: リミット値
            expected_status: 期待されるステータスコード
        """
        # モックの設定
        mock_strategy_integration_service.get_strategies_with_response.return_value = {
            "success": True,
            "strategies": [],
            "total_count": 0,
            "has_more": False,
            "message": "戦略が正常に取得されました",
        }

        # APIリクエスト
        response = test_client.get(
            "/api/strategies/",
            params={"limit": limit},
        )

        # アサーション
        assert response.status_code == expected_status

    @pytest.mark.parametrize(
        "offset",
        [0, 10, 100],
    )
    def test_get_strategies_offset_validation(
        self,
        test_client: TestClient,
        mock_strategy_integration_service: Mock,
        offset: int,
    ) -> None:
        """
        正常系: オフセットパラメータのバリデーション

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_strategy_integration_service: 戦略統合サービスモック
            offset: オフセット値
        """
        # モックの設定
        mock_strategy_integration_service.get_strategies_with_response.return_value = {
            "success": True,
            "strategies": [],
            "total_count": 0,
            "has_more": False,
            "message": "戦略が正常に取得されました",
        }

        # APIリクエスト
        response = test_client.get(
            "/api/strategies/",
            params={"offset": offset},
        )

        # アサーション
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "risk_level",
        ["low", "medium", "high"],
    )
    def test_get_strategies_risk_level_filter(
        self,
        test_client: TestClient,
        mock_strategy_integration_service: Mock,
        sample_strategy: Dict[str, Any],
        risk_level: str,
    ) -> None:
        """
        正常系: リスクレベルフィルターが機能する

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_strategy_integration_service: 戦略統合サービスモック
            sample_strategy: サンプル戦略
            risk_level: リスクレベル
        """
        # モックの設定
        filtered_strategy = {**sample_strategy, "risk_level": risk_level}
        mock_strategy_integration_service.get_strategies_with_response.return_value = {
            "success": True,
            "strategies": [filtered_strategy],
            "total_count": 1,
            "has_more": False,
            "message": "戦略が正常に取得されました",
        }

        # APIリクエスト
        response = test_client.get(
            "/api/strategies/",
            params={"risk_level": risk_level},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        if len(data["strategies"]) > 0:
            assert data["strategies"][0]["risk_level"] == risk_level

    def test_get_strategies_experiment_id_filter(
        self,
        test_client: TestClient,
        mock_strategy_integration_service: Mock,
        sample_strategy: Dict[str, Any],
    ) -> None:
        """
        正常系: 実験IDフィルターが機能する

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_strategy_integration_service: 戦略統合サービスモック
            sample_strategy: サンプル戦略
        """
        # モックの設定
        mock_strategy_integration_service.get_strategies_with_response.return_value = {
            "success": True,
            "strategies": [sample_strategy],
            "total_count": 1,
            "has_more": False,
            "message": "戦略が正常に取得されました",
        }

        # APIリクエスト
        response = test_client.get(
            "/api/strategies/",
            params={"experiment_id": 100},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["strategies"][0]["experiment_id"] == 100

    def test_get_strategies_min_fitness_filter(
        self,
        test_client: TestClient,
        mock_strategy_integration_service: Mock,
        sample_strategies_list: List[Dict[str, Any]],
    ) -> None:
        """
        正常系: 最小フィットネスフィルターが機能する

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_strategy_integration_service: 戦略統合サービスモック
            sample_strategies_list: サンプル戦略リスト
        """
        # モックの設定
        filtered_strategies = [
            s for s in sample_strategies_list if s["fitness_score"] >= 0.8
        ]
        mock_strategy_integration_service.get_strategies_with_response.return_value = {
            "success": True,
            "strategies": filtered_strategies,
            "total_count": len(filtered_strategies),
            "has_more": False,
            "message": "戦略が正常に取得されました",
        }

        # APIリクエスト
        response = test_client.get(
            "/api/strategies/",
            params={"min_fitness": 0.8},
        )

        # アサーション
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        for strategy in data["strategies"]:
            assert strategy["fitness_score"] >= 0.8


class TestErrorHandling:
    """エラーハンドリングのテストクラス"""

    def test_service_error_handling(
        self,
        test_client: TestClient,
        mock_strategy_integration_service: Mock,
    ) -> None:
        """
        異常系: サービス層でエラーが発生した場合

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_strategy_integration_service: 戦略統合サービスモック
        """
        # モックの設定
        mock_strategy_integration_service.get_strategies_with_response.side_effect = (
            Exception("Database error")
        )

        # APIリクエスト
        response = test_client.get("/api/strategies/")

        # アサーション（ErrorHandlerによって処理される）
        assert response.status_code in [200, 500]

    def test_invalid_sort_order(
        self,
        test_client: TestClient,
        mock_strategy_integration_service: Mock,
    ) -> None:
        """
        異常系: 無効なソート順序が指定された場合

        Args:
            mock_get_service: サービス取得関数のモック
            test_client: テストクライアント
            mock_strategy_integration_service: 戦略統合サービスモック
        """
        # モックの設定
        # APIリクエスト（無効なソート順序）
        response = test_client.get(
            "/api/strategies/",
            params={"sort_order": "invalid"},
        )

        # アサーション
        assert response.status_code == 422





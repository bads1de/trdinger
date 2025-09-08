"""
AutoStrategy API エンドポイントテスト

TDD approach: テストから実装を作成する。
"""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from httpx import AsyncClient

from app.api.auto_strategy import router as auto_strategy_router
from app.services.auto_strategy import AutoStrategyService
from app.api.dependencies import get_auto_strategy_service


@pytest.fixture
def app():
    """テスト用FastAPIアプリ設定"""
    app = FastAPI()
    app.include_router(auto_strategy_router)
    return app


@pytest.fixture
async def client(app):
    """テスト用HTTPクライアント"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_auto_strategy_service():
    """AutoStrategyServiceのモック"""
    mock_service = MagicMock(spec=AutoStrategyService)
    return mock_service


class TestExperimentResultsAPI:
    """実験結果取得APIテスト"""

    @patch('app.api.dependencies.get_auto_strategy_service')
    async def test_get_experiment_results_success(self, mock_get_service, client, mock_auto_strategy_service):
        """正常系: 実験結果取得成功"""
        # Arrange
        mock_auto_strategy_service.get_experiment_results.return_value = {
            "experiment_id": "test-exp-001",
            "status": "completed",
            "results": {
                "total_return": 15.5,
                "sharpe_ratio": 1.8,
                "max_drawdown": -8.2,
                "total_trades": 125
            }
        }
        mock_get_service.return_value = mock_auto_strategy_service

        # Act
        response = await client.get("/api/auto-strategy/experiments/test-exp-001/results")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["experiment_id"] == "test-exp-001"
        assert "results" in data["data"]


class TestExperimentStatusAPI:
    """実験ステータス取得APIテスト"""

    @patch('app.api.dependencies.get_auto_strategy_service')
    async def test_get_experiment_status_success(self, mock_get_service, client, mock_auto_strategy_service):
        """正常系: 実験ステータス取得成功"""
        # Arrange
        mock_auto_strategy_service.get_experiment_status.return_value = {
            "experiment_id": "test-exp-001",
            "status": "running",
            "progress": 65.0,
            "current_generation": 13,
            "eta_minutes": 12
        }
        mock_get_service.return_value = mock_auto_strategy_service

        # Act
        response = client.get("/api/auto-strategy/experiments/test-exp-001/status")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["experiment_id"] == "test-exp-001"
        assert data["data"]["status"] == "running"
        assert data["data"]["progress"] == 65.0


class TestExperimentResultsErrorHandling:
    """実験結果取得API エラーハンドリングテスト"""

    @patch('app.api.dependencies.get_auto_strategy_service')
    async def test_get_experiment_results_not_found(self, mock_get_service, client, mock_auto_strategy_service):
        """異常系: 実験結果が見つからない"""
        # Arrange
        mock_auto_strategy_service.get_experiment_results.side_effect = ValueError("Experiment not found")
        mock_get_service.return_value = mock_auto_strategy_service

        # Act
        response = client.get("/api/auto-strategy/experiments/nonexistent-exp/results")

        # Assert
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "not found" in data["message"].lower()


class TestExperimentStatusErrorHandling:
    """実験ステータス取得API エラーハンドリングテスト"""

    @patch('app.api.dependencies.get_auto_strategy_service')
    async def test_get_experiment_status_experiment_incomplete(self, mock_get_service, client, mock_auto_strategy_service):
        """異常系: 実験が未完了の場合"""
        # Arrange
        mock_auto_strategy_service.get_experiment_status.return_value = {
            "experiment_id": "test-exp-001",
            "status": "running",
            "progress": 45.0,
            "message": "Experiment is still running"
        }
        mock_get_service.return_value = mock_auto_strategy_service

        # Act
        response = client.get("/api/auto-strategy/experiments/test-exp-001/results")

        # Assert
        assert response.status_code == 202  # Accepted, processing
        data = response.json()
        assert data["success"] is True
        assert "running" in data["message"].lower()


# 統合テスト用マーカー
pytestmark = pytest.mark.asyncio
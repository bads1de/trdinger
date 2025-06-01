"""
バックテストAPIエンドポイントのテスト
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from app.main import app
from app.api.backtest import router
from app.core.services.backtest_service import BacktestService


@pytest.mark.skip(
    reason="TestClient compatibility issue with current Starlette version"
)
class TestBacktestAPI:
    """バックテストAPIのテスト"""

    @pytest.fixture
    def client(self):
        """テストクライアント"""
        return TestClient(app=app)

    @pytest.fixture
    def mock_backtest_service(self):
        """モックバックテストサービス"""
        return Mock(spec=BacktestService)

    @pytest.fixture
    def sample_backtest_request(self):
        """サンプルバックテストリクエスト"""
        return {
            "strategy_name": "SMA_CROSS",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-12-31T23:59:59Z",
            "initial_capital": 100000.0,
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

    @pytest.fixture
    def sample_backtest_result(self):
        """サンプルバックテスト結果"""
        return {
            "id": 1,
            "strategy_name": "SMA_CROSS",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "initial_capital": 100000.0,
            "performance_metrics": {
                "total_return": 25.5,
                "sharpe_ratio": 1.2,
                "max_drawdown": -15.3,
                "win_rate": 65.0,
                "total_trades": 45,
            },
            "equity_curve": [
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "equity": 100000,
                    "drawdown_pct": 0.0,
                },
                {
                    "timestamp": "2024-01-02T00:00:00Z",
                    "equity": 101000,
                    "drawdown_pct": -0.01,
                },
            ],
            "trade_history": [
                {
                    "size": 1.0,
                    "entry_price": 50000,
                    "exit_price": 51000,
                    "pnl": 1000,
                    "return_pct": 0.02,
                }
            ],
            "created_at": "2024-01-01T00:00:00Z",
        }

    @patch("app.api.backtest.BacktestService")
    def test_run_backtest_success(
        self,
        mock_service_class,
        client,
        sample_backtest_request,
        sample_backtest_result,
    ):
        """バックテスト実行成功テスト"""
        # モックの設定
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.run_backtest.return_value = sample_backtest_result

        # APIリクエスト実行
        response = client.post("/api/backtest/run", json=sample_backtest_request)

        # レスポンス検証
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "result" in data
        assert data["result"]["strategy_name"] == "SMA_CROSS"
        assert data["result"]["symbol"] == "BTC/USDT"
        assert data["result"]["performance_metrics"]["total_return"] == 25.5

        # サービスが正しく呼ばれたことを確認
        mock_service.run_backtest.assert_called_once()

    @patch("app.api.backtest.BacktestService")
    def test_run_backtest_validation_error(self, mock_service_class, client):
        """バックテスト実行バリデーションエラーテスト"""
        # 無効なリクエスト（必須フィールドが不足）
        invalid_request = {
            "strategy_name": "SMA_CROSS",
            # symbol が不足
            "timeframe": "1h",
        }

        # APIリクエスト実行
        response = client.post("/api/backtest/run", json=invalid_request)

        # バリデーションエラーの確認
        assert response.status_code == 422  # Unprocessable Entity

    @patch("app.api.backtest.BacktestService")
    def test_run_backtest_service_error(
        self, mock_service_class, client, sample_backtest_request
    ):
        """バックテスト実行サービスエラーテスト"""
        # モックの設定（例外を発生させる）
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.run_backtest.side_effect = ValueError("No data found")

        # APIリクエスト実行
        response = client.post("/api/backtest/run", json=sample_backtest_request)

        # エラーレスポンスの確認
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert "No data found" in data["error"]

    def test_get_backtest_results_success(self, client):
        """バックテスト結果一覧取得成功テスト"""
        with (
            patch("app.api.backtest.get_db") as mock_get_db,
            patch("app.api.backtest.BacktestResultRepository") as mock_repo_class,
        ):

            # モックの設定
            mock_db = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_db

            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            # サンプル結果
            sample_results = [
                {
                    "id": 1,
                    "strategy_name": "SMA_CROSS",
                    "symbol": "BTC/USDT",
                    "created_at": "2024-01-01T00:00:00Z",
                },
                {
                    "id": 2,
                    "strategy_name": "SMA_CROSS",
                    "symbol": "BTC/USDT",
                    "created_at": "2024-01-02T00:00:00Z",
                },
            ]
            mock_repo.get_backtest_results.return_value = sample_results

            # APIリクエスト実行
            response = client.get("/api/backtest/results")

            # レスポンス検証
            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "results" in data
            assert len(data["results"]) == 2
            assert data["results"][0]["id"] == 1

    def test_get_backtest_results_with_filters(self, client):
        """フィルター付きバックテスト結果取得テスト"""
        with (
            patch("app.api.backtest.get_db") as mock_get_db,
            patch("app.api.backtest.BacktestResultRepository") as mock_repo_class,
        ):

            # モックの設定
            mock_db = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_db

            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_backtest_results.return_value = []

            # フィルター付きAPIリクエスト実行
            response = client.get(
                "/api/backtest/results?symbol=BTC/USDT&limit=10&offset=0"
            )

            # レスポンス検証
            assert response.status_code == 200

            # リポジトリが正しいパラメータで呼ばれたことを確認
            mock_repo.get_backtest_results.assert_called_once_with(
                limit=10, offset=0, symbol="BTC/USDT", strategy_name=None
            )

    def test_get_backtest_result_by_id_success(self, client):
        """ID指定バックテスト結果取得成功テスト"""
        with (
            patch("app.api.backtest.get_db") as mock_get_db,
            patch("app.api.backtest.BacktestResultRepository") as mock_repo_class,
        ):

            # モックの設定
            mock_db = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_db

            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            sample_result = {
                "id": 1,
                "strategy_name": "SMA_CROSS",
                "symbol": "BTC/USDT",
                "performance_metrics": {"total_return": 25.5},
            }
            mock_repo.get_backtest_result_by_id.return_value = sample_result

            # APIリクエスト実行
            response = client.get("/api/backtest/results/1")

            # レスポンス検証
            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert data["result"]["id"] == 1
            assert data["result"]["strategy_name"] == "SMA_CROSS"

    def test_get_backtest_result_by_id_not_found(self, client):
        """ID指定バックテスト結果が見つからない場合のテスト"""
        with (
            patch("app.api.backtest.get_db") as mock_get_db,
            patch("app.api.backtest.BacktestResultRepository") as mock_repo_class,
        ):

            # モックの設定
            mock_db = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_db

            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_backtest_result_by_id.return_value = None

            # APIリクエスト実行
            response = client.get("/api/backtest/results/999")

            # 404エラーの確認
            assert response.status_code == 404
            data = response.json()
            assert data["success"] is False
            assert "not found" in data["error"].lower()

    def test_delete_backtest_result_success(self, client):
        """バックテスト結果削除成功テスト"""
        with (
            patch("app.api.backtest.get_db") as mock_get_db,
            patch("app.api.backtest.BacktestResultRepository") as mock_repo_class,
        ):

            # モックの設定
            mock_db = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_db

            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_repo.delete_backtest_result.return_value = True

            # APIリクエスト実行
            response = client.delete("/api/backtest/results/1")

            # レスポンス検証
            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "deleted" in data["message"].lower()

    def test_delete_backtest_result_not_found(self, client):
        """存在しないバックテスト結果削除テスト"""
        with (
            patch("app.api.backtest.get_db") as mock_get_db,
            patch("app.api.backtest.BacktestResultRepository") as mock_repo_class,
        ):

            # モックの設定
            mock_db = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_db

            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_repo.delete_backtest_result.return_value = False

            # APIリクエスト実行
            response = client.delete("/api/backtest/results/999")

            # 404エラーの確認
            assert response.status_code == 404
            data = response.json()
            assert data["success"] is False

    def test_get_supported_strategies(self, client):
        """サポート戦略一覧取得テスト"""
        with patch("app.api.backtest.BacktestService") as mock_service_class:
            # モックの設定
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            supported_strategies = {
                "SMA_CROSS": {
                    "name": "SMA Cross Strategy",
                    "description": "Simple Moving Average Crossover Strategy",
                    "parameters": {
                        "n1": {"type": "int", "default": 20},
                        "n2": {"type": "int", "default": 50},
                    },
                }
            }
            mock_service.get_supported_strategies.return_value = supported_strategies

            # APIリクエスト実行
            response = client.get("/api/backtest/strategies")

            # レスポンス検証
            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "strategies" in data
            assert "SMA_CROSS" in data["strategies"]
            assert data["strategies"]["SMA_CROSS"]["name"] == "SMA Cross Strategy"

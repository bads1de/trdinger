#!/usr/bin/env python3
"""
バックテストAPIの動作テスト（統合テスト）

実際のAPIサーバーを起動してエンドポイントをテストします。
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.core.services.backtest_service import BacktestService
from app.core.services.enhanced_backtest_service import EnhancedBacktestService
from database.connection import get_db
from database.repositories.backtest_result_repository import BacktestResultRepository


class TestBacktestAPI:
    """バックテストAPIの動作テスト"""

    @pytest.fixture
    def client(self):
        """テストクライアント"""
        return TestClient(app=app)

    @pytest.fixture
    def mock_backtest_service(self):
        """BacktestServiceのモック"""
        with patch("app.api.backtest.BacktestService") as mock_service:
            yield mock_service.return_value

    @pytest.fixture
    def mock_enhanced_backtest_service(self):
        """EnhancedBacktestServiceのモック"""
        with patch("app.api.backtest.EnhancedBacktestService") as mock_service:
            yield mock_service.return_value

    @pytest.fixture
    def mock_backtest_result_repository(self):
        """BacktestResultRepositoryのモック"""
        with patch("app.api.backtest.BacktestResultRepository") as mock_repo:
            yield mock_repo.return_value

    def test_strategies_endpoint(self, client, mock_backtest_service):
        """戦略一覧エンドポイントのテスト"""
        mock_backtest_service.get_available_strategies.return_value = [
            {"name": "SMA_CROSS", "description": "SMA Cross Strategy"},
            {"name": "RSI_STRATEGY", "description": "RSI Strategy"},
        ]
        response = client.get("/api/backtest/strategies")
        assert response.status_code == 200
        assert response.json() == {
            "success": True,
            "data": [
                {"name": "SMA_CROSS", "description": "SMA Cross Strategy"},
                {"name": "RSI_STRATEGY", "description": "RSI Strategy"},
            ],
        }

    def test_health_endpoint(self, client):
        """ヘルスチェックエンドポイントのテスト"""
        response = client.get("/api/backtest/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_backtest_run(
        self, client, mock_backtest_service, mock_backtest_result_repository
    ):
        """バックテスト実行テスト"""
        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-31T23:59:59Z",
            "initial_capital": 100000.0,
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }
        mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {
                "total_return": 10.5,
                "sharpe_ratio": 1.2,
                "max_drawdown": -5.0,
                "win_rate": 0.6,
                "total_trades": 100,
            },
            "equity_curve": [],
            "trade_history": [],
        }
        mock_backtest_result_repository.save_backtest_result.return_value = {
            "id": "test_id",
            "strategy_name": "SMA_CROSS",
            "symbol": "BTC/USDT",
            "initial_capital": 100000.0,
            "performance_metrics": {
                "total_return": 10.5,
                "sharpe_ratio": 1.2,
                "max_drawdown": -5.0,
                "win_rate": 0.6,
                "total_trades": 100,
            },
        }

        response = client.post("/api/backtest/run", json=config)
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert response.json()["result"]["performance_metrics"]["total_return"] == 10.5

    def test_results_endpoint(self, client, mock_backtest_result_repository):
        """バックテスト結果一覧取得テスト"""
        mock_backtest_result_repository.get_backtest_results.return_value = (
            [
                {
                    "id": "result1",
                    "strategy_name": "SMA_CROSS",
                    "symbol": "BTC/USDT",
                    "created_at": datetime.now().isoformat(),
                }
            ],
            1,
        )
        response = client.get("/api/backtest/results?limit=1")
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert len(response.json()["results"]) == 1
        assert response.json()["total"] == 1

    @pytest.fixture
    def sample_enhanced_optimization_request(self):
        """サンプル拡張最適化リクエスト"""
        return {
            "base_config": {
                "strategy_name": "SMA_CROSS_ENHANCED",
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-12-31T23:59:59Z",
                "initial_capital": 100000.0,
                "commission_rate": 0.001,
                "strategy_config": {
                    "strategy_type": "SMA_CROSS",
                    "parameters": {"n1": 20, "n2": 50},
                },
            },
            "optimization_params": {
                "method": "sambo",
                "max_tries": 50,
                "maximize": "Sharpe Ratio",
                "return_heatmap": True,
                "return_optimization": True,
                "random_state": 42,
                "constraint": "sma_cross",
                "parameters": {"n1": [10, 15, 20, 25], "n2": [30, 40, 50, 60]},
            },
        }

    @pytest.fixture
    def sample_multi_objective_request(self):
        """サンプルマルチ目的最適化リクエスト"""
        return {
            "base_config": {
                "strategy_name": "SMA_CROSS_MULTI",
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-12-31T23:59:59Z",
                "initial_capital": 100000.0,
                "commission_rate": 0.001,
                "strategy_config": {"strategy_type": "SMA_CROSS", "parameters": {}},
            },
            "objectives": ["Sharpe Ratio", "Return [%]", "-Max. Drawdown [%]"],
            "weights": [0.4, 0.4, 0.2],
            "optimization_params": {
                "method": "sambo",
                "max_tries": 30,
                "parameters": {"n1": [10, 15, 20], "n2": [30, 40, 50]},
            },
        }

    @pytest.fixture
    def sample_robustness_test_request(self):
        """サンプルロバストネステストリクエスト"""
        return {
            "base_config": {
                "strategy_name": "SMA_CROSS_ROBUST",
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "initial_capital": 100000.0,
                "commission_rate": 0.001,
                "strategy_config": {"strategy_type": "SMA_CROSS", "parameters": {}},
            },
            "test_periods": [
                ["2024-01-01", "2024-06-30"],
                ["2024-07-01", "2024-12-31"],
            ],
            "optimization_params": {
                "method": "grid",
                "maximize": "Sharpe Ratio",
                "parameters": {"n1": [10, 20], "n2": [30, 50]},
            },
        }

    @pytest.fixture
    def sample_enhanced_optimization_result(self):
        """サンプル拡張最適化結果"""
        return {
            "strategy_name": "SMA_CROSS_ENHANCED",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "initial_capital": 100000,
            "performance_metrics": {
                "total_return": 28.5,
                "sharpe_ratio": 1.9,
                "max_drawdown": -7.8,
                "win_rate": 68.0,
                "profit_factor": 1.6,
                "total_trades": 52,
            },
            "optimized_parameters": {"n1": 15, "n2": 45},
            "heatmap_summary": {
                "best_combination": (15, 45),
                "best_value": 1.9,
                "total_combinations": 16,
            },
            "optimization_details": {
                "method": "sambo",
                "n_calls": 50,
                "best_value": 1.9,
            },
            "optimization_metadata": {
                "method": "sambo",
                "maximize": "Sharpe Ratio",
                "parameter_space_size": 16,
            },
        }

    @patch("app.api.backtest.EnhancedBacktestService")
    @patch("app.api.backtest.get_db")
    @patch("app.api.backtest.BacktestResultRepository")
    def test_optimize_strategy_enhanced_success(
        self,
        mock_repo_class,
        mock_get_db,
        mock_service_class,
        client,
        sample_enhanced_optimization_request,
        sample_enhanced_optimization_result,
    ):
        """拡張最適化成功テスト"""
        # モックの設定
        mock_db = Mock()
        mock_get_db.return_value.__enter__.return_value = mock_db

        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.optimize_strategy_enhanced.return_value = (
            sample_enhanced_optimization_result
        )

        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.save_backtest_result.return_value = {
            "id": 123,
            **sample_enhanced_optimization_result,
        }

        # APIリクエスト実行
        response = client.post(
            "/api/backtest/optimize-enhanced", json=sample_enhanced_optimization_request
        )

        # レスポンス検証
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "result" in data
        assert data["result"]["strategy_name"] == "SMA_CROSS_ENHANCED"
        assert "optimized_parameters" in data["result"]
        assert "heatmap_summary" in data["result"]
        assert "optimization_details" in data["result"]

    @patch("app.api.backtest.EnhancedBacktestService")
    @patch("app.api.backtest.get_db")
    @patch("app.api.backtest.BacktestResultRepository")
    def test_multi_objective_optimization_success(
        self,
        mock_repo_class,
        mock_get_db,
        mock_service_class,
        client,
        sample_multi_objective_request,
    ):
        """マルチ目的最適化成功テスト"""
        # モックの設定
        mock_db = Mock()
        mock_get_db.return_value.__enter__.return_value = mock_db

        mock_service = Mock()
        mock_service_class.return_value = mock_service

        mock_result = {
            "strategy_name": "SMA_CROSS_MULTI",
            "performance_metrics": {
                "total_return": 25.0,
                "sharpe_ratio": 1.7,
                "max_drawdown": -9.2,
            },
            "multi_objective_details": {
                "objectives": ["Sharpe Ratio", "Return [%]", "-Max. Drawdown [%]"],
                "weights": [0.4, 0.4, 0.2],
                "individual_scores": {
                    "Sharpe Ratio": 1.7,
                    "Return [%]": 25.0,
                    "-Max. Drawdown [%]": 9.2,
                },
            },
        }
        mock_service.multi_objective_optimization.return_value = mock_result

        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.save_backtest_result.return_value = {"id": 124, **mock_result}

        # APIリクエスト実行
        response = client.post(
            "/api/backtest/multi-objective-optimization",
            json=sample_multi_objective_request,
        )

        # レスポンス検証
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "result" in data
        assert "multi_objective_details" in data["result"]
        assert data["result"]["multi_objective_details"]["objectives"] == [
            "Sharpe Ratio",
            "Return [%]",
            "-Max. Drawdown [%]",
        ]

    @patch("app.api.backtest.EnhancedBacktestService")
    @patch("app.api.backtest.get_db")
    @patch("app.api.backtest.BacktestResultRepository")
    def test_robustness_test_success(
        self,
        mock_repo_class,
        mock_get_db,
        mock_service_class,
        client,
        sample_robustness_test_request,
    ):
        """ロバストネステスト成功テスト"""
        # モックの設定
        mock_db = Mock()
        mock_get_db.return_value.__enter__.return_value = mock_db

        mock_service = Mock()
        mock_service_class.return_value = mock_service

        mock_result = {
            "individual_results": {
                "period_1": {
                    "strategy_name": "SMA_CROSS_ROBUST",
                    "performance_metrics": {"sharpe_ratio": 1.5},
                },
                "period_2": {
                    "strategy_name": "SMA_CROSS_ROBUST",
                    "performance_metrics": {"sharpe_ratio": 1.8},
                },
            },
            "robustness_analysis": {
                "robustness_score": 0.85,
                "successful_periods": 2,
                "failed_periods": 0,
            },
            "total_periods": 2,
        }
        mock_service.robustness_test.return_value = mock_result

        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.save_backtest_result.return_value = {"id": 125}

        # APIリクエスト実行
        response = client.post(
            "/api/backtest/robustness-test", json=sample_robustness_test_request
        )

        # レスポンス検証
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "result" in data
        assert "robustness_analysis" in data["result"]
        assert data["result"]["robustness_analysis"]["robustness_score"] == 0.85
        assert data["result"]["total_periods"] == 2

    def test_enhanced_optimization_invalid_request(self, client):
        """拡張最適化無効リクエストテスト"""
        invalid_request = {
            "base_config": {
                "strategy_name": "INVALID"
                # 必要なフィールドが不足
            }
        }

        response = client.post("/api/backtest/optimize-enhanced", json=invalid_request)

        assert response.status_code == 422  # Validation error

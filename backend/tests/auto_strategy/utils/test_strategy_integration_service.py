"""
StrategyIntegrationServiceのテスト
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from backend.app.services.auto_strategy.utils.strategy_integration_service import (
    StrategyIntegrationService,
)
from backend.database.models import GeneratedStrategy, BacktestResult


class TestStrategyIntegrationService:
    """StrategyIntegrationServiceのテスト"""

    @pytest.fixture
    def mock_db_session(self):
        return Mock()

    @pytest.fixture
    def service(self, mock_db_session):
        # リポジトリをモック化するためにpatchを使用する手もあるが、
        # ここではインスタンス作成後に属性を差し替える簡易的な方法をとる
        # または __init__ でリポジトリを作成しているので、それを patch するのが安全
        with (
            patch(
                "backend.app.services.auto_strategy.utils.strategy_integration_service.GeneratedStrategyRepository"
            ) as MockGenRepo,
            patch(
                "backend.app.services.auto_strategy.utils.strategy_integration_service.BacktestResultRepository"
            ) as MockBtRepo,
        ):

            service = StrategyIntegrationService(mock_db_session)
            service.generated_strategy_repo = MockGenRepo.return_value
            service.backtest_result_repo = MockBtRepo.return_value
            yield service

    def test_get_strategies_success(self, service):
        """戦略一覧取得の成功ケース"""
        # Mock data
        mock_strategy = MagicMock(spec=GeneratedStrategy)
        mock_strategy.id = 1
        mock_strategy.experiment_id = 123
        mock_strategy.generation = 5
        mock_strategy.fitness_score = 1.5
        mock_strategy.created_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_strategy.updated_at = datetime(2024, 1, 2, 12, 0, 0)

        # gene_data
        mock_strategy.gene_data = {
            "indicators": [
                {"type": "RSI", "enabled": True},
                {"type": "SMA", "enabled": False},
            ],
            "timeframe": "1h",
            "risk_management": {},
            "entry_conditions": {},
            "exit_conditions": {},
        }

        # backtest_result
        mock_bt = MagicMock(spec=BacktestResult)
        mock_bt.performance_metrics = {
            "total_return": 0.5,
            "sharpe_ratio": 2.0,
            "max_drawdown": 0.1,  # 10% -> medium risk
            "win_rate": 0.6,
            "profit_factor": 1.5,
            "total_trades": 50,
        }
        mock_strategy.backtest_result = mock_bt

        # Configure repository mock
        service.generated_strategy_repo.get_filtered_and_sorted_strategies.return_value = (
            1,
            [mock_strategy],
        )

        # Execute
        result = service.get_strategies(limit=10, offset=0)

        # Verify
        assert result["total_count"] == 1
        assert len(result["strategies"]) == 1

        s = result["strategies"][0]
        assert s["id"] == "auto_1"
        assert s["name"] == "GA生成戦略_RSI"
        assert s["indicators"] == ["RSI"]
        assert s["risk_level"] == "medium"
        assert s["sharpe_ratio"] == 2.0
        assert s["expected_return"] == 0.5

    def test_get_strategies_empty(self, service):
        """戦略がない場合"""
        service.generated_strategy_repo.get_filtered_and_sorted_strategies.return_value = (
            0,
            [],
        )

        result = service.get_strategies()

        assert result["total_count"] == 0
        assert result["strategies"] == []
        assert result["has_more"] is False

    @patch(
        "backend.app.utils.response.api_response"
    )  # api_response dependency might be tricky if not available, mocking it
    def test_get_strategies_with_response(self, mock_api_response, service):
        """レスポンス形式での取得"""
        # Mocking api_response to just return the data dict for assertion
        mock_api_response.side_effect = lambda success, data, message: {
            "success": success,
            "data": data,
        }

        # Mock get_strategies behavior via repo
        service.generated_strategy_repo.get_filtered_and_sorted_strategies.return_value = (
            0,
            [],
        )

        response = service.get_strategies_with_response(limit=5)

        assert response["success"] is True
        assert response["data"]["strategies"] == []

    def test_calculate_risk_level(self, service):
        """リスクレベル計算ロジック"""
        # Low risk
        assert service._calculate_risk_level({"max_drawdown": 0.04}) == "low"
        assert service._calculate_risk_level({"max_drawdown": 0.05}) == "low"

        # Medium risk
        assert service._calculate_risk_level({"max_drawdown": 0.06}) == "medium"
        assert service._calculate_risk_level({"max_drawdown": 0.15}) == "medium"

        # High risk
        assert service._calculate_risk_level({"max_drawdown": 0.16}) == "high"

        # Default (0.0)
        assert service._calculate_risk_level({}) == "low"

    def test_convert_strategy_handle_exceptions(self, service):
        """変換エラーのハンドリング"""
        # Strategy with invalid gene_data (None) causing exception
        mock_strategy = MagicMock(spec=GeneratedStrategy)
        mock_strategy.gene_data = None  # This might cause error in cast or access

        # Mocking directly to raise exception inside _convert... if needed,
        # or relying on None access raising TypeError/AttributeError

        # However, _convert... catches Exception and returns None.
        # Let's force it.
        # Mocking gene_data as None and _extract_strategy_name will fail if it expects dict

        # In implementation: gene_data = cast(Dict[str, Any], strategy.gene_data)
        # _extract_strategy_name(gene_data) -> gene_data.get(...) -> AttributeError if None

        result = service._convert_generated_strategy_to_display_format(mock_strategy)
        assert result is None

    def test_extract_strategy_name_dict_format(self, service):
        """辞書形式（旧形式）の指標データからの名前抽出"""
        gene_data = {
            "indicators": {
                "rsi": {"enabled": True},
                "sma": {"enabled": True},
                "ema": {"enabled": False},
            }
        }
        name = service._extract_strategy_name(gene_data)
        assert "RSI" in name
        assert "SMA" in name
        assert "EMA" not in name

    def test_extract_strategy_name_list_format(self, service):
        """リスト形式の指標データからの名前抽出"""
        gene_data = {
            "indicators": [
                {"type": "MACD", "enabled": True},
                {"type": "BB", "enabled": True},
            ]
        }
        name = service._extract_strategy_name(gene_data)
        assert "MACD" in name
        assert "BB" in name

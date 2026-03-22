"""
バックテストオーケストレーションサービスのテストモジュール

BacktestOrchestrationServiceの正常系、異常系、エッジケースをテストします。
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.backtest.orchestration.backtest_orchestration_service import (
    BacktestOrchestrationService,
)


@pytest.fixture
def mock_db_session() -> MagicMock:
    """
    データベースセッションのモック

    Returns:
        MagicMock: モックされたデータベースセッション
    """
    return MagicMock()


@pytest.fixture
def mock_backtest_result_repository() -> AsyncMock:
    """
    BacktestResultRepositoryのモック

    Returns:
        AsyncMock: モックされたリポジトリ
    """
    return AsyncMock()


@pytest.fixture
def mock_backtest_service() -> AsyncMock:
    """
    BacktestServiceのモック

    Returns:
        AsyncMock: モックされたサービス
    """
    return AsyncMock()


@pytest.fixture
def orchestration_service() -> BacktestOrchestrationService:
    """
    BacktestOrchestrationServiceのインスタンス

    Returns:
        BacktestOrchestrationService: テスト対象のサービス
    """
    return BacktestOrchestrationService()


@pytest.fixture
def sample_backtest_result() -> Dict[str, Any]:
    """
    サンプルバックテスト結果

    Returns:
        Dict[str, Any]: バックテスト結果のサンプルデータ
    """
    return {
        "id": 1,
        "strategy_name": "test_strategy",
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01T00:00:00",
        "end_date": "2024-01-31T00:00:00",
        "initial_capital": 10000.0,
        "final_capital": 11000.0,
        "total_return": 0.1,
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.05,
        "total_trades": 10,
        "winning_trades": 6,
        "losing_trades": 4,
    }


class TestServiceInitialization:
    """正常系: サービスの初期化テスト"""

    def test_service_creation(
        self, orchestration_service: BacktestOrchestrationService
    ):
        """
        正常系: サービスが正常に初期化される

        Args:
            orchestration_service: オーケストレーションサービス
        """
        assert orchestration_service is not None
        assert isinstance(orchestration_service, BacktestOrchestrationService)


class TestGetBacktestResults:
    """正常系: バックテスト結果取得のテスト"""

    @pytest.mark.asyncio
    async def test_get_backtest_results_success(
        self,
        orchestration_service: BacktestOrchestrationService,
        mock_db_session: MagicMock,
        sample_backtest_result: Dict[str, Any],
    ):
        """
        正常系: バックテスト結果一覧が正常に取得できる

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            sample_backtest_result: サンプル結果
        """
        with patch(
            "app.services.backtest.orchestration.backtest_orchestration_service.BacktestResultRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_backtest_results.return_value = [sample_backtest_result]
            mock_repo.count_backtest_results.return_value = 1
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.get_backtest_results(
                db=mock_db_session, limit=50, offset=0
            )

            assert result["success"] is True
            assert len(result["results"]) == 1
            assert result["total"] == 1
            assert result["results"][0]["id"] == 1

    @pytest.mark.asyncio
    async def test_get_backtest_results_with_filters(
        self,
        orchestration_service: BacktestOrchestrationService,
        mock_db_session: MagicMock,
        sample_backtest_result: Dict[str, Any],
    ):
        """
        正常系: フィルター付きでバックテスト結果が取得できる

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            sample_backtest_result: サンプル結果
        """
        with patch(
            "app.services.backtest.orchestration.backtest_orchestration_service.BacktestResultRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_backtest_results.return_value = [sample_backtest_result]
            mock_repo.count_backtest_results.return_value = 1
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.get_backtest_results(
                db=mock_db_session,
                limit=10,
                offset=0,
                symbol="BTC/USDT:USDT",
                strategy_name="test_strategy",
            )

            assert result["success"] is True
            assert len(result["results"]) == 1
            mock_repo.get_backtest_results.assert_called_once_with(
                limit=10,
                offset=0,
                symbol="BTC/USDT:USDT",
                strategy_name="test_strategy",
            )

    @pytest.mark.asyncio
    async def test_get_backtest_results_empty(
        self,
        orchestration_service: BacktestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        エッジケース: 結果が空の場合

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        with patch(
            "app.services.backtest.orchestration.backtest_orchestration_service.BacktestResultRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_backtest_results.return_value = []
            mock_repo.count_backtest_results.return_value = 0
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.get_backtest_results(
                db=mock_db_session, limit=50, offset=0
            )

            assert result["success"] is True
            assert len(result["results"]) == 0
            assert result["total"] == 0


class TestGetBacktestResultById:
    """正常系: ID指定でのバックテスト結果取得のテスト"""

    @pytest.mark.asyncio
    async def test_get_backtest_result_by_id_success(
        self,
        orchestration_service: BacktestOrchestrationService,
        mock_db_session: MagicMock,
        sample_backtest_result: Dict[str, Any],
    ):
        """
        正常系: ID指定でバックテスト結果が取得できる

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            sample_backtest_result: サンプル結果
        """
        with patch(
            "app.services.backtest.orchestration.backtest_orchestration_service.BacktestResultRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_backtest_result_by_id.return_value = sample_backtest_result
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.get_backtest_result_by_id(
                db=mock_db_session, result_id=1
            )

            assert result["success"] is True
            assert result["data"]["id"] == 1

    @pytest.mark.asyncio
    async def test_get_backtest_result_by_id_not_found(
        self,
        orchestration_service: BacktestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        異常系: 存在しないIDで404が返される

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        with patch(
            "app.services.backtest.orchestration.backtest_orchestration_service.BacktestResultRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_backtest_result_by_id.return_value = None
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.get_backtest_result_by_id(
                db=mock_db_session, result_id=9999
            )

            assert result["success"] is False
            assert result["status_code"] == 404


class TestDeleteBacktestResult:
    """正常系: バックテスト結果削除のテスト"""

    @pytest.mark.asyncio
    async def test_delete_backtest_result_success(
        self,
        orchestration_service: BacktestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        正常系: バックテスト結果が正常に削除できる

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        with (
            patch(
                "app.services.backtest.orchestration.backtest_orchestration_service.BacktestResultRepository"
            ) as mock_repo_class,
            patch(
                "app.services.backtest.orchestration.backtest_orchestration_service.GeneratedStrategyRepository"
            ) as mock_strategy_repo_class,
        ):
            mock_repo = MagicMock()
            mock_repo.delete_backtest_result.return_value = True
            mock_repo_class.return_value = mock_repo

            mock_strategy_repo = MagicMock()
            mock_strategy_repo_class.return_value = mock_strategy_repo

            result = await orchestration_service.delete_backtest_result(
                db=mock_db_session, result_id=1
            )

            assert result["success"] is True
            assert "削除" in result["message"]
            mock_strategy_repo.unlink_backtest_result.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_delete_backtest_result_not_found(
        self,
        orchestration_service: BacktestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        異常系: 存在しないIDの削除で404が返される

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        with (
            patch(
                "app.services.backtest.orchestration.backtest_orchestration_service.BacktestResultRepository"
            ) as mock_repo_class,
            patch(
                "app.services.backtest.orchestration.backtest_orchestration_service.GeneratedStrategyRepository"
            ) as mock_strategy_repo_class,
        ):
            mock_repo = MagicMock()
            mock_repo.delete_backtest_result.return_value = False
            mock_repo_class.return_value = mock_repo

            mock_strategy_repo = MagicMock()
            mock_strategy_repo_class.return_value = mock_strategy_repo

            result = await orchestration_service.delete_backtest_result(
                db=mock_db_session, result_id=9999
            )

            assert result["success"] is False
            assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_delete_all_backtest_results_success(
        self,
        orchestration_service: BacktestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        正常系: すべてのバックテスト結果が正常に削除できる

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        with (
            patch(
                "app.services.backtest.orchestration.backtest_orchestration_service.BacktestResultRepository"
            ) as mock_backtest_repo,
            patch(
                "app.services.backtest.orchestration.backtest_orchestration_service.GAExperimentRepository"
            ) as mock_ga_repo,
            patch(
                "app.services.backtest.orchestration.backtest_orchestration_service.GeneratedStrategyRepository"
            ) as mock_strategy_repo,
        ):
            mock_backtest = MagicMock()
            mock_backtest.delete_all_backtest_results.return_value = 10
            mock_backtest_repo.return_value = mock_backtest

            mock_ga = MagicMock()
            mock_ga.delete_all_experiments.return_value = 5
            mock_ga_repo.return_value = mock_ga

            mock_strategy = MagicMock()
            mock_strategy.delete_all_strategies.return_value = 8
            mock_strategy_repo.return_value = mock_strategy

            result = await orchestration_service.delete_all_backtest_results(
                db=mock_db_session
            )

            assert result["success"] is True
            assert result["data"]["deleted_backtest_results"] == 10
            assert result["data"]["deleted_ga_experiments"] == 5
            assert result["data"]["deleted_generated_strategies"] == 8


class TestGetSupportedStrategies:
    """正常系: サポート戦略取得のテスト"""

    @pytest.mark.asyncio
    async def test_get_supported_strategies_success(
        self, orchestration_service: BacktestOrchestrationService
    ):
        """
        正常系: サポート戦略一覧が正常に取得できる

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.backtest.orchestration.backtest_orchestration_service.BacktestService"
        ) as mock_service_class:
            mock_service = MagicMock()
            mock_service.get_supported_strategies.return_value = [
                "sma_crossover",
                "rsi_strategy",
            ]
            mock_service_class.return_value = mock_service

            result = await orchestration_service.get_supported_strategies()

            assert result["success"] is True
            assert len(result["data"]["strategies"]) == 2
            assert "sma_crossover" in result["data"]["strategies"]
            mock_service.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_supported_strategies_empty(
        self, orchestration_service: BacktestOrchestrationService
    ):
        """
        エッジケース: 戦略が空の場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.backtest.orchestration.backtest_orchestration_service.BacktestService"
        ) as mock_service_class:
            mock_service = MagicMock()
            mock_service.get_supported_strategies.return_value = []
            mock_service_class.return_value = mock_service

            result = await orchestration_service.get_supported_strategies()

            assert result["success"] is True
            assert len(result["data"]["strategies"]) == 0


class TestErrorHandling:
    """異常系: エラーハンドリングのテスト"""

    @pytest.mark.asyncio
    async def test_get_backtest_results_with_exception(
        self,
        orchestration_service: BacktestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        異常系: リポジトリでエラーが発生した場合

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        from fastapi import HTTPException

        with patch(
            "app.services.backtest.orchestration.backtest_orchestration_service.BacktestResultRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_backtest_results.side_effect = Exception("Database error")
            mock_repo_class.return_value = mock_repo

            with pytest.raises(HTTPException):
                await orchestration_service.get_backtest_results(
                    db=mock_db_session, limit=50, offset=0
                )


class TestEdgeCases:
    """境界値テスト"""

    @pytest.mark.asyncio
    async def test_get_backtest_results_with_zero_limit(
        self,
        orchestration_service: BacktestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        境界値: limit=0の場合

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        with patch(
            "app.services.backtest.orchestration.backtest_orchestration_service.BacktestResultRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_backtest_results.return_value = []
            mock_repo.count_backtest_results.return_value = 0
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.get_backtest_results(
                db=mock_db_session, limit=0, offset=0
            )

            assert result["success"] is True
            assert len(result["results"]) == 0

    @pytest.mark.asyncio
    async def test_get_backtest_results_with_large_offset(
        self,
        orchestration_service: BacktestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        境界値: 大きなoffsetの場合

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        with patch(
            "app.services.backtest.orchestration.backtest_orchestration_service.BacktestResultRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_backtest_results.return_value = []
            mock_repo.count_backtest_results.return_value = 0
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.get_backtest_results(
                db=mock_db_session, limit=50, offset=10000
            )

            assert result["success"] is True
            assert len(result["results"]) == 0





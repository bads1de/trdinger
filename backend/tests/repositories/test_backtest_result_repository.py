"""
BacktestResultRepositoryのテストモジュール

バックテスト結果リポジトリの機能をテストします。
"""

from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from database.models import BacktestResult
from database.repositories.backtest_result_repository import (
    BacktestResultRepository,
)


@pytest.fixture
def mock_session() -> MagicMock:
    """
    モックDBセッション

    Returns:
        MagicMock: モックされたデータベースセッション
    """
    session = MagicMock(spec=Session)
    session.execute = MagicMock()
    session.commit = MagicMock()
    session.rollback = MagicMock()
    session.refresh = MagicMock()
    session.add = MagicMock()
    session.delete = MagicMock()
    session.query = MagicMock()
    session.scalar = MagicMock()
    session.scalars = MagicMock()
    return session


@pytest.fixture
def repository(mock_session: MagicMock) -> BacktestResultRepository:
    """
    BacktestResultRepositoryインスタンス

    Args:
        mock_session: モックセッション

    Returns:
        BacktestResultRepository: テスト用リポジトリインスタンス
    """
    return BacktestResultRepository(mock_session)


@pytest.fixture
def sample_backtest_model() -> BacktestResult:
    """
    サンプルBacktestResultモデルインスタンス

    Returns:
        BacktestResult: テスト用バックテスト結果データ
    """
    return BacktestResult(
        id=1,
        strategy_name="test_strategy",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
        initial_capital=10000.0,
        commission_rate=0.001,
        config_json={"indicator": "RSI"},
        performance_metrics={
            "total_return": 0.15,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.10,
            "total_trades": 50,
            "win_rate": 0.60,
            "profit_factor": 1.8,
            "final_balance": 11500.0,
        },
        equity_curve=[10000, 10500, 11000, 11500],
        trade_history=[
            {"entry": 50000, "exit": 51000, "profit": 100}
        ],
        execution_time=5.5,
        status="completed",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_backtest_data() -> Dict[str, Any]:
    """
    サンプルバックテストデータ（辞書形式）

    Returns:
        Dict[str, Any]: テスト用バックテストデータ
    """
    return {
        "strategy_name": "test_strategy",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "end_date": datetime(2024, 1, 31, tzinfo=timezone.utc),
        "initial_capital": 10000.0,
        "commission_rate": 0.001,
        "config_json": {"indicator": "RSI"},
        "performance_metrics": {
            "total_return": 0.15,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.10,
        },
        "equity_curve": [10000, 10500, 11000],
        "trade_history": [{"entry": 50000, "exit": 51000}],
        "execution_time": 5.5,
        "status": "completed",
    }


class TestRepositoryInitialization:
    """リポジトリ初期化のテスト"""

    def test_repository_initialization(
        self, mock_session: MagicMock
    ) -> None:
        """リポジトリが正しく初期化される"""
        repo = BacktestResultRepository(mock_session)
        
        assert repo.db == mock_session
        assert repo.model_class == BacktestResult


class TestToDictMethod:
    """to_dictメソッドのテスト"""

    def test_to_dict_basic(
        self,
        repository: BacktestResultRepository,
        sample_backtest_model: BacktestResult,
    ) -> None:
        """モデルインスタンスが辞書に変換される"""
        result = repository.to_dict(sample_backtest_model)
        
        assert isinstance(result, dict)
        assert result["id"] == 1
        assert result["strategy_name"] == "test_strategy"
        assert result["symbol"] == "BTC/USDT"

    def test_to_dict_includes_performance_metrics(
        self,
        repository: BacktestResultRepository,
        sample_backtest_model: BacktestResult,
    ) -> None:
        """パフォーマンスメトリクスがトップレベルに含まれる"""
        result = repository.to_dict(sample_backtest_model)
        
        assert "total_return" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert result["total_return"] == 0.15


class TestSaveBacktestResult:
    """save_backtest_resultメソッドのテスト"""

    @patch("database.repositories.backtest_result_repository.BacktestResult")
    def test_save_backtest_result_success(
        self,
        mock_backtest_result_class: MagicMock,
        repository: BacktestResultRepository,
        sample_backtest_data: Dict[str, Any],
        sample_backtest_model: BacktestResult,
    ) -> None:
        """バックテスト結果が正常に保存される"""
        mock_backtest_result_class.return_value = sample_backtest_model
        sample_backtest_model.id = 1
        
        result = repository.save_backtest_result(sample_backtest_data)
        
        repository.db.add.assert_called_once()
        repository.db.commit.assert_called_once()
        repository.db.refresh.assert_called_once()
        assert result["id"] == 1

    def test_save_backtest_result_with_iso_dates(
        self,
        repository: BacktestResultRepository,
        sample_backtest_data: Dict[str, Any],
    ) -> None:
        """ISO形式の日付文字列が処理される"""
        sample_backtest_data["start_date"] = "2024-01-01T00:00:00+00:00"
        sample_backtest_data["end_date"] = "2024-01-31T00:00:00+00:00"
        
        mock_result = MagicMock(spec=BacktestResult)
        mock_result.id = 1
        
        with patch(
            "database.repositories.backtest_result_repository.BacktestResult",
            return_value=mock_result,
        ):
            result = repository.save_backtest_result(sample_backtest_data)
            assert result is not None


class TestGetBacktestResults:
    """get_backtest_resultsメソッドのテスト"""

    def test_get_backtest_results_success(
        self,
        repository: BacktestResultRepository,
        sample_backtest_model: BacktestResult,
    ) -> None:
        """バックテスト結果一覧が取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_backtest_model]
        repository.db.scalars.return_value = mock_scalars
        
        results = repository.get_backtest_results(limit=10, offset=0)
        
        assert len(results) == 1
        assert results[0]["id"] == 1

    def test_get_backtest_results_with_symbol_filter(
        self,
        repository: BacktestResultRepository,
        sample_backtest_model: BacktestResult,
    ) -> None:
        """シンボルフィルター付きで取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_backtest_model]
        repository.db.scalars.return_value = mock_scalars
        
        results = repository.get_backtest_results(
            symbol="BTC/USDT", limit=10
        )
        
        assert len(results) == 1
        assert results[0]["symbol"] == "BTC/USDT"

    def test_get_backtest_results_with_strategy_filter(
        self,
        repository: BacktestResultRepository,
        sample_backtest_model: BacktestResult,
    ) -> None:
        """戦略名フィルター付きで取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_backtest_model]
        repository.db.scalars.return_value = mock_scalars
        
        results = repository.get_backtest_results(
            strategy_name="test_strategy", limit=10
        )
        
        assert len(results) == 1
        assert results[0]["strategy_name"] == "test_strategy"

    def test_get_backtest_results_empty(
        self, repository: BacktestResultRepository
    ) -> None:
        """結果が空の場合空リストが返される"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        repository.db.scalars.return_value = mock_scalars
        
        results = repository.get_backtest_results()
        
        assert len(results) == 0


class TestGetBacktestResultById:
    """get_backtest_result_by_idメソッドのテスト"""

    def test_get_backtest_result_by_id_success(
        self,
        repository: BacktestResultRepository,
        sample_backtest_model: BacktestResult,
    ) -> None:
        """IDでバックテスト結果が取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_backtest_model]
        repository.db.scalars.return_value = mock_scalars
        
        result = repository.get_backtest_result_by_id(1)
        
        assert result is not None
        assert result["id"] == 1

    def test_get_backtest_result_by_id_not_found(
        self, repository: BacktestResultRepository
    ) -> None:
        """存在しないIDの場合Noneが返される"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        repository.db.scalars.return_value = mock_scalars
        
        result = repository.get_backtest_result_by_id(999)
        
        assert result is None


class TestDeleteBacktestResult:
    """delete_backtest_resultメソッドのテスト"""

    def test_delete_backtest_result_success(
        self,
        repository: BacktestResultRepository,
        sample_backtest_model: BacktestResult,
    ) -> None:
        """バックテスト結果が削除できる"""
        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = sample_backtest_model
        repository.db.query.return_value = mock_query
        
        result = repository.delete_backtest_result(1)
        
        assert result is True
        repository.db.delete.assert_called_once()
        repository.db.commit.assert_called_once()

    def test_delete_backtest_result_not_found(
        self, repository: BacktestResultRepository
    ) -> None:
        """存在しないIDの場合Falseが返される"""
        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = None
        repository.db.query.return_value = mock_query
        
        result = repository.delete_backtest_result(999)
        
        assert result is False
        repository.db.delete.assert_not_called()


class TestDeleteAllBacktestResults:
    """delete_all_backtest_resultsメソッドのテスト"""

    def test_delete_all_backtest_results_success(
        self, repository: BacktestResultRepository
    ) -> None:
        """全バックテスト結果が削除できる"""
        mock_query = MagicMock()
        mock_query.delete.return_value = 5
        repository.db.query.return_value = mock_query
        
        count = repository.delete_all_backtest_results()
        
        assert count == 5
        repository.db.commit.assert_called_once()


class TestCountBacktestResults:
    """count_backtest_resultsメソッドのテスト"""

    def test_count_backtest_results_all(
        self, repository: BacktestResultRepository
    ) -> None:
        """全バックテスト結果の数が取得できる"""
        mock_query = MagicMock()
        mock_query.count.return_value = 10
        repository.db.query.return_value = mock_query
        
        count = repository.count_backtest_results()
        
        assert count == 10

    def test_count_backtest_results_with_symbol_filter(
        self, repository: BacktestResultRepository
    ) -> None:
        """シンボルフィルター付きでカウントできる"""
        mock_query = MagicMock()
        mock_query.filter.return_value.count.return_value = 3
        repository.db.query.return_value = mock_query
        
        count = repository.count_backtest_results(symbol="BTC/USDT")
        
        assert count == 3

    def test_count_backtest_results_with_strategy_filter(
        self, repository: BacktestResultRepository
    ) -> None:
        """戦略名フィルター付きでカウントできる"""
        mock_query = MagicMock()
        mock_query.filter.return_value.count.return_value = 2
        repository.db.query.return_value = mock_query
        
        count = repository.count_backtest_results(
            strategy_name="test_strategy"
        )
        
        assert count == 2


class TestGetRecentBacktestResults:
    """get_recent_backtest_resultsメソッドのテスト"""

    def test_get_recent_backtest_results_success(
        self,
        repository: BacktestResultRepository,
        sample_backtest_model: BacktestResult,
    ) -> None:
        """最近のバックテスト結果が取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_backtest_model]
        repository.db.scalars.return_value = mock_scalars
        
        results = repository.get_recent_backtest_results(limit=5)
        
        assert len(results) == 1
        assert results[0]["id"] == 1

    def test_get_recent_backtest_results_with_limit(
        self, repository: BacktestResultRepository
    ) -> None:
        """リミット指定で最近の結果が取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        repository.db.scalars.return_value = mock_scalars
        
        results = repository.get_recent_backtest_results(limit=3)
        
        assert isinstance(results, list)


class TestDataNormalization:
    """データ正規化のテスト"""

    def test_normalize_result_data_complete(
        self,
        repository: BacktestResultRepository,
        sample_backtest_data: Dict[str, Any],
    ) -> None:
        """完全なデータが正規化される"""
        normalized = repository._normalize_result_data(sample_backtest_data)
        
        assert "strategy_name" in normalized
        assert "performance_metrics" in normalized
        assert "equity_curve" in normalized

    def test_normalize_result_data_with_legacy_format(
        self, repository: BacktestResultRepository
    ) -> None:
        """レガシー形式のデータが正規化される"""
        legacy_data = {
            "strategy_name": "legacy_strategy",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01T00:00:00+00:00",
            "end_date": "2024-01-31T00:00:00+00:00",
            "initial_capital": 10000.0,
            "total_return": 0.15,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.10,
        }
        
        normalized = repository._normalize_result_data(legacy_data)
        
        assert "performance_metrics" in normalized
        assert normalized["performance_metrics"]["total_return"] == 0.15


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_save_backtest_result_handles_error(
        self, repository: BacktestResultRepository
    ) -> None:
        """保存エラーが適切に処理される"""
        repository.db.add.side_effect = Exception("DB Error")
        
        with pytest.raises(Exception):
            repository.save_backtest_result({
                "strategy_name": "test",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": datetime.now(timezone.utc),
                "end_date": datetime.now(timezone.utc),
                "initial_capital": 10000,
            })

    def test_delete_handles_error(
        self, repository: BacktestResultRepository
    ) -> None:
        """削除エラーが適切に処理される"""
        repository.db.query.side_effect = Exception("Delete Error")
        
        with pytest.raises(Exception):
            repository.delete_backtest_result(1)


class TestJsonSafeConversion:
    """JSON変換のテスト"""

    def test_to_json_safe_datetime(
        self, repository: BacktestResultRepository
    ) -> None:
        """datetimeがISO形式に変換される"""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = repository._to_json_safe(dt)
        
        assert isinstance(result, str)
        assert "2024-01-01" in result

    def test_to_json_safe_dict(
        self, repository: BacktestResultRepository
    ) -> None:
        """辞書が再帰的に変換される"""
        data = {
            "date": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "value": 100,
        }
        result = repository._to_json_safe(data)
        
        assert isinstance(result, dict)
        assert isinstance(result["date"], str)

    def test_to_json_safe_list(
        self, repository: BacktestResultRepository
    ) -> None:
        """リストが再帰的に変換される"""
        data = [
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            100,
        ]
        result = repository._to_json_safe(data)
        
        assert isinstance(result, list)
        assert isinstance(result[0], str)
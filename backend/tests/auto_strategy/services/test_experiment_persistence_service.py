
import pytest
from unittest.mock import MagicMock, patch, ANY
from app.services.auto_strategy.services.experiment_persistence_service import ExperimentPersistenceService
from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.models.strategy_models import StrategyGene

# --- テストデータとヘルパー --- 

@pytest.fixture
def mock_db_session_factory():
    """DBセッションファクトリのモック"""
    db_session = MagicMock()
    # コンテキストマネージャとして動作するように設定
    db_session_factory = MagicMock()
    db_session_factory.return_value.__enter__.return_value = db_session
    return db_session_factory

@pytest.fixture
def mock_backtest_service():
    """バックテストサービスのモック"""
    service = MagicMock()
    service.run_backtest.return_value = {
        "performance_metrics": {"total_return": 0.5},
        "equity_curve": [],
        "trade_history": [],
        "execution_time": 10.0
    }
    return service

@pytest.fixture
def mock_repositories(monkeypatch):
    """すべてのリポジトリをモック化するフィクスチャ"""
    mock_ga_repo = MagicMock()
    mock_ga_repo.create_experiment.return_value.id = 1
    
    mock_strategy_repo = MagicMock()
    mock_backtest_repo = MagicMock()

    monkeypatch.setattr('app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository', lambda x: mock_ga_repo)
    monkeypatch.setattr('app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository', lambda x: mock_strategy_repo)
    monkeypatch.setattr('app.services.auto_strategy.services.experiment_persistence_service.BacktestResultRepository', lambda x: mock_backtest_repo)
    
    return mock_ga_repo, mock_strategy_repo, mock_backtest_repo

@pytest.fixture
def persistence_service(mock_db_session_factory, mock_backtest_service):
    """テスト対象のサービスインスタンス"""
    return ExperimentPersistenceService(mock_db_session_factory, mock_backtest_service)

def create_test_ga_config(is_multi_objective=False):
    return GAConfig.from_dict({"generations": 10, "enable_multi_objective": is_multi_objective})

def create_test_strategy_gene(id_suffix):
    return StrategyGene(id=f"gene_{id_suffix}", indicators=[], entry_conditions=[], exit_conditions=[], risk_management={})

# --- テストクラス --- 

class TestExperimentPersistenceService:

    def test_create_experiment(self, persistence_service, mock_repositories):
        """実験作成が正しくリポジトリを呼び出すか"""
        mock_ga_repo, _, _ = mock_repositories
        experiment_id = "test-uuid-123"
        ga_config = create_test_ga_config()
        backtest_config = {"symbol": "BTC/USDT"}

        result_id = persistence_service.create_experiment(experiment_id, "Test Exp", ga_config, backtest_config)

        assert result_id == experiment_id
        mock_ga_repo.create_experiment.assert_called_once_with(
            name="Test Exp",
            config=ANY, # configの中身は複雑なのでANYでチェック
            total_generations=10,
            status="running"
        )
        # configの中身を部分的に検証
        _, kwargs = mock_ga_repo.create_experiment.call_args
        assert kwargs['config']['experiment_id'] == experiment_id

    def test_save_experiment_result_single_objective(self, persistence_service, mock_repositories, mock_backtest_service):
        """単一目的GAの結果保存フローが正しく動作するか"""
        mock_ga_repo, mock_strategy_repo, mock_backtest_repo = mock_repositories
        experiment_id = "test-uuid-456"
        db_id = 1
        
        # get_experiment_infoが正しい情報を返すようにモックを設定
        persistence_service.get_experiment_info = MagicMock(return_value={"db_id": db_id, "name": "Test Exp"})

        result = {
            "best_strategy": create_test_strategy_gene(1),
            "best_fitness": 1.23,
            "all_strategies": [create_test_strategy_gene(1), create_test_strategy_gene(2)],
            "fitness_scores": [1.23, 1.10]
        }
        ga_config = create_test_ga_config()
        backtest_config = {
            "symbol": "BTC/USDT", 
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 100000
        }

        persistence_service.save_experiment_result(experiment_id, result, ga_config, backtest_config)

        persistence_service.get_experiment_info.assert_called_once_with(experiment_id)
        mock_strategy_repo.save_strategy.assert_called_once()
        mock_backtest_service.run_backtest.assert_called_once()
        mock_backtest_repo.save_backtest_result.assert_called_once()
        mock_strategy_repo.save_strategies_batch.assert_called_once()

    def test_save_experiment_result_multi_objective(self, persistence_service, mock_repositories):
        """多目的GAの結果保存でパレート最適解が保存されるか"""
        mock_ga_repo, mock_strategy_repo, _ = mock_repositories
        experiment_id = "test-uuid-789"
        db_id = 2
        persistence_service.get_experiment_info = MagicMock(return_value={"db_id": db_id, "name": "Multi-Obj Test"})

        result = {
            "best_strategy": create_test_strategy_gene(1),
            "best_fitness": (1.5, 0.2),
            "pareto_front": [
                {"strategy": create_test_strategy_gene(1), "fitness_values": (1.5, 0.2)},
                {"strategy": create_test_strategy_gene(2), "fitness_values": (1.2, 0.1)}
            ]
        }
        ga_config = create_test_ga_config(is_multi_objective=True)
        backtest_config = {"symbol": "ETH/USDT"}

        persistence_service.save_experiment_result(experiment_id, result, ga_config, backtest_config)

        # save_strategies_batchがパレート最適解のデータで呼び出されることを確認
        # 1回目は最良戦略以外(このテストケースではなし)、2回目はパレート解
        assert mock_strategy_repo.save_strategies_batch.call_count == 1
        call_args, _ = mock_strategy_repo.save_strategies_batch.call_args
        saved_data = call_args[0]
        assert len(saved_data) == 2
        assert saved_data[0]["fitness_values"] == (1.5, 0.2)

    def test_update_status_methods(self, persistence_service, mock_repositories):
        """complete, fail, stopの各メソッドが正しくステータスを更新するか"""
        mock_ga_repo, _, _ = mock_repositories
        experiment_id = "test-uuid-status"
        db_id = 3
        persistence_service.get_experiment_info = MagicMock(return_value={"db_id": db_id})

        # complete
        persistence_service.complete_experiment(experiment_id)
        mock_ga_repo.update_experiment_status.assert_called_with(db_id, "completed")

        # fail
        persistence_service.fail_experiment(experiment_id)
        mock_ga_repo.update_experiment_status.assert_called_with(db_id, "failed")

        # stop
        persistence_service.stop_experiment(experiment_id)
        mock_ga_repo.update_experiment_status.assert_called_with(db_id, "stopped")

        assert mock_ga_repo.update_experiment_status.call_count == 3

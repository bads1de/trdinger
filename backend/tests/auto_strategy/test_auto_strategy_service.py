from unittest.mock import MagicMock, patch

import pytest
from fastapi import BackgroundTasks
from fastapi.exceptions import HTTPException

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.services.auto_strategy_service import (
    AutoStrategyService,
)


@pytest.fixture
def mock_db_session_factory():
    """データベースセッションファクトリのモック"""
    return MagicMock()


@pytest.fixture
def mock_backtest_service():
    """バックテストサービスのモック"""
    return MagicMock()


@pytest.fixture
def mock_persistence_service():
    """永続化サービスのモック"""
    return MagicMock()


@pytest.fixture
def mock_experiment_manager():
    """実験管理マネージャーのモック"""
    manager = MagicMock()
    manager.stop_experiment.return_value = True
    return manager


@pytest.fixture
def auto_strategy_service(
    mock_db_session_factory,
    mock_backtest_service,
    mock_persistence_service,
    mock_experiment_manager,
):
    """AutoStrategyServiceのテストインスタンス"""
    with (
        patch(
            "app.services.auto_strategy.services.auto_strategy_service.SessionLocal",
            mock_db_session_factory,
        ),
        patch(
            "app.services.auto_strategy.services.auto_strategy_service.BacktestService",
            return_value=mock_backtest_service,
        ),
        patch(
            "app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService",
            return_value=mock_persistence_service,
        ),
        patch(
            "app.services.auto_strategy.services.auto_strategy_service.ExperimentManager",
            return_value=mock_experiment_manager,
        ),
    ):
        service = AutoStrategyService()
        # _init_servicesで上書きされるため、再度モックを割り当てる
        service.db_session_factory = mock_db_session_factory
        service.backtest_service = mock_backtest_service
        service.persistence_service = mock_persistence_service
        service.experiment_manager = mock_experiment_manager
        return service


# --- テストケース ---


def test_start_strategy_generation_success(
    auto_strategy_service, mock_persistence_service, mock_experiment_manager
):
    """正常系: 戦略生成が正常に開始されること"""
    # 準備 - GA設定を有効な値に修正
    experiment_id = "test-exp-id"
    experiment_name = "Test Experiment"
    ga_config_dict = {
        "population_size": 10,
        "generations": 5,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1,
        "elite_size": 2,
    }
    backtest_config_dict = {"symbol": "BTC/USDT", "timeframe": "1h"}
    background_tasks = BackgroundTasks()

    # 実行
    result_exp_id = auto_strategy_service.start_strategy_generation(
        experiment_id,
        experiment_name,
        ga_config_dict,
        backtest_config_dict,
        background_tasks,
    )

    # 検証
    assert result_exp_id == experiment_id
    mock_persistence_service.create_experiment.assert_called_once()
    mock_experiment_manager.initialize_ga_engine.assert_called_once()
    assert len(background_tasks.tasks) == 1


def test_start_strategy_generation_invalid_ga_config(auto_strategy_service):
    """異常系: 無効なGA設定でHTTPExceptionが発生すること"""
    # 準備
    with patch(
        "app.services.auto_strategy.config.GAConfig.validate",
        return_value=(False, ["error"]),
    ):
        with pytest.raises(HTTPException):
            auto_strategy_service.start_strategy_generation(
                "test-id", "test-name", {}, {}, BackgroundTasks()
            )


def test_list_experiments(auto_strategy_service, mock_persistence_service):
    """正常系: 実験一覧が正常に取得できること"""
    # 準備
    mock_experiments = [{"id": "exp1", "name": "Experiment 1"}]
    mock_persistence_service.list_experiments.return_value = mock_experiments

    # 実行
    experiments = auto_strategy_service.list_experiments()

    # 検証
    assert experiments == mock_experiments
    mock_persistence_service.list_experiments.assert_called_once()


def test_stop_experiment_success(auto_strategy_service, mock_experiment_manager):
    """正常系: 実験が正常に停止できること"""
    # 準備
    experiment_id = "test-exp-id-to-stop"
    mock_experiment_manager.stop_experiment.return_value = True

    # 実行
    result = auto_strategy_service.stop_experiment(experiment_id)

    # 検証
    mock_experiment_manager.stop_experiment.assert_called_with(experiment_id)
    assert result == {"success": True, "message": "実験が正常に停止されました"}


def test_stop_experiment_failure(auto_strategy_service, mock_experiment_manager):
    """正常系: 実験の停止に失敗した場合"""
    # 準備
    experiment_id = "test-exp-id-to-fail-stop"
    mock_experiment_manager.stop_experiment.return_value = False

    # 実行
    result = auto_strategy_service.stop_experiment(experiment_id)

    # 検証
    assert result == {"success": False, "message": "実験の停止に失敗しました"}


def test_stop_experiment_manager_not_initialized(auto_strategy_service):
    """異常系: experiment_managerが初期化されていない場合"""
    # 準備
    auto_strategy_service.experiment_manager = None

    # 実行
    result = auto_strategy_service.stop_experiment("some-id")

    # 検証
    assert result == {
        "success": False,
        "message": "実験管理マネージャーが初期化されていません",
    }


def test_prepare_ga_config_valid(auto_strategy_service):
    """正常系: GA設定の準備が成功すること"""
    ga_config_dict = {"population_size": 50, "generations": 10}
    ga_config = auto_strategy_service._prepare_ga_config(ga_config_dict)
    assert isinstance(ga_config, GAConfig)
    assert ga_config.population_size == 50


def test_create_experiment_called_with_correct_args(
    auto_strategy_service, mock_persistence_service
):
    """正常系: create_experimentが正しい引数で呼ばれること"""
    experiment_id = "uuid-test"
    experiment_name = "Test Create"
    ga_config = GAConfig.from_dict({})  # Assuming GAConfig has a from_dict method
    backtest_config = {"symbol": "ETH/USDT"}

    auto_strategy_service._create_experiment(
        experiment_id, experiment_name, ga_config, backtest_config
    )

    mock_persistence_service.create_experiment.assert_called_with(
        experiment_id, experiment_name, ga_config, backtest_config
    )


def test_initialize_ga_engine_runtime_error(auto_strategy_service):
    """異常系: experiment_managerなしでGAエンジンを初期化しようとするとRuntimeError"""
    auto_strategy_service.experiment_manager = None
    with pytest.raises(
        RuntimeError, match="実験管理マネージャーが初期化されていません。"
    ):
        auto_strategy_service._initialize_ga_engine(
            GAConfig.from_dict({})
        )  # Assuming GAConfig has a from_dict method


def test_start_background_task_added(auto_strategy_service, mock_experiment_manager):
    """正常系: バックグラウンドタスクが追加されること"""
    background_tasks = BackgroundTasks()
    experiment_id = "bg-task-test"
    ga_config = GAConfig.from_dict({})  # Assuming GAConfig has a from_dict method
    backtest_config = {}

    auto_strategy_service._start_experiment_in_background(
        experiment_id, ga_config, backtest_config, background_tasks
    )

    assert len(background_tasks.tasks) == 1
    task = background_tasks.tasks[0]
    assert task.func == mock_experiment_manager.run_experiment
    assert task.args == (experiment_id, ga_config, backtest_config)

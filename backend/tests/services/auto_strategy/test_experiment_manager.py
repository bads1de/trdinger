"""
ExperimentManagerのテスト
"""

from unittest.mock import MagicMock, Mock, patch

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.genes import StrategyGene
from app.services.auto_strategy.services.experiment_engine_registry import (
    ExperimentEngineRegistry,
)
from app.services.auto_strategy.services import (
    experiment_manager as experiment_manager_module,
)
from app.services.auto_strategy.services.experiment_manager import ExperimentManager


class TestExperimentManager:
    """ExperimentManagerのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.registry = ExperimentEngineRegistry()
        self.mock_backtest_service = Mock()
        self.mock_persistence_service = Mock()
        self.manager = ExperimentManager(
            self.mock_backtest_service,
            self.mock_persistence_service,
            engine_registry=self.registry,
        )

    def teardown_method(self):
        """テスト後の後始末"""
        self.registry.clear()

    def test_init(self):
        """初期化のテスト"""
        assert self.manager.backtest_service == self.mock_backtest_service
        assert self.manager.persistence_service == self.mock_persistence_service
        assert self.manager._engine_registry is self.registry
        assert self.manager._get_active_engine("missing") is None

    def test_initialize_ga_engine(self):
        """GAエンジン初期化のテスト"""
        ga_config = GAConfig()
        ga_config.population_size = 10
        ga_config.generations = 5

        mock_ga_engine = MagicMock()
        with patch(
            "app.services.auto_strategy.core.engine.ga_engine_factory.GeneticAlgorithmEngineFactory.create_engine",
            return_value=mock_ga_engine,
        ):
            returned_engine = self.manager.initialize_ga_engine(
                ga_config, "test_exp_001"
            )

        assert returned_engine is mock_ga_engine
        assert self.manager._get_active_engine("test_exp_001") is mock_ga_engine

    def test_run_experiment_success(self):
        """実験実行成功のテスト"""
        # GA設定の準備
        ga_config = GAConfig()
        ga_config.population_size = 10
        ga_config.generations = 5

        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-12-19",
            "initial_capital": 10000,
        }

        experiment_info = {
            "db_id": 7,
            "name": "AUTO_STRATEGY_GA_2024-01-02_TEST_RUN_",
            "status": "running",
            "config": {"experiment_id": "test_exp_001"},
        }
        self.manager.persistence_service.get_experiment_info.return_value = (
            experiment_info
        )

        strategy = StrategyGene(id="abcdef123456")
        mock_ga_engine = MagicMock()
        mock_ga_engine.run_evolution.return_value = {
            "best_strategy": strategy,
            "best_fitness": 1.5,
            "all_strategies": [strategy],
            "fitness_scores": [1.5],
        }
        mock_ga_engine.is_stop_requested.return_value = False
        with patch(
            "app.services.auto_strategy.core.engine.ga_engine_factory.GeneticAlgorithmEngineFactory.create_engine",
            return_value=mock_ga_engine,
        ):
            self.manager.initialize_ga_engine(ga_config, "test_exp_001")

        self.mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {"total_return": 0.1},
            "equity_curve": [],
            "trade_history": [],
            "execution_time": 0.25,
        }

        self.manager.run_experiment("test_exp_001", ga_config, backtest_config)

        # 検証
        expected_backtest_config = dict(backtest_config)
        expected_backtest_config["experiment_id"] = "test_exp_001"

        mock_ga_engine.run_evolution.assert_called_once()
        call_args = mock_ga_engine.run_evolution.call_args
        assert call_args[0][0] == ga_config
        assert call_args[0][1] == expected_backtest_config
        assert "progress_callback" in call_args[1]
        self.manager.persistence_service.save_experiment_result.assert_called_once_with(
            "test_exp_001",
            {
                "best_strategy": strategy,
                "best_fitness": 1.5,
                "all_strategies": [strategy],
                "fitness_scores": [1.5],
            },
            ga_config,
            expected_backtest_config,
            experiment_info=experiment_info,
        )
        self.manager.persistence_service.save_backtest_result.assert_called_once()
        saved_backtest_result = (
            self.manager.persistence_service.save_backtest_result.call_args.args[0]
        )
        assert saved_backtest_result["strategy_name"] == "AS_GA_240102_abcdef"
        assert saved_backtest_result["config_json"]["experiment_id"] == "test_exp_001"
        assert saved_backtest_result["config_json"]["db_experiment_id"] == 7
        assert saved_backtest_result["config_json"]["fitness_score"] == 1.5
        self.manager.persistence_service.complete_experiment.assert_called_once_with(
            "test_exp_001"
        )
        self.mock_backtest_service.run_backtest.assert_called_once()
        assert self.manager._get_active_engine("test_exp_001") is None

    def test_run_experiment_exception(self):
        """実験実行中の例外テスト"""
        ga_config = GAConfig()
        backtest_config = {}

        mock_ga_engine = MagicMock()
        mock_ga_engine.run_evolution.side_effect = Exception("Test exception")
        mock_ga_engine.is_stop_requested.return_value = False
        self.manager._register_active_engine("test_exp_001", mock_ga_engine)

        self.manager.run_experiment("test_exp_001", ga_config, backtest_config)

        # エラーハンドリングの検証
        self.manager.persistence_service.fail_experiment.assert_called_once_with(
            "test_exp_001"
        )
        assert self.manager._get_active_engine("test_exp_001") is None

    def test_stop_experiment(self):
        """実験停止のテスト"""
        mock_ga_engine = MagicMock()
        self.manager._register_active_engine("test_exp_001", mock_ga_engine)

        self.manager.stop_experiment("test_exp_001")

        mock_ga_engine.stop_evolution.assert_called_once()
        self.manager.persistence_service.stop_experiment.assert_called_once_with(
            "test_exp_001"
        )

    def test_stop_experiment_across_instances_with_shared_default_registry(self):
        """デフォルト共有レジストリなら別インスタンスから停止できること"""
        ga_config = GAConfig()
        ga_config.population_size = 10
        ga_config.generations = 5
        experiment_id = "test_exp_registry"
        mock_ga_engine = MagicMock()
        other_persistence = Mock()
        experiment_manager_module._DEFAULT_ENGINE_REGISTRY.clear()
        try:
            manager_a = ExperimentManager(
                self.mock_backtest_service, self.mock_persistence_service
            )
            other_manager = ExperimentManager(
                self.mock_backtest_service, other_persistence
            )

            with patch(
                "app.services.auto_strategy.core.engine.ga_engine_factory.GeneticAlgorithmEngineFactory.create_engine",
                return_value=mock_ga_engine,
            ):
                returned_engine = manager_a.initialize_ga_engine(
                    ga_config, experiment_id
                )

            result = other_manager.stop_experiment(experiment_id)

            assert result is True
            assert returned_engine is mock_ga_engine
            mock_ga_engine.stop_evolution.assert_called_once()
            other_persistence.stop_experiment.assert_called_once_with(experiment_id)
        finally:
            experiment_manager_module._DEFAULT_ENGINE_REGISTRY.clear()

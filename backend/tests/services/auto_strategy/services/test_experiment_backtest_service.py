from unittest.mock import Mock

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.genes import StrategyGene
from app.services.auto_strategy.services.experiment_backtest_service import (
    ExperimentBacktestService,
)


class TestExperimentBacktestService:
    def setup_method(self):
        self.mock_backtest_service = Mock()
        self.service = ExperimentBacktestService(self.mock_backtest_service)

    def test_create_detailed_backtest_result_data(self):
        strategy = StrategyGene(id="abcdef123456")
        result = {
            "best_strategy": strategy,
            "best_fitness": 1.5,
        }
        ga_config = GAConfig()
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "initial_capital": 10000,
            "commission_rate": 0.002,
        }
        experiment_info = {
            "db_id": 42,
            "name": "AUTO_STRATEGY_GA_2024-01-02_TEST_RUN_",
            "status": "running",
        }
        self.mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {"total_return": 0.25},
            "equity_curve": [1, 2, 3],
            "trade_history": [{"id": 1}],
            "execution_time": 0.5,
        }

        detailed_result = self.service.create_detailed_backtest_result_data(
            result=result,
            ga_config=ga_config,
            backtest_config=backtest_config,
            experiment_id="exp_001",
            experiment_info=experiment_info,
        )

        self.mock_backtest_service.run_backtest.assert_called_once()
        run_config = self.mock_backtest_service.run_backtest.call_args.args[0]
        assert run_config["strategy_name"] == "AS_GA_240102_abcdef"
        assert run_config["strategy_config"]["parameters"]["strategy_gene"]["id"] == (
            "abcdef123456"
        )
        assert detailed_result["strategy_name"] == "AS_GA_240102_abcdef"
        assert detailed_result["config_json"]["experiment_id"] == "exp_001"
        assert detailed_result["config_json"]["db_experiment_id"] == 42
        assert detailed_result["config_json"]["fitness_score"] == 1.5
        assert detailed_result["performance_metrics"] == {"total_return": 0.25}
        assert detailed_result["equity_curve"] == [1, 2, 3]
        assert detailed_result["trade_history"] == [{"id": 1}]
        assert detailed_result["execution_time"] == 0.5
        assert detailed_result["status"] == "completed"

    def test_create_detailed_backtest_result_data_multi_objective(self):
        strategy = StrategyGene(id="123456abcdef")
        result = {
            "best_strategy": strategy,
            "best_fitness": [2.25, 0.5],
        }
        ga_config = GAConfig()
        ga_config.enable_multi_objective = True
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "4h",
            "start_date": "2024-02-01",
            "end_date": "2024-02-02",
            "initial_capital": 5000,
        }
        self.mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {},
            "equity_curve": [],
            "trade_history": [],
            "execution_time": 0.1,
        }

        detailed_result = self.service.create_detailed_backtest_result_data(
            result=result,
            ga_config=ga_config,
            backtest_config=backtest_config,
            experiment_id="exp_multi",
            experiment_info={"db_id": 7, "name": "AUTO_STRATEGY_GA_TEST"},
        )

        assert detailed_result["config_json"]["fitness_score"] == 2.25

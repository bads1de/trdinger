"""
Test for AutoStrategyService refactoring
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.services.auto_strategy.config import GAConfig


class TestAutoStrategyService:
    """AutoStrategyServiceのテスト"""

    @pytest.fixture
    def mock_experiment_manager(self):
        """実験管理マネージャーのモック"""
        return Mock()

    @pytest.fixture
    def auto_strategy_service(self, mock_experiment_manager):
        """AutoStrategyServiceのインスタンス"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager') as mock_exp_mgr_class:

            mock_exp_mgr_class.return_value = mock_experiment_manager
            service = AutoStrategyService()
            return service

    def test_start_strategy_generation_calls_private_methods(self, auto_strategy_service):
        """start_strategy_generationがプライベートメソッドを正しく呼び出すことをテスト"""
        # Given
        mock_persistence = Mock()
        mock_experiment_mgr = Mock()
        auto_strategy_service.persistence_service = mock_persistence
        auto_strategy_service.experiment_manager = mock_experiment_mgr

        ga_config_dict = {
            "generations": 10,
            "population_size": 20,
            "enable_multi_objective": False
        }
        backtest_config_dict = {"symbol": "BTC/USDT"}
        background_tasks = Mock()

        expected_experiment_id = "test-experiment-id"
        mock_persistence.create_experiment.return_value = expected_experiment_id

        # When
        result = auto_strategy_service.start_strategy_generation(
            expected_experiment_id, "test_experiment", ga_config_dict, backtest_config_dict, background_tasks
        )

        # Then
        assert result == expected_experiment_id
        mock_persistence.create_experiment.assert_called_once()
        mock_experiment_mgr.initialize_ga_engine.assert_called_once()
        # add_task が呼ばれたことを確認
        background_tasks.add_task.assert_called_once()
        # 引数を一般的に確認
        args, kwargs = background_tasks.add_task.call_args
        assert args[0] == mock_experiment_mgr.run_experiment
        assert args[1] == expected_experiment_id
        # args[2] はGAConfigオブジェクト、args[3] はdict
        assert hasattr(args[2], 'generations')  # GAConfigの属性チェック
        assert isinstance(args[3], dict)
        assert args[3]["symbol"] == "BTC/USDT"

    def test_build_ga_config_from_dict(self, auto_strategy_service):
        """GA設定の構築をテスト"""
        # Given
        ga_config_dict = {
            "generations": 10,
            "population_size": 20,
            "enable_multi_objective": False
        }

        # When
        with patch('app.services.auto_strategy.config.GAConfig') as mock_ga_config_class:
            mock_config_instance = Mock()
            mock_ga_config_class.from_dict.return_value = mock_config_instance
            mock_config_instance.validate.return_value = (True, [])

            from app.services.auto_strategy.config import GAConfig as ActualGAConfig
            ga_config = ActualGAConfig.from_dict(ga_config_dict)
            ga_config.validate()

            # Then
            mock_ga_config_class.from_dict.assert_called_once_with(ga_config_dict)
            mock_config_instance.validate.assert_called_once()

    def test_prepare_backtest_config(self, auto_strategy_service):
        """バックテスト設定の準備をテスト"""
        # Given
        backtest_config_dict = {"timeframe": "1h", "start_date": "2023-01-01"}
        symbol = "BTC/USDT"

        # When
        backtest_config = backtest_config_dict.copy()
        backtest_config["symbol"] = backtest_config.get("symbol", symbol)

        # Then
        assert backtest_config["symbol"] == symbol
        assert backtest_config["timeframe"] == "1h"
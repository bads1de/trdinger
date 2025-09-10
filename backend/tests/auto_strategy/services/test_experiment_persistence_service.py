"""
Experiment Persistence Service テスト

ExperimentPersistenceServiceの機能をテストします。
"""

import unittest
from unittest.mock import Mock, MagicMock
import uuid
from sqlalchemy.orm import Session

from app.services.auto_strategy.services.experiment_persistence_service import ExperimentPersistenceService
from app.services.auto_strategy.config import GAConfig


class TestExperimentPersistenceService(unittest.TestCase):
    """ExperimentPersistenceServiceテスト"""

    def setUp(self):
        """セットアップ"""
        self.db_session_factory_mock = Mock()
        self.backtest_service_mock = Mock()
        self.service = ExperimentPersistenceService(
            self.db_session_factory_mock,
            self.backtest_service_mock
        )

    def test_create_experiment(self):
        """実験作成テスト"""
        experiment_id = str(uuid.uuid4())
        experiment_name = "Test Experiment"
        ga_config = GAConfig(population_size=10, generations=5)
        backtest_config = {"symbol": "BTC/USDT"}

        # Mock DB session and repository
        db_mock = Mock()
        ga_experiment_repo_mock = Mock()
        db_experiment_mock = Mock()
        db_experiment_mock.id = 123

        ga_experiment_repo_mock.create_experiment.return_value = db_experiment_mock
        self.db_session_factory_mock.return_value.__enter__.return_value = db_mock
        self.db_session_factory_mock.return_value.__exit__.return_value = None

        with unittest.mock.patch('app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository') as mock_registry:
            mock_registry.return_value = ga_experiment_repo_mock

            result = self.service.create_experiment(
                experiment_id, experiment_name, ga_config, backtest_config
            )

            self.assertEqual(result, experiment_id)
            ga_experiment_repo_mock.create_experiment.assert_called_once()

    def test_save_experiment_result(self):
        """実験結果保存テスト"""
        experiment_id = str(uuid.uuid4())
        result = {
            "best_strategy": Mock(),
            "best_fitness": 100.0,
            "all_strategies": [Mock(), Mock()],
            "fitness_scores": [99.0, 95.0]
        }
        ga_config = GAConfig(population_size=10, generations=5)
        backtest_config = {"symbol": "BTC/USDT"}

        db_mock = Mock()
        self.db_session_factory_mock.return_value.__enter__.return_value = db_mock
        self.db_session_factory_mock.return_value.__exit__.return_value = None

        # Mock repositories
        with unittest.mock.patch('app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository') as mock_gen_repo, \
             unittest.mock.patch('app.services.auto_strategy.services.experiment_persistence_service.BacktestResultRepository') as mock_back_repo, \
             unittest.mock.patch.object(self.service, 'get_experiment_info') as mock_get_info:

            mock_gen_repo_instance = Mock()
            mock_gen_repo.report.call_count = Mock()
            mock_back_repo_instance = Mock()
            mock_get_info.return_value = {"db_id": 123, "name": "Test"}

            mock_gen_repo.return_value.something = mock_gen_repo_instance
            mock_back_repo.return_value = mock_back_repo_instance

            with unittest.mock.patch.object(self.service, '_save_best_strategy_and_run_detailed_backtest'), \
                 unittest.mock.patch.object(self.service, '_save_other_strategies'), \
                 unittest.mock.patch.object(self.service, '_save_pareto_front'):

                self.service.save_experiment_result(
                    experiment_id, result, ga_config, backtest_config
                )

                # Verify get_experiment_info was called
                mock_get_info.assert_called_once_with(experiment_id)

    def test_get_experiment_info_by_config_experiment_id(self):
        """config内のexperiment_idによる実験情報取得テスト"""
        experiment_id = str(uuid.uuid4())
        expected_info = {
            "db_id": 123,
            "name": "Test Experiment",
            "status": "running",
            "config": {"experiment_id": experiment_id}
        }

        db_mock = Mock()
        ga_experiment_repo_mock = Mock()
        experiment_mock = Mock()
        experiment_mock.id = 123
        experiment_mock.name = "Test Experiment"
        experiment_mock.status = "running"
        experiment_mock.config = {"experiment_id": experiment_id}

        ga_experiment_repo_mock.get_recent_experiments.return_value = [experiment_mock]
        self.db_session_factory_mock.return_value.__enter__.return_value = db_mock
        self.db_session_factory_mock.return_value.__exit__.return_value = None

        with unittest.mock.patch('app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository') as mock_registry:
            mock_registry.return_value = ga_experiment_repo_mock

            result = self.service.get_experiment_info(experiment_id)

            self.assertIsNotNone(result)
            self.assertEqual(result["db_id"], 123)
            self.assertEqual(result["name"], "Test Experiment")

    def test_get_experiment_info_not_found(self):
        """実験情報見つからない場合のテスト"""
        experiment_id = str(uuid.uuid4())

        db_mock = Mock()
        ga_experiment_repo_mock = Mock()
        ga_experiment_repo_mock.get_recent_experiments.return_value = []

        self.db_session_factory_mock.return_value.__enter__.return_value = db_mock
        self.db_session_factory_mock.return_value.__exit__.return_value = None

        with unittest.mock.patch('app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository') as mock_registry:
            mock_registry.return_value = ga_experiment_repo_mock

            result = self.service.get_experiment_info(experiment_id)

            self.assertIsNone(result)

    def test_list_experiments(self):
        """実験一覧取得テスト"""
        db_mock = Mock()
        ga_experiment_repo_mock = Mock()
        experiment_mock = Mock()
        experiment_mock.id = 1
        experiment_mock.name = "Test Experiment"
        experiment_mock.status = "completed"
        experiment_mock.created_at = Mock()
        experiment_mock.completed_at = Mock()
        experiment_mock.created_at.isoformat.return_value = "2023-01-01T00:00:00"
        experiment_mock.completed_at.isoformat.return_value = "2023-01-02T00:00:00"

        ga_experiment_repo_mock.get_recent_experiments.return_value = [experiment_mock]
        self.db_session_factory_mock.return_value.__enter__.return_value = db_mock
        self.db_session_factory_mock.return_value.__exit__.return_value = None

        with unittest.mock.patch('app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository') as mock_registry:
            mock_registry.return_value = ga_experiment_repo_mock

            result = self.service.list_experiments()

            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["experiment_name"], "Test Experiment")

    def test_complete_experiment(self):
        """実験完了処理テスト"""
        experiment_id = str(uuid.uuid4())

        db_mock = Mock()
        ga_experiment_repo_mock = Mock()

        self.db_session_factory_mock.return_value.__enter__.return_value = db_mock
        self.db_session_factory_mock.return_value.__exit__.return_value = None

        with unittest.mock.patch('app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository') as mock_registry, \
             unittest.mock.patch.object(self.service, 'get_experiment_info') as mock_get_info:

            mock_registry.return_value = ga_experiment_repo_mock
            mock_get_info.return_value = {"db_id": 123}

            self.service.complete_experiment(experiment_id)

            mock_get_info.assert_called_once_with(experiment_id)
            ga_experiment_repo_mock.update_experiment_status.assert_called_once_with(123, "completed")

    def test_fail_experiment(self):
        """実験失敗処理テスト"""
        experiment_id = str(uuid.uuid4())

        db_mock = Mock()
        ga_experiment_repo_mock = Mock()

        self.db_session_factory_mock.return_value.__enter__.return_value = db_mock
        self.db_session_factory_mock.return_value.__exit__.return_value = None

        with unittest.mock.patch('app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository') as mock_registry, \
             unittest.mock.patch.object(self.service, 'get_experiment_info') as mock_get_info:

            mock_registry.return_value = ga_experiment_repo_mock
            mock_get_info.return_value = {"db_id": 123}

            self.service.fail_experiment(experiment_id)

            mock_get_info.assert_called_once_with(experiment_id)
            ga_experiment_repo_mock.update_experiment_status.assert_called_once_with(123, "failed")

    def test_stop_experiment(self):
        """実験停止処理テスト"""
        experiment_id = str(uuid.uuid4())

        db_mock = Mock()
        ga_experiment_repo_mock = Mock()

        self.db_session_factory_mock.return_value.__enter__.return_value = db_mock
        self.db_session_factory_mock.return_value.__exit__.return_value = None

        with unittest.mock.patch('app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository') as mock_registry, \
             unittest.mock.patch.object(self.service, 'get_experiment_info') as mock_get_info:

            mock_registry.return_value = ga_experiment_repo_mock
            mock_get_info.return_value = {"db_id": 123}

            self.service.stop_experiment(experiment_id)

            mock_get_info.assert_called_once_with(experiment_id)
            ga_experiment_repo_mock.update_experiment_status.assert_called_once_with(123, "stopped")


if __name__ == '__main__':
    unittest.main()
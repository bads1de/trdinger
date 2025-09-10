"""
Experiment Managerテスト

ExperimentManagerの機能をテストします。
"""

import unittest
from unittest.mock import Mock, MagicMock

from app.services.auto_strategy.services.experiment_manager import ExperimentManager
from app.services.auto_strategy.config import GAConfig


class TestExperimentManager(unittest.TestCase):
    """ExperimentManagerテスト"""

    def setUp(self):
        """セットアップ"""
        self.backtest_service_mock = Mock()
        self.persistence_service_mock = Mock()
        self.persistence_service_mock.fail_experiment = Mock()
        self.persistence_service_mock.complete_experiment = Mock()
        self.persistence_service_mock.stop_experiment = Mock()
        self.persistence_service_mock.save_experiment_result = Mock()

        self.manager = ExperimentManager(
            backtest_service=self.backtest_service_mock,
            persistence_service=self.persistence_service_mock
        )

    def test_initialize_ga_engine(self):
        """GAエンジン初期化テスト"""
        ga_config = GAConfig(
            population_size=10,
            generations=5,
            log_level="INFO"
        )

        self.manager.initialize_ga_engine(ga_config)

        self.assertIsNotNone(self.manager.ga_engine)

    def test_run_experiment_valid(self):
        """有効な実験実行テスト"""
        # GAエンジンを初期化
        ga_config = GAConfig(
            population_size=10,
            generations=5,
            log_level="INFO"
        )
        self.manager.initialize_ga_engine(ga_config)

        # mockの設定
        mock_result = {"best_fitness": 100}
        self.manager.ga_engine.run_evolution.return_value = mock_result

        experiment_id = "test_experiment"
        backtest_config = {"symbol": "BTC/USDT"}

        # テスト実行
        self.manager.run_experiment(experiment_id, ga_config, backtest_config)

        # ガエンジンが呼ばれたか確認
        self.manager.ga_engine.run_evolution.assert_called_once_with(ga_config, backtest_config)

        # 永続化サービスが呼ばれたか確認
        self.persistence_service_mock.save_experiment_result.assert_called_once_with(
            experiment_id, mock_result, ga_config, backtest_config
        )
        self.persistence_service_mock.complete_experiment.assert_called_once_with(experiment_id)

    def test_run_experiment_ga_not_initialized(self):
        """GAエンジン初期化されていない場合のテスト - RuntimeErrorが発生するか確認"""
        ga_config = GAConfig(
            population_size=10,
            generations=5,
            log_level="INFO"
        )
        experiment_id = "test_experiment"
        backtest_config = {"symbol": "BTC/USDT"}

        with self.assertRaises(RuntimeError) as context:
            self.manager.run_experiment(experiment_id, ga_config, backtest_config)

        self.assertIn("GAエンジンが初期化されていません", str(context.exception))

    def test_stop_experiment_with_ga_engine(self):
        """GAエンジンがある場合の実験停止テスト"""
        # GAエンジンを初期化
        ga_config = GAConfig(
            population_size=10,
            generations=5,
            log_level="INFO"
        )
        self.manager.initialize_ga_engine(ga_config)

        experiment_id = "test_experiment"

        # テスト実行
        result = self.manager.stop_experiment(experiment_id)

        self.assertTrue(result)
        self.manager.ga_engine.stop_evolution.assert_called_once()
        self.persistence_service_mock.stop_experiment.assert_called_once_with(experiment_id)

    def test_stop_experiment_without_ga_engine(self):
        """GAエンジンがない場合の実験停止テスト"""
        experiment_id = "test_experiment"

        # テスト実行
        result = self.manager.stop_experiment(experiment_id)

        self.assertTrue(result)
        self.persistence_service_mock.stop_experiment.assert_called_once_with(experiment_id)


if __name__ == '__main__':
    unittest.main()
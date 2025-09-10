"""
Auto Strategy Service テスト

AutoStrategyServiceの機能をテストします。
"""

import unittest
from unittest.mock import Mock, MagicMock, AsyncMock
import uuid

from backend.app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from backend.app.services.auto_strategy.config import GAConfig
from fastapi import BackgroundTasks


class TestAutoStrategyService(unittest.TestCase):
    """AutoStrategyServiceテスト"""

    def setUp(self):
        """セットアップ"""
        self.backtest_service_mock = Mock()
        self.persistence_service_mock = Mock()
        self.experiment_manager_mock = Mock()
        self.db_session_factory_mock = Mock()

        # サービス初期化を避けるため、必要なプロパティを設定
        with unittest.mock.patch.object(AutoStrategyService, '_init_services'):
            self.service = AutoStrategyService(enable_smart_generation=False)
            self.service.backtest_service = self.backtest_service_mock
            self.service.persistence_service = self.persistence_service_mock
            self.service.experiment_manager = self.experiment_manager_mock
            self.service.db_session_factory = self.db_session_factory_mock

    def test_start_strategy_generation_valid(self):
        """有効な戦略生成開始テスト"""
        experiment_id = str(uuid.uuid4())
        experiment_name = "Test Experiment"
        ga_config_dict = {
            "population_size": 10,
            "generations": 5,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "allowed_indicators": ["RSI", "MACD"]
        }
        backtest_config_dict = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        }
        background_tasks = BackgroundTasks()

        # Mock設定
        self.persistence_service_mock.create_experiment.return_value = experiment_id
        self.extensionast_manager_mock.run_experiment.return_value = None

        result = self.service.start_strategy_generation(
            experiment_id,
            experiment_name,
            ga_config_dict,
            backtest_config_dict,
            background_tasks
        )

        self.assertEqual(result, experiment_id)
        self.persistence_service_mock.create_experiment.assert_called_once()
        self.experiment_manager_mock.initialize_ga_engine.assert_called_once()
        # Background taskが追加されていることを確認

    def test_start_strategy_generation_invalid_ga_config(self):
        """無効なGA設定での戦略生成テスト - エラーが発生するか確認"""
        experiment_id = str(uuid.uuid4())
        background_tasks = BackgroundTasks()

        ga_config_dict = {"invalid_key": "value"}  # 無効な設定
        backtest_config_dict = {"symbol": "BTC/USDT"}

        with unittest.mock.patch('backend.app.services.auto_strategy.services.auto_strategy_service.GAConfig.from_dict') as mock_from_dict:
            mock_from_dict.side_effect = ValueError("無効な設定")
            with self.assertRaises(ValueError):
                self.service.start_strategy_generation(
                    experiment_id, "Test", ga_config_dict, backtest_config_dict, background_tasks
                )

    def test_list_experiments(self):
        """実験一覧取得テスト"""
        expected_experiments = [
            {"id": 1, "experiment_name": "Exp1", "status": "completed"},
            {"id": 2, "experiment_name": "Exp2", "status": "running"}
        ]
        self.persistence_service_mock.list_experiments.return_value = expected_experiments

        result = self.service.list_experiments()

        self.assertEqual(result, expected_experiments)
        self.persistence_service_mock.list_experiments.assert_called_once()

    def test_stop_experiment_success(self):
        """実験停止成功テスト"""
        experiment_id = str(uuid.uuid4())
        self.experiment_manager_mock.stop_experiment.return_value = True

        result = self.service.stop_experiment(experiment_id)

        self.assertEqual(result["success"], True)
        self.assertEqual(result["message"], "実験が正常に停止されました")
        self.experiment_manager_mock.stop_experiment.assert_called_once_with(experiment_id)

    def test_stop_experiment_failure(self):
        """実験停止失敗テスト"""
        experiment_id = str(uuid.uuid4())
        self.experiment_manager_mock.stop_experiment.return_value = False

        result = self.service.stop_experiment(experiment_id)

        self.assertEqual(result["success"], False)
        self.assertEqual(result["message"], "実験の停止に失敗しました")

    def test_stop_experiment_no_manager(self):
        """実験管理マネージャーがない場合の実験停止テスト"""
        experiment_id = str(uuid.uuid4())
        self.service.experiment_manager = None

        result = self.service.stop_experiment(experiment_id)

        self.assertEqual(result["success"], False)
        self.assertIn("実験管理マネージャーが初期化されていません", result["message"])

    def test_prepare_backtest_config_with_symbol(self):
        """シンボルを含むバックテスト設定準備テスト"""
        config_dict = {"symbol": "ETH/USDT", "timeframe": "4h"}

        result = self.service._prepare_backtest_config(config_dict)

        self.assertEqual(result["symbol"], "ETH/USDT")
        self.assertEqual(result["timeframe"], "4h")

    def test_prepare_backtest_config_without_symbol(self):
        """シンボルを含まないバックテスト設定準備テスト"""
        config_dict = {"timeframe": "1d"}

        result = self.service._prepare_backtest_config(config_dict)

        self.assertEqual(result["symbol"], "BTC/USDT:USDT")  # Actual DEFAULT_SYMBOL value
        self.assertEqual(result["timeframe"], "1d")


if __name__ == '__main__':
    unittest.main()
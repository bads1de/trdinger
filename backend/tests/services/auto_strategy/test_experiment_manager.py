"""
ExperimentManagerのテスト
"""

from unittest.mock import MagicMock, Mock, patch

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.services.experiment_manager import ExperimentManager


class TestExperimentManager:
    """ExperimentManagerのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        ExperimentManager._active_engines.clear()
        self.mock_backtest_service = Mock()
        self.mock_persistence_service = Mock()
        self.manager = ExperimentManager(
            self.mock_backtest_service, self.mock_persistence_service
        )

    def teardown_method(self):
        """テスト後の後始末"""
        ExperimentManager._active_engines.clear()

    def test_init(self):
        """初期化のテスト"""
        assert self.manager.backtest_service == self.mock_backtest_service
        assert self.manager.persistence_service == self.mock_persistence_service
        assert self.manager.ga_engine is None

    def test_initialize_ga_engine(self):
        """GAエンジン初期化のテスト"""
        ga_config = GAConfig()
        ga_config.population_size = 10
        ga_config.generations = 5

        self.manager.initialize_ga_engine(ga_config)

        assert self.manager.ga_engine is not None
        assert isinstance(self.manager.ga_engine, GeneticAlgorithmEngine)

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
        }

        # GAエンジンと永続化サービスをモック
        mock_ga_engine = MagicMock()
        mock_ga_engine.run_evolution.return_value = {"winning_individuals": []}
        mock_ga_engine.is_stop_requested.return_value = False
        self.manager.ga_engine = mock_ga_engine

        # スレッド実行ではなく直接実行される部分をテストするのは難しい（@safe_operationデコレータがあるため）
        # しかし、内部関数 _run_experiment が呼び出されることを確認したい
        # safe_operation は同期的に実行されるはず（スレッド化はこのクラスの責務外に見える）

        # run_experiment自体がスレッドまたは同期で実行するかは実装次第だが、
        # ここではモックされたga_engineのrun_evolutionが呼ばれるかを確認する

        # GAエンジンを設定（initialize_ga_engineをバイパス）
        self.manager.ga_engine = mock_ga_engine

        self.manager.run_experiment("test_exp_001", ga_config, backtest_config)

        # 検証
        expected_backtest_config = dict(backtest_config)
        expected_backtest_config["experiment_id"] = "test_exp_001"

        mock_ga_engine.run_evolution.assert_called_once_with(
            ga_config, expected_backtest_config
        )
        self.manager.persistence_service.save_experiment_result.assert_called_once_with(
            "test_exp_001",
            {"winning_individuals": []},
            ga_config,
            expected_backtest_config,
        )
        self.manager.persistence_service.complete_experiment.assert_called_once_with(
            "test_exp_001"
        )
        assert ExperimentManager._get_active_engine("test_exp_001") is None
        assert self.manager.ga_engine is None

    def test_run_experiment_exception(self):
        """実験実行中の例外テスト"""
        ga_config = GAConfig()
        backtest_config = {}

        # GAエンジンをモックして例外を発生させる
        mock_ga_engine = MagicMock()
        mock_ga_engine.run_evolution.side_effect = Exception("Test exception")
        self.manager.ga_engine = mock_ga_engine

        self.manager.run_experiment("test_exp_001", ga_config, backtest_config)

        # エラーハンドリングの検証
        self.manager.persistence_service.fail_experiment.assert_called_once_with(
            "test_exp_001"
        )
        assert ExperimentManager._get_active_engine("test_exp_001") is None
        assert self.manager.ga_engine is None

    def test_stop_experiment(self):
        """実験停止のテスト"""
        mock_ga_engine = MagicMock()
        self.manager.ga_engine = mock_ga_engine

        self.manager.stop_experiment("test_exp_001")

        mock_ga_engine.stop_evolution.assert_called_once()
        self.manager.persistence_service.stop_experiment.assert_called_once_with(
            "test_exp_001"
        )

    def test_stop_experiment_across_instances(self):
        """別インスタンスからでも実行中の実験を停止できること"""
        ga_config = GAConfig()
        ga_config.population_size = 10
        ga_config.generations = 5
        experiment_id = "test_exp_registry"
        mock_ga_engine = MagicMock()
        other_persistence = Mock()
        other_manager = ExperimentManager(self.mock_backtest_service, other_persistence)

        with patch(
            "app.services.auto_strategy.core.engine.ga_engine_factory.GeneticAlgorithmEngineFactory.create_engine",
            return_value=mock_ga_engine,
        ):
            self.manager.initialize_ga_engine(ga_config, experiment_id)

        result = other_manager.stop_experiment(experiment_id)

        assert result is True
        mock_ga_engine.stop_evolution.assert_called_once()
        other_persistence.stop_experiment.assert_called_once_with(experiment_id)



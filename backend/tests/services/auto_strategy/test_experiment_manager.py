"""
ExperimentManagerのテスト
"""

from unittest.mock import MagicMock, Mock

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.services.experiment_manager import ExperimentManager


class TestExperimentManager:
    """ExperimentManagerのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.mock_backtest_service = Mock()
        self.mock_persistence_service = Mock()
        self.manager = ExperimentManager(
            self.mock_backtest_service, self.mock_persistence_service
        )

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
        mock_ga_engine.run_evolution.assert_called_once_with(ga_config, backtest_config)
        self.manager.persistence_service.save_experiment_result.assert_called_once()
        self.manager.persistence_service.complete_experiment.assert_called_once_with(
            "test_exp_001"
        )

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

    def test_stop_experiment(self):
        """実験停止のテスト"""
        mock_ga_engine = MagicMock()
        self.manager.ga_engine = mock_ga_engine

        self.manager.stop_experiment("test_exp_001")

        mock_ga_engine.stop_evolution.assert_called_once()
        self.manager.persistence_service.stop_experiment.assert_called_once_with(
            "test_exp_001"
        )





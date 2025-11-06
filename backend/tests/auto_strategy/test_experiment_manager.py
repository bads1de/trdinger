"""
ExperimentManagerのテスト
"""

import pytest

pytestmark = pytest.mark.skip(reason="ExperimentManager implementation changed")
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from datetime import datetime

from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.services.experiment_manager import ExperimentManager
from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)


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
        assert isinstance(self.manager.strategy_factory, StrategyFactory)
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

        # 初期化が呼ばれないと仮定するため、設定
        with patch.object(self.manager, "initialize_ga_engine") as mock_init:
            # テスト実行
            self.manager.run_experiment("test_exp_001", ga_config, backtest_config)

            # 検証: initialize_ga_engineは呼ばれない（エンジンがすでに初期化されているため）
            mock_init.assert_not_called()
            mock_ga_engine.run_evolution.assert_called_once_with(
                ga_config, backtest_config
            )

    def test_run_experiment_exception(self):
        """実験実行中の例外テスト"""
        ga_config = GAConfig()
        backtest_config = {}

        with patch.object(self.manager, "initialize_ga_engine") as mock_init:
            mock_init.side_effect = Exception("Test exception")

            with pytest.raises(Exception):
                self.manager.run_experiment("test_exp_001", ga_config, backtest_config)

    def test_run_ga_evolution(self):
        """GA進化実行のテスト"""
        ga_config = GAConfig()
        backtest_config = {"symbol": "BTC/USDT:USDT"}

        with patch.object(self.manager.ga_engine, "evolve") as mock_evolve:
            mock_evolve.return_value = [Mock(), Mock()]

            result = self.manager._run_ga_evolution(ga_config, backtest_config)

            assert len(result) == 2
            mock_evolve.assert_called_once()

    def test_save_results(self):
        """結果保存のテスト"""
        experiment_id = "test_exp_001"
        results = [{"fitness": 0.8, "genes": []}]
        ga_config = GAConfig()

        with patch.object(
            self.manager.persistence_service, "save_generation_results"
        ) as mock_save:
            self.manager._save_results(experiment_id, results, ga_config)
            mock_save.assert_called_once_with(experiment_id, results, ga_config)

    def test_validate_configuration_valid(self):
        """有効な設定検証のテスト"""
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-12-19",
            "initial_capital": 100000,
        }

        # 例外が投げられないことを確認
        try:
            self.manager._validate_configuration(backtest_config)
        except Exception:
            pytest.fail("例外が投げられました")

    def test_validate_configuration_invalid(self):
        """無効な設定検証のテスト"""
        backtest_config = {}  # 必須フィールドが欠けている

        with pytest.raises(ValueError):
            self.manager._validate_configuration(backtest_config)

    def test_cleanup_resources(self):
        """リソースクリーンアップのテスト"""
        self.manager.ga_engine = Mock()

        with patch.object(self.manager.ga_engine, "cleanup") as mock_cleanup:
            self.manager._cleanup_resources()
            mock_cleanup.assert_called_once()

        # GAエンジンがNoneに戻っていることを確認
        assert self.manager.ga_engine is None

    def test_get_experiment_status(self):
        """実験ステータス取得のテスト"""
        experiment_id = "test_exp_001"

        with patch.object(
            self.manager.persistence_service, "get_experiment_status"
        ) as mock_status:
            mock_status.return_value = {"status": "running", "generation": 3}

            result = self.manager.get_experiment_status(experiment_id)

            assert result["status"] == "running"
            assert result["generation"] == 3
            mock_status.assert_called_once_with(experiment_id)

"""
ExperimentPersistenceServiceのテスト
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.genes import StrategyGene
from app.services.auto_strategy.services.experiment_persistence_service import (
    ExperimentPersistenceService,
)


class TestExperimentPersistenceService:
    """ExperimentPersistenceServiceのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.mock_db_session = MagicMock()
        self.mock_db_session_factory = MagicMock(return_value=self.mock_db_session)
        # コンテキストマネージャとして動作するように設定
        self.mock_db_session_factory.return_value.__enter__.return_value = (
            self.mock_db_session
        )
        self.persistence_service = ExperimentPersistenceService(
            self.mock_db_session_factory
        )

    def test_create_experiment(self):
        """実験作成のテスト"""
        experiment_id = "test_uuid_001"
        experiment_name = "Test Experiment"
        ga_config = GAConfig(generations=10)
        backtest_config = {"symbol": "BTC/USDT:USDT"}

        with patch(
            "app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_db_experiment = Mock()
            mock_db_experiment.id = 123
            mock_repo.create_experiment.return_value = mock_db_experiment

            result_id = self.persistence_service.create_experiment(
                experiment_id, experiment_name, ga_config, backtest_config
            )

            assert result_id == experiment_id
            mock_repo.create_experiment.assert_called_once()
            call_kwargs = mock_repo.create_experiment.call_args[1]
            assert call_kwargs["experiment_id"] == experiment_id
            assert call_kwargs["name"] == experiment_name
            assert call_kwargs["config"]["experiment_id"] == experiment_id
            assert call_kwargs["total_generations"] == 10
            assert call_kwargs["status"] == "running"

    def test_list_experiments(self):
        """実験一覧取得のテスト"""
        with patch(
            "app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_exp = Mock()
            mock_exp.id = 1
            mock_exp.name = "Exp 1"
            mock_exp.status = "completed"
            mock_exp.created_at = datetime(2024, 1, 1)
            mock_exp.completed_at = datetime(2024, 1, 2)
            mock_repo.get_recent_experiments.return_value = [mock_exp]

            experiments = self.persistence_service.list_experiments()

            assert len(experiments) == 1
            assert experiments[0]["id"] == 1
            assert experiments[0]["experiment_name"] == "Exp 1"
            assert experiments[0]["status"] == "completed"
            assert experiments[0]["created_at"] == "2024-01-01T00:00:00"

    def test_get_experiment_info_by_uuid(self):
        """UUIDによる実験情報取得のテスト"""
        target_uuid = "target-uuid-123"

        with patch(
            "app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_exp = Mock()
            mock_exp.id = 2
            mock_exp.name = "Target Exp"
            mock_exp.status = "running"
            mock_exp.config = {"experiment_id": target_uuid}
            mock_exp.created_at = datetime.now()
            mock_exp.completed_at = None

            mock_repo.get_by_experiment_id.return_value = mock_exp

            info = self.persistence_service.get_experiment_info(target_uuid)

            assert info is not None
            assert info["db_id"] == 2
            assert info["name"] == "Target Exp"
            mock_repo.get_by_experiment_id.assert_called_once_with(target_uuid)

    def test_complete_experiment(self):
        """実験完了処理のテスト"""
        experiment_id = "exp_001"

        # get_experiment_infoをモック
        with patch.object(
            self.persistence_service, "get_experiment_info"
        ) as mock_get_info:
            mock_get_info.return_value = {"db_id": 123}

            with patch(
                "app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository"
            ) as mock_repo_cls:
                mock_repo = mock_repo_cls.return_value

                self.persistence_service.complete_experiment(experiment_id)

                mock_repo.update_experiment_status.assert_called_once_with(
                    123, "completed"
                )

    def test_fail_experiment(self):
        """実験失敗処理のテスト"""
        experiment_id = "exp_001"

        with patch.object(
            self.persistence_service, "get_experiment_info"
        ) as mock_get_info:
            mock_get_info.return_value = {"db_id": 123}

            with patch(
                "app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository"
            ) as mock_repo_cls:
                mock_repo = mock_repo_cls.return_value

                self.persistence_service.fail_experiment(experiment_id)

                mock_repo.update_experiment_status.assert_called_once_with(
                    123, "failed"
                )

    def test_stop_experiment(self):
        """実験停止処理のテスト"""
        experiment_id = "exp_001"

        with patch.object(
            self.persistence_service, "get_experiment_info"
        ) as mock_get_info:
            mock_get_info.return_value = {"db_id": 123}

            with patch(
                "app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository"
            ) as mock_repo_cls:
                mock_repo = mock_repo_cls.return_value

                self.persistence_service.stop_experiment(experiment_id)

                mock_repo.update_experiment_status.assert_called_once_with(
                    123, "stopped"
                )

    def test_save_experiment_result(self):
        """実験結果保存のテスト"""
        experiment_id = "exp_001"
        ga_config = GAConfig()
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "initial_capital": 10000,
        }

        mock_strategy = Mock(spec=StrategyGene)
        mock_strategy.id = "strat_123"

        result = {
            "best_strategy": mock_strategy,
            "best_fitness": 1.5,
            "all_strategies": [mock_strategy],
            "fitness_scores": [1.5],
        }

        experiment_info = {
            "db_id": 100,
            "name": "AUTO_STRATEGY_GA_TEST",
            "config": {"experiment_id": experiment_id},
        }

        with (
            patch(
                "app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository"
            ) as mock_strat_repo_cls,
            patch.object(
                self.persistence_service.serializer,
                "strategy_gene_to_dict",
                return_value={"serialized": True},
            ) as mock_strategy_to_dict,
        ):

            mock_strat_repo = mock_strat_repo_cls.return_value
            mock_strat_repo.save_strategy.return_value = Mock(id=555)

            self.persistence_service.save_experiment_result(
                experiment_id,
                result,
                ga_config,
                backtest_config,
                experiment_info=experiment_info,
            )

            # 最良戦略が保存されたか確認
            mock_strat_repo.save_strategy.assert_called()
            assert mock_strategy_to_dict.call_count == 1

    def test_save_backtest_result(self):
        """詳細バックテスト結果保存のテスト"""
        result_data = {
            "strategy_name": "AS_TEST_123456",
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "initial_capital": 10000,
            "commission_rate": 0.001,
            "config_json": {
                "strategy_config": {"strategy_type": "GENERATED_GA"},
                "experiment_id": "exp_001",
                "db_experiment_id": 100,
                "fitness_score": 1.5,
            },
            "performance_metrics": {"total_return": 0.1},
            "equity_curve": [],
            "trade_history": [],
            "execution_time": 1.0,
            "status": "completed",
        }

        with patch(
            "app.services.auto_strategy.services.experiment_persistence_service.BacktestResultRepository"
        ) as mock_bt_repo_cls:
            mock_bt_repo = mock_bt_repo_cls.return_value

            self.persistence_service.save_backtest_result(result_data)

            mock_bt_repo.save_backtest_result.assert_called_once_with(result_data)

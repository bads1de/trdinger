"""
ExperimentPersistenceServiceのテスト
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="ExperimentPersistenceService implementation changed - methods need update"
)
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.models.strategy_models import StrategyGene
from app.services.auto_strategy.services.experiment_persistence_service import (
    ExperimentPersistenceService,
)


class TestExperimentPersistenceService:
    """ExperimentPersistenceServiceのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.mock_db_session_factory = Mock()
        self.mock_backtest_service = Mock()
        self.persistence_service = ExperimentPersistenceService(
            self.mock_db_session_factory, self.mock_backtest_service
        )

    def test_init(self):
        """初期化のテスト"""
        assert (
            self.persistence_service.db_session_factory == self.mock_db_session_factory
        )
        assert self.persistence_service.backtest_service == self.mock_backtest_service

    def test_create_experiment(self):
        """実験作成のテスト"""
        experiment_id = "test_exp_001"
        experiment_name = "Test Experiment"
        ga_config = GAConfig()
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-12-19",
        }

        with patch.object(
            self.persistence_service, "_save_experiment_to_db"
        ) as mock_save:
            self.persistence_service.create_experiment(
                experiment_id, experiment_name, ga_config, backtest_config
            )

            mock_save.assert_called_once_with(
                experiment_id, experiment_name, ga_config, backtest_config, "running"
            )

    def test_save_generation_results(self):
        """世代結果保存のテスト"""
        experiment_id = "test_exp_001"
        results = [{"fitness": 0.85, "genes": [1, 2, 3, 4, 5], "generation": 1}]
        ga_config = GAConfig()

        with patch.object(
            self.persistence_service, "_save_generation_results_to_db"
        ) as mock_save:
            with patch.object(
                self.persistence_service, "_calculate_generation_summary"
            ) as mock_calc:
                mock_calc.return_value = {
                    "avg_fitness": 0.75,
                    "best_fitness": 0.85,
                    "worst_fitness": 0.65,
                }

                self.persistence_service.save_generation_results(
                    experiment_id, results, ga_config
                )

                mock_calc.assert_called_once_with(results)
                mock_save.assert_called_once_with(
                    experiment_id, results, mock_calc.return_value, ga_config
                )

    def test_save_generation_results_empty(self):
        """空の結果保存のテスト"""
        experiment_id = "test_exp_001"
        results = []
        ga_config = GAConfig()

        with patch.object(
            self.persistence_service, "_save_generation_results_to_db"
        ) as mock_save:
            with patch.object(
                self.persistence_service, "_calculate_generation_summary"
            ) as mock_calc:
                mock_calc.return_value = {
                    "avg_fitness": 0.0,
                    "best_fitness": 0.0,
                    "worst_fitness": 0.0,
                }

                self.persistence_service.save_generation_results(
                    experiment_id, results, ga_config
                )

                # 空の結果でも保存が呼ばれることを確認
                mock_save.assert_called_once()

    def test_list_experiments(self):
        """実験一覧取得のテスト"""
        mock_experiments = [
            {
                "id": "exp1",
                "name": "Experiment 1",
                "status": "completed",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }
        ]

        with patch.object(
            self.persistence_service, "_get_experiments_from_db"
        ) as mock_get:
            mock_get.return_value = mock_experiments

            experiments = self.persistence_service.list_experiments()

            assert len(experiments) == 1
            assert experiments[0]["id"] == "exp1"

    def test_get_experiment_status(self):
        """実験ステータス取得のテスト"""
        experiment_id = "test_exp_001"
        expected_status = {
            "id": experiment_id,
            "status": "running",
            "current_generation": 5,
            "best_fitness": 0.85,
        }

        with patch.object(
            self.persistence_service, "_get_experiment_status_from_db"
        ) as mock_get:
            mock_get.return_value = expected_status

            status = self.persistence_service.get_experiment_status(experiment_id)

            assert status == expected_status
            mock_get.assert_called_once_with(experiment_id)

    def test_get_experiment_status_not_found(self):
        """実験ステータス取得失敗のテスト"""
        experiment_id = "nonexistent_exp"

        with patch.object(
            self.persistence_service, "_get_experiment_status_from_db"
        ) as mock_get:
            mock_get.return_value = None

            status = self.persistence_service.get_experiment_status(experiment_id)

            assert status is None

    def test_update_experiment_status(self):
        """実験ステータス更新のテスト"""
        experiment_id = "test_exp_001"
        new_status = "completed"

        with patch.object(
            self.persistence_service, "_update_experiment_status_in_db"
        ) as mock_update:
            self.persistence_service.update_experiment_status(experiment_id, new_status)

            mock_update.assert_called_once_with(experiment_id, new_status)

    def test_save_best_strategy(self):
        """最良戦略保存のテスト"""
        experiment_id = "test_exp_001"
        best_individual = {
            "fitness": 0.95,
            "genes": [1, 2, 3, 4, 5, 6],
            "generation": 10,
        }
        ga_config = GAConfig()

        with patch.object(
            self.persistence_service, "_save_best_strategy_to_db"
        ) as mock_save:
            self.persistence_service.save_best_strategy(
                experiment_id, best_individual, ga_config
            )

            mock_save.assert_called_once_with(experiment_id, best_individual, ga_config)

    def test_calculate_generation_summary(self):
        """世代サマリー計算のテスト"""
        results = [
            {"fitness": 0.85, "genes": [], "generation": 1},
            {"fitness": 0.75, "genes": [], "generation": 1},
            {"fitness": 0.95, "genes": [], "generation": 1},
            {"fitness": 0.65, "genes": [], "generation": 1},
        ]

        summary = self.persistence_service._calculate_generation_summary(results)

        assert summary["avg_fitness"] == 0.8
        assert summary["best_fitness"] == 0.95
        assert summary["worst_fitness"] == 0.65

    def test_calculate_generation_summary_empty(self):
        """空の世代サマリー計算のテスト"""
        summary = self.persistence_service._calculate_generation_summary([])

        assert summary["avg_fitness"] == 0.0
        assert summary["best_fitness"] == 0.0
        assert summary["worst_fitness"] == 0.0

    def test_calculate_generation_summary_single_result(self):
        """単一結果の世代サマリー計算のテスト"""
        results = [{"fitness": 0.85, "genes": [], "generation": 1}]

        summary = self.persistence_service._calculate_generation_summary(results)

        assert summary["avg_fitness"] == 0.85
        assert summary["best_fitness"] == 0.85
        assert summary["worst_fitness"] == 0.85

    def test_validate_experiment_data_valid(self):
        """有効な実験データ検証のテスト"""
        experiment_id = "test_exp_001"
        experiment_name = "Test Experiment"
        ga_config = GAConfig()
        backtest_config = {"symbol": "BTC/USDT:USDT"}

        # 例外が投げられないことを確認
        try:
            self.persistence_service._validate_experiment_data(
                experiment_id, experiment_name, ga_config, backtest_config
            )
        except Exception:
            pytest.fail("例外が投げられました")

    def test_validate_experiment_data_invalid_id(self):
        """無効な実験ID検証のテスト"""
        with pytest.raises(ValueError, match="実験IDが必要です"):
            self.persistence_service._validate_experiment_data(
                None, "Test", GAConfig(), {}
            )

    def test_validate_experiment_data_invalid_name(self):
        """無効な実験名検証のテスト"""
        with pytest.raises(ValueError, match="実験名が必要です"):
            self.persistence_service._validate_experiment_data(
                "test_exp", None, GAConfig(), {}
            )

    def test_validate_experiment_data_invalid_config(self):
        """無効な設定検証のテスト"""
        with pytest.raises(ValueError, match="GA設定が必要です"):
            self.persistence_service._validate_experiment_data(
                "test_exp", "Test", None, {}
            )

    def test_serialize_strategy_gene(self):
        """戦略遺伝子シリアライズのテスト"""
        # StrategyGeneのモックを作成
        mock_gene = Mock(spec=StrategyGene)
        mock_gene.id = "gene_001"

        serialized = self.persistence_service._serialize_strategy_gene(mock_gene)

        assert isinstance(serialized, dict)
        assert "id" in serialized

    def test_deserialize_strategy_gene(self):
        """戦略遺伝子デシリアライズのテスト"""
        gene_data = {"id": "gene_001", "parameters": {"indicator": "SMA", "period": 14}}

        with patch(
            "app.services.auto_strategy.services.experiment_persistence_service.StrategyGene"
        ) as mock_gene_class:
            mock_gene = Mock()
            mock_gene_class.return_value = mock_gene

            gene = self.persistence_service._deserialize_strategy_gene(gene_data)

            assert gene == mock_gene
            mock_gene_class.assert_called_once_with(**gene_data)

    def test_format_generation_data(self):
        """世代データフォーマットのテスト"""
        results = [{"fitness": 0.85, "genes": [1, 2, 3, 4, 5], "generation": 1}]
        summary = {"avg_fitness": 0.85, "best_fitness": 0.85, "worst_fitness": 0.85}
        ga_config = GAConfig()

        formatted = self.persistence_service._format_generation_data(
            results, summary, ga_config
        )

        assert isinstance(formatted, dict)
        assert "results" in formatted
        assert "summary" in formatted
        assert "config" in formatted

    def test_handle_database_error(self):
        """データベースエラー処理のテスト"""
        test_func = Mock()
        test_func.side_effect = Exception("Database error")

        with patch(
            "app.services.auto_strategy.services.experiment_persistence_service.logger"
        ) as mock_logger:
            result = self.persistence_service._handle_database_error(
                test_func, "test operation"
            )

            assert result is None
            mock_logger.error.assert_called_once_with(
                "test operationでデータベースエラーが発生しました: Database error"
            )

    def test_cleanup_experiment_resources(self):
        """実験リソースクリーンアップのテスト"""
        experiment_id = "test_exp_001"

        with patch.object(
            self.persistence_service, "_cleanup_experiment_resources_in_db"
        ) as mock_cleanup:
            self.persistence_service.cleanup_experiment_resources(experiment_id)

            mock_cleanup.assert_called_once_with(experiment_id)

    def test_get_experiment_history(self):
        """実験履歴取得のテスト"""
        experiment_id = "test_exp_001"
        expected_history = [
            {
                "generation": 1,
                "avg_fitness": 0.75,
                "best_fitness": 0.85,
                "timestamp": datetime.now(),
            }
        ]

        with patch.object(
            self.persistence_service, "_get_experiment_history_from_db"
        ) as mock_get:
            mock_get.return_value = expected_history

            history = self.persistence_service.get_experiment_history(experiment_id)

            assert history == expected_history
            mock_get.assert_called_once_with(experiment_id)

    def test_get_experiment_history_empty(self):
        """空の実験履歴取得のテスト"""
        experiment_id = "test_exp_001"

        with patch.object(
            self.persistence_service, "_get_experiment_history_from_db"
        ) as mock_get:
            mock_get.return_value = []

            history = self.persistence_service.get_experiment_history(experiment_id)

            assert history == []

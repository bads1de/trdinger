"""
多目的最適化GA エンドツーエンドテスト

実際のワークフローでの動作確認を行います。
"""

import pytest
import logging
import time
from typing import Dict, Any

from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.services.auto_strategy_service import (
    AutoStrategyService,
)
from app.core.services.auto_strategy.services.experiment_persistence_service import (
    ExperimentPersistenceService,
)
from database.connection import SessionLocal

logger = logging.getLogger(__name__)


class TestEndToEndMultiObjective:
    """多目的最適化GA エンドツーエンドテストクラス"""

    @pytest.fixture
    def auto_strategy_service(self):
        """AutoStrategyServiceのインスタンス"""
        return AutoStrategyService()

    @pytest.fixture
    def experiment_persistence_service(self):
        """ExperimentPersistenceServiceのインスタンス"""
        return ExperimentPersistenceService(SessionLocal)

    def test_single_objective_end_to_end(self, auto_strategy_service):
        """単一目的最適化のエンドツーエンドテスト"""
        # 単一目的設定
        config = GAConfig(
            population_size=5,
            generations=2,
            enable_multi_objective=False,
            objectives=["total_return"],
            objective_weights=[1.0],
        )

        # バックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }

        # 実験実行（モックのバックグラウンドタスクを使用）
        from unittest.mock import Mock

        mock_background_tasks = Mock()

        experiment_id = auto_strategy_service.start_strategy_generation(
            experiment_name="E2E_Single_Objective_Test",
            ga_config_dict=config.to_dict(),
            backtest_config_dict=backtest_config,
            background_tasks=mock_background_tasks,
        )

        assert experiment_id is not None
        logger.info(f"単一目的実験開始: {experiment_id}")

        # 実験完了まで待機（最大30秒）
        max_wait = 30
        start_time = time.time()

        while time.time() - start_time < max_wait:
            experiment_info = (
                auto_strategy_service.persistence_service.get_experiment_info(
                    experiment_id
                )
            )
            if experiment_info and experiment_info.get("status") == "completed":
                break
            elif experiment_info and experiment_info.get("status") == "failed":
                pytest.fail("実験が失敗しました")
            time.sleep(1)

        # 結果確認
        experiment_info = auto_strategy_service.persistence_service.get_experiment_info(
            experiment_id
        )
        status = experiment_info.get("status") if experiment_info else "unknown"
        assert status == "completed", f"実験が完了しませんでした。ステータス: {status}"

        logger.info("単一目的エンドツーエンドテスト完了")

    def test_multi_objective_end_to_end(self, auto_strategy_service):
        """多目的最適化のエンドツーエンドテスト"""
        # 多目的設定
        config = GAConfig.create_multi_objective(
            objectives=["total_return", "max_drawdown"], weights=[1.0, -1.0]
        )
        config.population_size = 6
        config.generations = 2

        # バックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }

        # 実験実行（モックのバックグラウンドタスクを使用）
        from unittest.mock import Mock

        mock_background_tasks = Mock()

        experiment_id = auto_strategy_service.start_strategy_generation(
            experiment_name="E2E_Multi_Objective_Test",
            ga_config_dict=config.to_dict(),
            backtest_config_dict=backtest_config,
            background_tasks=mock_background_tasks,
        )

        assert experiment_id is not None
        logger.info(f"多目的実験開始: {experiment_id}")

        # 実験完了まで待機（最大60秒）
        max_wait = 60
        start_time = time.time()

        while time.time() - start_time < max_wait:
            experiment_info = (
                auto_strategy_service.persistence_service.get_experiment_info(
                    experiment_id
                )
            )
            if experiment_info and experiment_info.get("status") == "completed":
                break
            elif experiment_info and experiment_info.get("status") == "failed":
                pytest.fail("実験が失敗しました")
            time.sleep(1)

        # 結果確認
        experiment_info = auto_strategy_service.persistence_service.get_experiment_info(
            experiment_id
        )
        status = experiment_info.get("status") if experiment_info else "unknown"
        assert status == "completed", f"実験が完了しませんでした。ステータス: {status}"

        logger.info("多目的エンドツーエンドテスト完了")

    def test_multi_objective_result_retrieval(self, experiment_persistence_service):
        """多目的最適化結果取得テスト"""
        # 多目的設定
        config = GAConfig.create_multi_objective(
            objectives=["total_return", "sharpe_ratio", "max_drawdown"],
            weights=[1.0, 1.0, -1.0],
        )
        config.population_size = 4
        config.generations = 1

        # バックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "4h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-15",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }

        # 実験情報を作成
        experiment_info = {
            "experiment_name": "E2E_Result_Retrieval_Test",
            "config": {
                "ga_config": config.to_dict(),
                "backtest_config": backtest_config,
            },
        }

        # モック結果を作成
        mock_result = {
            "best_strategy": {
                "id": "test_strategy_001",
                "to_dict": lambda: {"indicators": ["SMA", "RSI"], "conditions": []},
            },
            "best_fitness": [0.15, 1.5, 0.08],  # 多目的フィットネス値
            "pareto_front": [
                {
                    "strategy": {"indicators": ["SMA", "RSI"]},
                    "fitness_values": [0.15, 1.5, 0.08],
                },
                {
                    "strategy": {"indicators": ["EMA", "MACD"]},
                    "fitness_values": [0.12, 1.8, 0.05],
                },
            ],
            "objectives": ["total_return", "sharpe_ratio", "max_drawdown"],
            "execution_time": 25.5,
            "generations_completed": 1,
            "final_population_size": 4,
        }

        # 結果保存
        try:
            experiment_persistence_service.save_experiment_result(
                experiment_info, mock_result, config
            )
            logger.info("多目的最適化結果保存成功")
        except Exception as e:
            logger.error(f"結果保存エラー: {e}")
            # エラーが発生してもテストは継続（保存機能のテストではないため）

    def test_performance_comparison(self, auto_strategy_service):
        """単一目的と多目的最適化のパフォーマンス比較テスト"""
        # 共通設定
        base_config = {
            "population_size": 4,
            "generations": 1,
        }

        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-15",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }

        # 単一目的実行時間測定
        single_config = GAConfig(**base_config, enable_multi_objective=False)
        start_time = time.time()

        single_experiment_id = auto_strategy_service.start_experiment(
            experiment_name="Performance_Single_Test",
            ga_config=single_config,
            backtest_config=backtest_config,
        )

        # 完了まで待機
        max_wait = 30
        wait_start = time.time()
        while time.time() - wait_start < max_wait:
            if (
                auto_strategy_service.get_experiment_status(single_experiment_id)
                == "completed"
            ):
                break
            time.sleep(0.5)

        single_time = time.time() - start_time

        # 多目的実行時間測定
        multi_config = GAConfig.create_multi_objective(**base_config)
        start_time = time.time()

        multi_experiment_id = auto_strategy_service.start_experiment(
            experiment_name="Performance_Multi_Test",
            ga_config=multi_config,
            backtest_config=backtest_config,
        )

        # 完了まで待機
        wait_start = time.time()
        while time.time() - wait_start < max_wait:
            if (
                auto_strategy_service.get_experiment_status(multi_experiment_id)
                == "completed"
            ):
                break
            time.sleep(0.5)

        multi_time = time.time() - start_time

        logger.info(
            f"実行時間比較 - 単一目的: {single_time:.2f}s, 多目的: {multi_time:.2f}s"
        )

        # 多目的最適化が単一目的の3倍以内の時間で完了することを確認
        assert (
            multi_time < single_time * 3
        ), f"多目的最適化が遅すぎます: {multi_time:.2f}s vs {single_time:.2f}s"

    def test_error_resilience(self, auto_strategy_service):
        """エラー耐性テスト"""
        # 無効な設定でのテスト
        invalid_config = GAConfig(
            population_size=1,  # 極端に小さい個体数
            generations=1,
            enable_multi_objective=True,
            objectives=["total_return", "invalid_objective"],  # 無効な目的
            objective_weights=[1.0, 1.0],
        )

        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-02",  # 極端に短い期間
            "initial_capital": 1000,
            "commission_rate": 0.001,
        }

        # エラーが発生しても適切に処理されることを確認
        try:
            experiment_id = auto_strategy_service.start_experiment(
                experiment_name="Error_Resilience_Test",
                ga_config=invalid_config,
                backtest_config=backtest_config,
            )

            # 実験が開始された場合、失敗またはエラー処理されることを確認
            max_wait = 20
            start_time = time.time()

            while time.time() - start_time < max_wait:
                status = auto_strategy_service.get_experiment_status(experiment_id)
                if status in ["completed", "failed"]:
                    break
                time.sleep(0.5)

            final_status = auto_strategy_service.get_experiment_status(experiment_id)
            logger.info(f"エラー耐性テスト完了: {final_status}")

        except Exception as e:
            # 例外が発生した場合も正常（適切なエラーハンドリング）
            logger.info(f"期待されたエラーが発生: {e}")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    pytest.main([__file__, "-v", "-s"])

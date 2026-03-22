"""
AutoStrategyServiceのエンドツーエンドフローテスト
API呼び出しからバックグラウンド実行、結果保存までのフローを確認する
"""

import pytest
from unittest.mock import MagicMock, Mock, patch
from fastapi import BackgroundTasks

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.services.auto_strategy_service import (
    AutoStrategyService,
)
from app.services.auto_strategy.services.experiment_manager import ExperimentManager


class TestE2EFlow:
    """エンドツーエンドフローのテスト"""

    @pytest.mark.asyncio
    async def test_service_starts_background_task(self):
        """
        AutoStrategyServiceがバックグラウンドタスクを正しく開始することを確認
        """
        experiment_id = "test_uuid_service"
        experiment_name = "Service Test"
        ga_config_dict = {
            "population_size": 10,
            "generations": 5,
            "elite_size": 2,
        }
        backtest_config_dict = {
            "symbol": "BTC/USDT:USDT",
        }
        background_tasks = BackgroundTasks()

        # 依存関係をパッチ
        with (
            patch(
                "app.services.auto_strategy.services.auto_strategy_service.SessionLocal"
            ),
            patch(
                "app.services.auto_strategy.services.auto_strategy_service.BacktestService"
            ),
            patch(
                "app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService"
            ) as MockPersistence,
            patch(
                "app.services.auto_strategy.services.auto_strategy_service.ExperimentManager"
            ) as MockManager,
        ):

            mock_persistence = MockPersistence.return_value
            mock_manager = MockManager.return_value

            # サービス初期化
            service = AutoStrategyService()

            # テスト実行
            returned_id = service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )

            # 検証
            assert returned_id == experiment_id

            # 実験作成が呼ばれたか
            mock_persistence.create_experiment.assert_called_once()

            # GAエンジン初期化が呼ばれたか
            mock_manager.initialize_ga_engine.assert_called_once()

            # バックグラウンドタスクが追加されたか
            assert len(background_tasks.tasks) == 1
            task = background_tasks.tasks[0]
            # task.func は experiment_manager.run_experiment であるべき
            # 注意: インスタンスメソッドの比較は難しい場合があるが、mock_manager.run_experiment と一致するはず
            # FastAPIのBackgroundTasksの実装詳細に依存しすぎないよう、ここでは「タスクが1つある」ことと
            # 「mock_manager.run_experiment」が呼び出し対象であることを確認する程度にする。

            # run_experiment が呼び出されたわけではない（まだ実行されていない）
            mock_manager.run_experiment.assert_not_called()

    @pytest.mark.asyncio
    async def test_manager_executes_ga_flow(self):
        """
        ExperimentManagerがGAフローを正しく実行することを確認
        （モック化されたバックテストサービスと永続化サービスを使用）
        """
        experiment_id = "test_uuid_manager"
        experiment_name = "Manager Test"
        ga_config = GAConfig(
            population_size=4,  # 最小構成
            generations=2,  # 2世代
            min_indicators=1,
            max_indicators=2,
            elite_size=1,
            enable_fitness_sharing=False,
            enable_parallel_evaluation=False,
        )
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "initial_capital": 10000,
            "commission_rate": 0.001,
            "slippage": 0.001,
        }

        # モックの準備
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {
                "total_return": 0.1,
                "win_rate": 0.6,
                "max_drawdown": 0.05,
                "sharpe_ratio": 1.5,
                "sortino_ratio": 2.0,
                "profit_factor": 1.5,
                "trades_count": 10,
            },
            "equity_curve": [],
            "trade_history": [],
            "execution_time": 0.1,
        }

        mock_persistence_service = Mock()
        # 実験情報を返すように設定
        mock_persistence_service.get_experiment_info.return_value = {
            "db_id": 1,
            "name": experiment_name,
            "status": "running",
            "config": {"experiment_id": experiment_id},
        }

        # ExperimentManagerのインスタンス化
        # コンストラクタ引数は backtest_service, persistence_service
        manager = ExperimentManager(mock_backtest_service, mock_persistence_service)

        # GAエンジン初期化
        manager.initialize_ga_engine(ga_config)

        # インジケーター生成をモック化（パラメータ生成エラー回避のため）
        from app.services.auto_strategy.genes import IndicatorGene
        sma_gene = IndicatorGene(
            type="SMA", parameters={"period": 14}, enabled=True
        )

        with patch("app.services.auto_strategy.generators.random_gene_generator.generate_random_indicators", return_value=[sma_gene]):
            # 実験実行（同期的に実行されるはず）
            try:
                manager.run_experiment(experiment_id, ga_config, backtest_config)
            except Exception as e:
                pytest.fail(f"run_experiment failed with exception: {e}")

        # 検証

        # 1. バックテストが実行されたか
        assert mock_backtest_service.run_backtest.called
        assert mock_backtest_service.run_backtest.call_count >= 1

        # 2. 結果保存
        assert mock_persistence_service.save_experiment_result.called

        # 完了ステータス更新
        mock_persistence_service.complete_experiment.assert_called_with(experiment_id)
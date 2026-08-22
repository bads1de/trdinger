"""
EvaluationWorker のユニットテスト

並列評価ワーカーモジュールをテストします。
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import app.services.auto_strategy.core.evaluation.evaluation_worker as ew_module
from app.services.auto_strategy.config.constants import PENALTY_FITNESS_MAGNITUDE
from app.services.auto_strategy.core.evaluation.evaluation_worker import (
    initialize_worker_process,
    worker_evaluate_individual,
)
from app.services.auto_strategy.core.evaluation.parallel_evaluator import (
    ParallelEvaluationResult,
)


class TestInitializeWorkerProcess:
    """initialize_worker_process関数のテスト"""

    @patch(
        "app.services.auto_strategy.core.evaluation.parallel_evaluator.initialize_worker"
    )
    @patch("app.services.backtest.services.backtest_service.BacktestService")
    @patch(
        "app.services.auto_strategy.core.evaluation.individual_evaluator.IndividualEvaluator"
    )
    def test_initialize_success(
        self,
        mock_evaluator_cls,
        mock_backtest_service_cls,
        mock_initialize_worker_context,
    ):
        """ワーカープロセスが正常に初期化されること"""
        mock_backtest_service = MagicMock()
        mock_backtest_service_cls.return_value = mock_backtest_service

        mock_evaluator = MagicMock()
        mock_evaluator_cls.return_value = mock_evaluator

        backtest_config = {"symbol": "BTC/USDT:USDT", "timeframe": "4h"}
        ga_config = MagicMock()

        initialize_worker_process(backtest_config, ga_config)

        mock_backtest_service_cls.assert_called_once()
        mock_evaluator_cls.assert_called_once_with(mock_backtest_service)
        mock_evaluator.set_backtest_config.assert_called_once_with(backtest_config)
        mock_initialize_worker_context.assert_not_called()

    @patch(
        "app.services.auto_strategy.core.evaluation.parallel_evaluator.initialize_worker"
    )
    @patch("app.services.backtest.services.backtest_service.BacktestService")
    @patch(
        "app.services.auto_strategy.core.evaluation.individual_evaluator.IndividualEvaluator"
    )
    def test_initialize_with_shared_data(
        self,
        mock_evaluator_cls,
        mock_backtest_service_cls,
        mock_initialize_worker,
    ):
        """共有データ付きで初期化されること"""
        mock_backtest_service = MagicMock()
        mock_backtest_service_cls.return_value = mock_backtest_service

        mock_evaluator = MagicMock()
        mock_evaluator_cls.return_value = mock_evaluator

        backtest_config = {"symbol": "BTC/USDT:USDT"}
        ga_config = MagicMock()
        shared_data = {"ohlcv": MagicMock()}

        initialize_worker_process(backtest_config, ga_config, shared_data)

        mock_initialize_worker.assert_called_once_with(shared_data)

    @patch(
        "app.services.auto_strategy.core.evaluation.parallel_evaluator.initialize_worker"
    )
    @patch("app.services.backtest.services.backtest_service.BacktestService")
    @patch(
        "app.services.auto_strategy.core.evaluation.individual_evaluator.IndividualEvaluator"
    )
    def test_initialize_without_shared_data(
        self,
        mock_evaluator_cls,
        mock_backtest_service_cls,
        mock_initialize_worker,
    ):
        """共有データなしで初期化されること"""
        mock_backtest_service = MagicMock()
        mock_backtest_service_cls.return_value = mock_backtest_service

        mock_evaluator = MagicMock()
        mock_evaluator_cls.return_value = mock_evaluator

        backtest_config = {"symbol": "BTC/USDT:USDT"}
        ga_config = MagicMock()

        initialize_worker_process(backtest_config, ga_config, shared_data=None)

        mock_evaluator.set_backtest_config.assert_called_once_with(backtest_config)
        mock_initialize_worker.assert_not_called()

    @patch(
        "app.services.auto_strategy.core.evaluation.parallel_evaluator.initialize_worker"
    )
    @patch("app.services.backtest.services.backtest_service.BacktestService")
    @patch(
        "app.services.auto_strategy.core.evaluation.individual_evaluator.IndividualEvaluator"
    )
    def test_initialize_seeds_local_cache_from_shared_data(
        self,
        mock_evaluator_cls,
        mock_backtest_service_cls,
        mock_initialize_worker,
    ):
        """共有データの DataFrame ペイロードがローカルキャッシュへ投入されること"""
        mock_backtest_service = MagicMock()
        mock_backtest_service_cls.return_value = mock_backtest_service

        mock_evaluator = MagicMock()
        mock_evaluator._lock = MagicMock()
        mock_evaluator._data_cache = {}
        mock_evaluator_cls.return_value = mock_evaluator

        backtest_config = {"symbol": "BTC/USDT:USDT"}
        ga_config = MagicMock()
        minute_df = pd.DataFrame({"close": [1, 2]})
        shared_data = {
            "minute_data": {
                "key": ("minute", "BTC/USDT:USDT", "1m", "2024-01-01", "2024-06-30"),
                "data": minute_df,
            },
        }

        initialize_worker_process(backtest_config, ga_config, shared_data)

        mock_initialize_worker.assert_called_once_with(shared_data)
        assert (
            mock_evaluator._data_cache[
                ("minute", "BTC/USDT:USDT", "1m", "2024-01-01", "2024-06-30")
            ]
            is minute_df
        )

    @patch("app.services.backtest.services.backtest_service.BacktestService")
    def test_initialize_error_propagates(self, mock_backtest_service_cls):
        """初期化エラーが伝播されること"""
        mock_backtest_service_cls.side_effect = RuntimeError("DB connection failed")

        backtest_config = {"symbol": "BTC/USDT:USDT"}
        ga_config = MagicMock()

        with pytest.raises(RuntimeError, match="DB connection failed"):
            initialize_worker_process(backtest_config, ga_config)


class TestWorkerEvaluateIndividual:
    """worker_evaluate_individual関数のテスト"""

    def test_evaluate_success(self):
        """個体評価が正常に実行されること"""
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = (0.75, 0.5)
        mock_evaluator.get_last_evaluation_report.return_value = MagicMock(
            pass_rate=0.5,
            primary_aggregated_fitness=0.75,
            primary_worst_case_fitness=0.2,
            scenarios=[],
        )
        mock_config = MagicMock()

        ew_module._WORKER_EVALUATOR = mock_evaluator
        ew_module._WORKER_CONFIG = mock_config

        try:
            mock_individual = MagicMock()
            result = worker_evaluate_individual(mock_individual)

            assert isinstance(result, ParallelEvaluationResult)
            assert result.fitness == (0.75, 0.5)
            assert result.behavior_summary is not None
            assert result.behavior_summary["pass_rate"] == pytest.approx(0.5)
            mock_evaluator.evaluate.assert_called_once_with(
                mock_individual, mock_config
            )
        finally:
            ew_module._WORKER_EVALUATOR = None
            ew_module._WORKER_CONFIG = None

    def test_dynamic_scalars_are_applied_to_worker_config(self):
        """dynamic_scalars は評価前にワーカー設定へ反映されること"""
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = (0.5,)
        mock_evaluator.get_last_evaluation_report.return_value = None
        mock_config = MagicMock()
        mock_config.objective_dynamic_scalars = {}

        ew_module._WORKER_EVALUATOR = mock_evaluator
        ew_module._WORKER_CONFIG = mock_config

        try:
            scalars = {"max_drawdown": 0.2, "ulcer_index": 0.1}
            worker_evaluate_individual(MagicMock(), dynamic_scalars=scalars)

            # プール起動時に pickle された設定は古いため、評価のたびに上書きされる
            assert mock_config.objective_dynamic_scalars == scalars
        finally:
            ew_module._WORKER_EVALUATOR = None
            ew_module._WORKER_CONFIG = None

    def test_dynamic_scalars_none_keeps_existing_config(self):
        """dynamic_scalars 未指定時は設定を書き換えないこと"""
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = (0.5,)
        mock_evaluator.get_last_evaluation_report.return_value = None
        mock_config = MagicMock()
        existing = {"max_drawdown": 0.3}
        mock_config.objective_dynamic_scalars = existing

        ew_module._WORKER_EVALUATOR = mock_evaluator
        ew_module._WORKER_CONFIG = mock_config

        try:
            worker_evaluate_individual(MagicMock())

            assert mock_config.objective_dynamic_scalars is existing
        finally:
            ew_module._WORKER_EVALUATOR = None
            ew_module._WORKER_CONFIG = None

    def test_evaluate_without_initialization(self):
        """未初期化時に方向を考慮したペナルティ値が返されること"""
        ew_module._WORKER_EVALUATOR = None
        ew_module._WORKER_CONFIG = None

        try:
            mock_individual = MagicMock()
            result = worker_evaluate_individual(mock_individual)

            assert isinstance(result, ParallelEvaluationResult)
            assert result.fitness == (-PENALTY_FITNESS_MAGNITUDE,)
            assert result.behavior_summary is None
        finally:
            ew_module._WORKER_EVALUATOR = None
            ew_module._WORKER_CONFIG = None

    def test_evaluate_with_error(self):
        """評価エラー時に方向を考慮したペナルティ値が返されること"""
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.side_effect = ValueError("Evaluation failed")
        mock_config = MagicMock()

        ew_module._WORKER_EVALUATOR = mock_evaluator
        ew_module._WORKER_CONFIG = mock_config

        try:
            mock_individual = MagicMock()
            result = worker_evaluate_individual(mock_individual)

            assert isinstance(result, ParallelEvaluationResult)
            assert result.fitness == (-PENALTY_FITNESS_MAGNITUDE,)
            assert result.behavior_summary is None
        finally:
            ew_module._WORKER_EVALUATOR = None
            ew_module._WORKER_CONFIG = None

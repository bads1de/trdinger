"""
EvaluationWorker のユニットテスト

並列評価ワーカーモジュールをテストします。
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.auto_strategy.core.evaluation.evaluation_worker import (
    initialize_worker_process,
    worker_evaluate_individual,
    _WORKER_EVALUATOR,
    _WORKER_CONFIG,
)
import app.services.auto_strategy.core.evaluation.evaluation_worker as ew_module


class TestInitializeWorkerProcess:
    """initialize_worker_process関数のテスト"""

    @patch("app.services.auto_strategy.core.evaluation.evaluation_worker.BacktestService")
    @patch("app.services.auto_strategy.core.evaluation.evaluation_worker.IndividualEvaluator")
    def test_initialize_success(self, mock_evaluator_cls, mock_backtest_service_cls):
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

    @patch("app.services.auto_strategy.core.evaluation.evaluation_worker.BacktestService")
    @patch("app.services.auto_strategy.core.evaluation.evaluation_worker.IndividualEvaluator")
    def test_initialize_with_shared_data(self, mock_evaluator_cls, mock_backtest_service_cls):
        """共有データ付きで初期化されること"""
        mock_backtest_service = MagicMock()
        mock_backtest_service_cls.return_value = mock_backtest_service
        
        mock_evaluator = MagicMock()
        mock_evaluator_cls.return_value = mock_evaluator
        
        backtest_config = {"symbol": "BTC/USDT:USDT"}
        ga_config = MagicMock()
        shared_data = {"ohlcv": MagicMock()}
        
        with patch("app.services.auto_strategy.core.evaluation.parallel_evaluator.initialize_worker") as mock_init_worker:
            initialize_worker_process(backtest_config, ga_config, shared_data)
            mock_init_worker.assert_called_once_with(shared_data)

    @patch("app.services.auto_strategy.core.evaluation.evaluation_worker.BacktestService")
    @patch("app.services.auto_strategy.core.evaluation.evaluation_worker.IndividualEvaluator")
    def test_initialize_without_shared_data(self, mock_evaluator_cls, mock_backtest_service_cls):
        """共有データなしで初期化されること"""
        mock_backtest_service = MagicMock()
        mock_backtest_service_cls.return_value = mock_backtest_service
        
        mock_evaluator = MagicMock()
        mock_evaluator_cls.return_value = mock_evaluator
        
        backtest_config = {"symbol": "BTC/USDT:USDT"}
        ga_config = MagicMock()
        
        initialize_worker_process(backtest_config, ga_config, shared_data=None)
        
        mock_evaluator.set_backtest_config.assert_called_once_with(backtest_config)

    @patch("app.services.auto_strategy.core.evaluation.evaluation_worker.BacktestService")
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
        mock_config = MagicMock()
        
        ew_module._WORKER_EVALUATOR = mock_evaluator
        ew_module._WORKER_CONFIG = mock_config
        
        try:
            mock_individual = MagicMock()
            result = worker_evaluate_individual(mock_individual)
            
            assert result == (0.75, 0.5)
            mock_evaluator.evaluate.assert_called_once_with(mock_individual, mock_config)
        finally:
            ew_module._WORKER_EVALUATOR = None
            ew_module._WORKER_CONFIG = None

    def test_evaluate_without_initialization(self):
        """未初期化時にデフォルト値が返されること"""
        ew_module._WORKER_EVALUATOR = None
        ew_module._WORKER_CONFIG = None
        
        try:
            mock_individual = MagicMock()
            result = worker_evaluate_individual(mock_individual)
            
            assert result == (0.0,)
        finally:
            ew_module._WORKER_EVALUATOR = None
            ew_module._WORKER_CONFIG = None

    def test_evaluate_with_error(self):
        """評価エラー時にデフォルト値が返されること"""
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.side_effect = ValueError("Evaluation failed")
        mock_config = MagicMock()
        
        ew_module._WORKER_EVALUATOR = mock_evaluator
        ew_module._WORKER_CONFIG = mock_config
        
        try:
            mock_individual = MagicMock()
            result = worker_evaluate_individual(mock_individual)
            
            assert result == (0.0,)
        finally:
            ew_module._WORKER_EVALUATOR = None
            ew_module._WORKER_CONFIG = None

import threading
from unittest.mock import Mock, patch

import pandas as pd

from app.services.auto_strategy.core.evaluation.backtest_data_provider import (
    BacktestDataProvider,
)


class TestBacktestDataProviderCache:
    def test_get_cached_backtest_data_ignores_worker_data_for_mismatched_key(self):
        mock_service = Mock()
        mock_service.ensure_data_service_initialized = Mock()
        mock_service.data_service = Mock()
        mock_df = pd.DataFrame({"close": [1, 2, 3]})
        mock_service.data_service.get_data_for_backtest.return_value = mock_df

        provider = BacktestDataProvider(
            backtest_service=mock_service,
            data_cache={},
            lock=threading.Lock(),
            prefetch_enabled=False,
        )

        config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
        }

        with patch(
            "app.services.auto_strategy.core.evaluation.parallel_evaluator.get_worker_data",
            return_value={
                "key": ("BTC/USDT:USDT", "1h", "2023-12-01", "2023-12-02"),
                "data": "stale-worker-data",
            },
        ):
            result = provider.get_cached_backtest_data(config)

        assert result is mock_df
        assert mock_service.data_service.get_data_for_backtest.call_count == 1

    def test_get_cached_backtest_data_uses_worker_data_for_matching_key(self):
        mock_service = Mock()
        mock_service.ensure_data_service_initialized = Mock()
        mock_service.data_service = Mock()
        mock_df = pd.DataFrame({"close": [1, 2, 3]})

        provider = BacktestDataProvider(
            backtest_service=mock_service,
            data_cache={},
            lock=threading.Lock(),
            prefetch_enabled=False,
        )

        config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
        }

        with patch(
            "app.services.auto_strategy.core.evaluation.parallel_evaluator.get_worker_data",
            return_value={"key": ("BTC/USDT:USDT", "1h", "2024-01-01", "2024-01-02"), "data": mock_df},
        ):
            result = provider.get_cached_backtest_data(config)

        assert result is mock_df
        mock_service.data_service.get_data_for_backtest.assert_not_called()

from unittest.mock import Mock

import pandas as pd

from app.services.auto_strategy.core.evaluation.backtest_data_provider import (
    BacktestDataProvider,
)


class TestBacktestDataProvider:
    def test_get_cached_backtest_data_uses_cache_after_first_load(self):
        mock_service = Mock()
        mock_service.ensure_data_service_initialized = Mock()
        mock_service.data_service = Mock()
        mock_df = pd.DataFrame({"close": [1, 2, 3]})
        mock_service.data_service.get_data_for_backtest.return_value = mock_df

        cache = {}
        provider = BacktestDataProvider(mock_service, cache, Mock())
        provider._lock.__enter__ = Mock(return_value=None)
        provider._lock.__exit__ = Mock(return_value=None)

        config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
        }

        first = provider.get_cached_backtest_data(config)
        second = provider.get_cached_backtest_data(config)

        assert first is mock_df
        assert second is mock_df
        assert mock_service.data_service.get_data_for_backtest.call_count == 1

    def test_get_cached_ohlcv_data_returns_none_when_params_missing(self):
        mock_service = Mock()
        cache = {}
        provider = BacktestDataProvider(mock_service, cache, Mock())
        provider._lock.__enter__ = Mock(return_value=None)
        provider._lock.__exit__ = Mock(return_value=None)

        result = provider.get_cached_ohlcv_data(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=None,
            end_date="2024-01-01",
        )

        assert result is None

"""
backtest/conversion の data_transformers と statistics_calculator のユニットテスト
"""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.conversion.data_transformers import (
    EquityCurveTransformer,
    TradeHistoryTransformer,
)
from app.services.backtest.conversion.statistics_calculator import (
    BacktestStatisticsCalculator,
)


# ---------------------------------------------------------------------------
# TradeHistoryTransformer
# ---------------------------------------------------------------------------

class TestTradeHistoryTransformer:
    @pytest.fixture
    def transformer(self):
        return TradeHistoryTransformer()

    def test_empty_trades(self, transformer):
        stats = MagicMock()
        stats._trades = None
        assert transformer.transform(stats) == []

    def test_empty_dataframe(self, transformer):
        stats = MagicMock()
        stats._trades = pd.DataFrame()
        assert transformer.transform(stats) == []

    def test_normal_trades(self, transformer):
        trades = pd.DataFrame({
            "EntryTime": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "ExitTime": [datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "EntryPrice": [100.0, 101.0],
            "ExitPrice": [102.0, 100.0],
            "Size": [1.0, 1.0],
            "PnL": [2.0, -1.0],
            "ReturnPct": [0.02, -0.01],
            "Duration": [24, 24],
        })
        stats = MagicMock()
        stats._trades = trades

        result = transformer.transform(stats)
        assert len(result) == 2
        assert result[0]["entry_price"] == 100.0
        assert result[0]["pnl"] == 2.0

    def test_pnl_alias_columns_are_supported(self, transformer):
        trades = pd.DataFrame({
            "EntryTime": [datetime(2024, 1, 1)],
            "ExitTime": [datetime(2024, 1, 1, 1)],
            "EntryPrice": [100.0],
            "ExitPrice": [102.0],
            "Size": [1.0],
            "Pnl": [2.5],
            "ReturnPct": [0.025],
            "Duration": [1],
        })
        stats = MagicMock()
        stats._trades = trades

        result = transformer.transform(stats)

        assert len(result) == 1
        assert result[0]["pnl"] == 2.5

    def test_missing_columns_fill_defaults(self, transformer):
        trades = pd.DataFrame({"EntryPrice": [100.0]})
        stats = MagicMock()
        stats._trades = trades

        result = transformer.transform(stats)
        assert len(result) == 1
        # 欠損カラムはデフォルト値で埋められる
        assert result[0]["size"] == 0.0

    def test_no_trades_attr(self, transformer):
        stats = MagicMock(spec=[])  # _trades 属性なし
        result = transformer.transform(stats)
        assert result == []


# ---------------------------------------------------------------------------
# EquityCurveTransformer
# ---------------------------------------------------------------------------

class TestEquityCurveTransformer:
    @pytest.fixture
    def transformer(self):
        return EquityCurveTransformer()

    def test_empty_curve(self, transformer):
        stats = MagicMock()
        stats._equity_curve = pd.DataFrame()
        assert transformer.transform(stats) == []

    def test_none_curve(self, transformer):
        stats = MagicMock()
        stats._equity_curve = None
        assert transformer.transform(stats) == []

    def test_normal_curve(self, transformer):
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        curve = pd.DataFrame(
            {"Equity": np.linspace(10000, 11000, 10), "DrawdownPct": np.zeros(10)},
            index=dates,
        )
        stats = MagicMock()
        stats._equity_curve = curve

        result = transformer.transform(stats)
        assert len(result) == 10
        assert result[0]["equity"] == 10000.0
        assert "timestamp" in result[0]

    def test_downsampling(self, transformer):
        dates = pd.date_range("2024-01-01", periods=5000, freq="h")
        curve = pd.DataFrame(
            {"Equity": np.random.randn(5000).cumsum() + 10000, "DrawdownPct": np.zeros(5000)},
            index=dates,
        )
        stats = MagicMock()
        stats._equity_curve = curve

        result = transformer.transform(stats, max_points=100)
        assert len(result) <= 100

    def test_safe_timestamp_conversion_pd_timestamp(self, transformer):
        ts = pd.Timestamp("2024-06-15 12:00:00")
        result = EquityCurveTransformer._safe_timestamp_conversion(ts)
        assert isinstance(result, datetime)

    def test_safe_timestamp_conversion_none(self, transformer):
        assert EquityCurveTransformer._safe_timestamp_conversion(None) is None

    def test_safe_timestamp_conversion_datetime(self, transformer):
        dt = datetime(2024, 6, 15, 12, 0)
        result = EquityCurveTransformer._safe_timestamp_conversion(dt)
        assert result == dt


# ---------------------------------------------------------------------------
# BacktestStatisticsCalculator
# ---------------------------------------------------------------------------

class TestBacktestStatisticsCalculator:
    @pytest.fixture
    def calculator(self):
        return BacktestStatisticsCalculator()

    def test_calculate_from_series(self, calculator):
        stats = pd.Series({
            "Return [%]": 15.0,
            "Win Rate [%]": 60.0,
            "Profit Factor": 1.5,
            "# Trades": 100,
            "Best Trade [%]": 5.0,
            "Worst Trade [%]": -3.0,
            "Sharpe Ratio": 1.2,
            "Equity Final [$]": 11500.0,
        })
        result = calculator.calculate_statistics(stats)
        assert result["total_return"] == 15.0
        assert result["win_rate"] == 60.0
        assert result["total_trades"] == 100
        assert result["sharpe_ratio"] == 1.2

    def test_calculate_from_dict(self, calculator):
        stats = {
            "Return [%]": 10.0,
            "Win Rate [%]": 55.0,
            "# Trades": 50,
        }
        result = calculator.calculate_statistics(stats)
        assert result["total_return"] == 10.0
        assert result["total_trades"] == 50

    def test_callable_stats(self, calculator):
        stats = lambda: pd.Series({"Return [%]": 20.0, "# Trades": 30})
        result = calculator.calculate_statistics(stats)
        assert result["total_return"] == 20.0

    def test_empty_stats(self, calculator):
        result = calculator.calculate_statistics({})
        assert isinstance(result, dict)

    def test_with_trades_df(self, calculator):
        stats = pd.Series({"Return [%]": 5.0, "# Trades": 0})
        trades = pd.DataFrame({
            "PnL": [100, -50, 200, -30],
            "ReturnPct": [0.01, -0.005, 0.02, -0.003],
        })
        stats._trades = trades

        result = calculator.calculate_statistics(stats)
        assert result["total_trades"] == 4
        assert result["win_rate"] == 50.0

    def test_validate_and_fill_defaults(self, calculator):
        stats = pd.Series({"Return [%]": 10.0})
        result = calculator.calculate_statistics(stats)
        # デフォルト値で埋められる
        assert "profit_factor" in result
        assert "sharpe_ratio" in result
        assert result["total_trades"] >= 0

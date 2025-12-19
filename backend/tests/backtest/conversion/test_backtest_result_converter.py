import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock
from app.services.backtest.conversion.backtest_result_converter import BacktestResultConverter, BacktestResultConversionError

class TestBacktestResultConverter:
    @pytest.fixture
    def converter(self):
        return BacktestResultConverter()

    @pytest.fixture
    def mock_stats(self):
        # backtesting.pyのstatsを模倣したSeries
        stats = pd.Series({
            "Return [%]": 10.5,
            "# Trades": 5,
            "Win Rate [%]": 60.0,
            "Profit Factor": 1.5,
            "Max. Drawdown [%]": 5.0,
            "Sharpe Ratio": 2.0,
            "Equity Final [$]": 11050.0
        })
        
        # 内部属性としての取引データ
        trades_df = pd.DataFrame({
            "EntryTime": pd.to_datetime(["2023-01-01 10:00", "2023-01-01 11:00"]),
            "ExitTime": pd.to_datetime(["2023-01-01 10:30", "2023-01-01 11:30"]),
            "EntryPrice": [100, 105],
            "ExitPrice": [102, 104],
            "PnL": [2.0, -1.0],
            "Size": [1, 1],
            "ReturnPct": [0.02, -0.01],
            "Duration": [30, 30]
        })
        stats._trades = trades_df
        
        # 内部属性としてのエクイティカーブ
        equity_df = pd.DataFrame({
            "Equity": [10000, 10020, 10010, 11050],
            "DrawdownPct": [0, 0, 0.001, 0]
        }, index=pd.to_datetime(["2023-01-01 09:00", "2023-01-01 10:00", "2023-01-01 11:00", "2023-01-01 12:00"]))
        stats._equity_curve = equity_df
        
        return stats

    def test_normalize_date(self, converter):
        dt = datetime(2023, 1, 1)
        assert converter._normalize_date(dt) == dt
        assert converter._normalize_date("2023-01-01T00:00:00") == dt
        
        with pytest.raises(ValueError):
            converter._normalize_date(12345)

    def test_safe_conversions(self, converter):
        assert converter._safe_float_conversion("10.5") == 10.5
        assert converter._safe_float_conversion(None) == 0.0
        assert converter._safe_float_conversion("abc") == 0.0
        
        assert converter._safe_int_conversion("5") == 5
        assert converter._safe_int_conversion(None) == 0
        
        ts = pd.Timestamp("2023-01-01")
        assert converter._safe_timestamp_conversion(ts) == ts.to_pydatetime()
        assert converter._safe_timestamp_conversion(None) is None

    def test_extract_statistics_from_series(self, converter, mock_stats):
        metrics = converter._extract_statistics(mock_stats)
        
        assert metrics["total_return"] == 10.5
        assert metrics["total_trades"] == 2 # trades_dfの長さで上書きされる
        assert metrics["win_rate"] == 50.0 # 2トレード中1勝
        assert metrics["profit_factor"] == 2.0 # 2 / 1
        assert metrics["final_equity"] == 11050.0

    def test_extract_statistics_from_dict(self, converter):
        stats_dict = {
            "Return [%]": 5.0,
            "# Trades": 1,
            "Win Rate [%]": 100.0
        }
        # 取引データなし
        metrics = converter._extract_statistics(stats_dict)
        assert metrics["total_return"] == 5.0
        assert metrics["total_trades"] == 1

    def test_convert_trade_history(self, converter, mock_stats):
        history = converter._convert_trade_history(mock_stats)
        
        assert len(history) == 2
        assert history[0]["entry_price"] == 100.0
        assert history[0]["pnl"] == 2.0
        assert isinstance(history[0]["entry_time"], datetime)

    def test_convert_equity_curve(self, converter, mock_stats):
        curve = converter._convert_equity_curve(mock_stats)
        
        assert len(curve) == 4
        assert curve[0]["equity"] == 10000.0
        assert curve[3]["equity"] == 11050.0
        assert isinstance(curve[0]["timestamp"], datetime)

    def test_convert_backtest_results_full(self, converter, mock_stats):
        result = converter.convert_backtest_results(
            stats=mock_stats,
            strategy_name="SMA",
            symbol="BTC/USDT",
            timeframe="1h",
            initial_capital=10000.0,
            start_date="2023-01-01T00:00:00",
            end_date="2023-01-02T00:00:00",
            config_json={"p": 1}
        )
        
        assert result["strategy_name"] == "SMA"
        assert result["performance_metrics"]["total_return"] == 10.5
        assert len(result["trade_history"]) == 2
        assert len(result["equity_curve"]) == 4
        assert result["status"] == "completed"

    def test_convert_backtest_results_empty_stats(self, converter):
        # statsがNoneの場合でも、例外を投げずにデフォルトの結果を構築することを期待
        result = converter.convert_backtest_results(
            stats=None, 
            strategy_name="S", symbol="S", timeframe="1h", 
            initial_capital=100, 
            start_date="2023-01-01T00:00:00", end_date="2023-01-02T00:00:00", 
            config_json={}
        )
        assert result["status"] == "completed"
        # デフォルト値が入っているはず
        assert result["performance_metrics"]["total_trades"] == 0
        assert result["trade_history"] == []

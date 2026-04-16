"""
EvaluationWindowService のユニットテスト
"""

from unittest.mock import Mock, patch

import pandas as pd

from app.services.auto_strategy.core.evaluation.evaluation_window_service import (
    EvaluationWindowService,
)
from app.services.auto_strategy.genes import Condition, IndicatorGene, StrategyGene


class TestEvaluationWindowService:
    """EvaluationWindowService のテストクラス"""

    def setup_method(self):
        self.service = EvaluationWindowService()

    def test_prepare_backtest_config_for_evaluation_adds_warmup_window(self):
        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="EMA", parameters={"length": 20}, enabled=True)
            ],
            long_entry_conditions=[],
            short_entry_conditions=[],
        )
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-10 00:00:00",
            "end_date": "2024-01-12 00:00:00",
        }

        prepared = self.service.prepare_backtest_config_for_evaluation(
            gene, backtest_config
        )

        assert prepared["_evaluation_start"] == "2024-01-10 00:00:00"
        assert prepared["end_date"] == "2024-01-12 00:00:00"
        assert prepared["start_date"] == "2024-01-09 03:00:00"

    def test_public_and_private_helpers_share_same_implementation(self):
        assert (
            self.service.prepare_backtest_config_for_evaluation.__func__
            is self.service._prepare_backtest_config_for_evaluation.__func__
        )
        assert (
            self.service.estimate_required_warmup_bars.__func__
            is self.service._estimate_required_warmup_bars.__func__
        )
        assert (
            self.service.apply_evaluation_window_to_result.__func__
            is self.service._apply_evaluation_window_to_result.__func__
        )
        assert (
            EvaluationWindowService.extract_lookback_from_parameters
            is EvaluationWindowService._extract_lookback_from_parameters
        )
        assert (
            EvaluationWindowService.timeframe_to_minutes
            is EvaluationWindowService._timeframe_to_minutes
        )
        assert (
            EvaluationWindowService.format_datetime_like
            is EvaluationWindowService._format_datetime_like
        )
        assert (
            EvaluationWindowService.normalize_timestamp_to_index
            is EvaluationWindowService._normalize_timestamp_to_index
        )
        assert (
            EvaluationWindowService.normalize_ohlc_data_for_stats
            is EvaluationWindowService._normalize_ohlc_data_for_stats
        )
        assert (
            EvaluationWindowService.slice_equity_curve_for_window
            is EvaluationWindowService._slice_equity_curve_for_window
        )
        assert (
            EvaluationWindowService.slice_trades_for_window
            is EvaluationWindowService._slice_trades_for_window
        )

    def test_apply_evaluation_window_to_result_recomputes_trimmed_window(self):
        market_data = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 110.0],
                "high": [101.0, 101.0, 111.0, 111.0],
                "low": [99.0, 99.0, 109.0, 109.0],
                "close": [100.0, 100.0, 110.0, 110.0],
                "volume": [10.0, 10.0, 10.0, 10.0],
            },
            index=pd.date_range("2024-01-01 00:00:00", periods=4, freq="D"),
        )
        raw_stats = Mock()
        raw_stats._equity_curve = pd.DataFrame(
            {
                "Equity": [10000.0, 10000.0, 11000.0, 11000.0],
                "DrawdownPct": [0.0, 0.0, 0.0, 0.0],
            },
            index=market_data.index,
        )
        raw_stats._trades = pd.DataFrame(
            {
                "Size": [1],
                "EntryBar": [2],
                "ExitBar": [3],
                "EntryPrice": [100.0],
                "ExitPrice": [110.0],
                "SL": [None],
                "TP": [None],
                "PnL": [10.0],
                "Commission": [0.0],
                "ReturnPct": [0.10],
                "EntryTime": [market_data.index[2]],
                "ExitTime": [market_data.index[3]],
                "Duration": [market_data.index[3] - market_data.index[2]],
                "Tag": [None],
            }
        )

        converted_result = {
            "strategy_name": "WarmupTest",
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "initial_capital": 10000.0,
            "config_json": {},
            "performance_metrics": {},
            "trade_history": [],
            "equity_curve": [],
        }

        with (
            patch.object(
                self.service,
                "_compute_window_stats",
                return_value="window_stats",
            ) as mock_compute_window_stats,
            patch(
                "app.services.backtest.conversion.backtest_result_converter.BacktestResultConverter.convert_backtest_results",
                return_value={
                    "strategy_name": "WarmupTest",
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "1h",
                    "initial_capital": 10000.0,
                    "performance_metrics": {"total_return": 10.0},
                    "trade_history": [{"pnl": 10.0}],
                    "equity_curve": [{"equity": 10000.0}, {"equity": 11000.0}],
                    "start_date": pd.Timestamp("2024-01-03 00:00:00").to_pydatetime(),
                    "end_date": pd.Timestamp("2024-01-04 00:00:00").to_pydatetime(),
                    "config_json": {},
                },
            ),
        ):
            adjusted = self.service.apply_evaluation_window_to_result(
                converted_result,
                raw_stats,
                market_data,
                pd.Timestamp("2024-01-03 00:00:00"),
                pd.Timestamp("2024-01-04 00:00:00"),
            )

        trades_df, equity_values, ohlc_data = mock_compute_window_stats.call_args.args
        assert list(ohlc_data.index) == list(market_data.index[2:])
        assert list(equity_values) == [11000.0, 11000.0]
        assert trades_df["EntryBar"].tolist() == [0]
        assert adjusted["performance_metrics"]["total_return"] == 10.0

    def test_normalize_ohlc_data_for_stats_preserves_non_ohlcv_columns(self):
        market_data = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "open_interest": [500.0, 510.0],
                "MarketRegime": [1, 2],
            },
            index=pd.date_range("2024-01-01 00:00:00", periods=2, freq="D"),
        )

        normalized = self.service.normalize_ohlc_data_for_stats(market_data)

        assert "Open" in normalized.columns
        assert "High" in normalized.columns
        assert "Low" in normalized.columns
        assert "Close" in normalized.columns
        assert "Volume" in normalized.columns
        assert "open_interest" in normalized.columns
        assert "MarketRegime" in normalized.columns
        assert "Open_interest" not in normalized.columns
        assert "Marketregime" not in normalized.columns
        assert normalized["Volume"].tolist() == [0.0, 0.0]

    def test_slice_equity_curve_for_window_uses_initial_capital_for_leading_gaps(self):
        target_index = pd.date_range("2024-01-01 00:00:00", periods=4, freq="D")
        raw_equity_curve = pd.DataFrame(
            {
                "Equity": [10100.0, 10200.0, 10300.0],
                "DrawdownPct": [0.1, 0.2, 0.3],
            },
            index=target_index[1:],
        )

        trimmed = self.service.slice_equity_curve_for_window(
            raw_equity_curve,
            target_index,
            0,
            len(target_index),
            10000.0,
        )

        assert trimmed.loc[target_index[0], "Equity"] == 10000.0
        assert trimmed["Equity"].isna().sum() == 0
        assert trimmed.index.equals(target_index)

    def test_estimate_required_warmup_bars_scales_multi_timeframe_indicator(self):
        gene = StrategyGene(
            id="mtf-gene",
            indicators=[
                IndicatorGene(
                    type="EMA",
                    parameters={"length": 20},
                    enabled=True,
                    timeframe="4h",
                )
            ],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="EMA_20")
            ],
            short_entry_conditions=[],
        )

        warmup_bars = self.service.estimate_required_warmup_bars(gene, "1h")

        assert warmup_bars == 84

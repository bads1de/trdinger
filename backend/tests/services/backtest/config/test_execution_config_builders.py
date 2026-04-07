from unittest.mock import Mock

from app.services.backtest.config.builders import (
    build_execution_config,
    ensure_backtest_defaults,
)


class TestExecutionConfigBuilders:
    def test_build_execution_config_from_dict_preserves_extra_keys(self):
        strategy_config = Mock()
        strategy_config.model_dump.return_value = {
            "strategy_type": "GENERATED_GA",
            "parameters": {"strategy_gene": {"id": "gene-1"}},
        }
        source = {
            "strategy_name": "TestStrategy",
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "initial_capital": 10000,
            "commission_rate": 0.001,
            "_skip_validation": True,
            "strategy_config": strategy_config,
        }

        result = build_execution_config(source)

        assert result["strategy_name"] == "TestStrategy"
        assert result["strategy_config"] == {
            "strategy_type": "GENERATED_GA",
            "parameters": {"strategy_gene": {"id": "gene-1"}},
        }
        assert result["_skip_validation"] is True
        assert result["slippage"] == 0.0
        assert result["leverage"] == 1.0

    def test_build_execution_config_supports_explicit_overrides(self):
        source = Mock()
        source.strategy_name = "Original"
        source.symbol = "BTC/USDT:USDT"
        source.timeframe = "1h"
        source.start_date = "2024-01-01"
        source.end_date = "2024-01-02"
        source.initial_capital = 10000
        source.commission_rate = 0.001
        source.slippage = 0.0002
        source.leverage = 2.0
        source.strategy_config = {"strategy_type": "MANUAL", "parameters": {}}

        result = build_execution_config(
            source,
            strategy_name="Override",
            strategy_config={"strategy_type": "MANUAL", "parameters": {"x": 1}},
        )

        assert result["strategy_name"] == "Override"
        assert result["strategy_config"] == {
            "strategy_type": "MANUAL",
            "parameters": {"x": 1},
        }
        assert result["slippage"] == 0.0002
        assert result["leverage"] == 2.0

    def test_ensure_backtest_defaults_fills_missing_values_without_mutating_input(self):
        source = {"symbol": "BTC/USDT:USDT", "start_date": None}
        defaults = {
            "symbol": "ETH/USDT:USDT",
            "timeframe": "4h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }

        result = ensure_backtest_defaults(source, defaults)

        assert result["symbol"] == "BTC/USDT:USDT"
        assert result["timeframe"] == "4h"
        assert result["start_date"] == "2024-01-01"
        assert result["end_date"] == "2024-01-31"
        assert source["start_date"] is None

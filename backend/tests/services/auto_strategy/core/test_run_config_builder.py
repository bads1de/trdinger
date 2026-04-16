from types import SimpleNamespace
from unittest.mock import Mock

from app.services.auto_strategy.config.ga.nested_configs import EarlyTerminationSettings
from app.services.auto_strategy.core.evaluation.run_config_builder import (
    RunConfigBuilder,
)


class TestRunConfigBuilder:
    def test_build_run_config_sets_strategy_payload_and_skip_validation(self):
        builder = RunConfigBuilder()
        gene = Mock()
        gene.id = "gene-123456789"

        ga_config = SimpleNamespace(
            volatility_gate_enabled=True,
            volatility_model_path="/tmp/model.pkl",
            evaluation_config=SimpleNamespace(
                early_termination_settings=EarlyTerminationSettings()
            ),
        )

        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
        }

        result = builder.build_run_config(gene, backtest_config, ga_config)

        assert result is not None
        assert result["_skip_validation"] is True
        assert result["strategy_name"] == "GA_Individual_gene-123"
        assert result["strategy_config"]["parameters"]["strategy_gene"] is gene
        assert (
            result["strategy_config"]["parameters"]["volatility_gate_enabled"] is True
        )
        assert (
            result["strategy_config"]["parameters"]["volatility_model_path"]
            == "/tmp/model.pkl"
        )

    def test_build_run_config_applies_volatility_gate(self):
        builder = RunConfigBuilder()
        gene = Mock()
        gene.id = "gene-123456789"

        ga_config = SimpleNamespace(
            volatility_gate_enabled=True,
            volatility_model_path="/tmp/vol-model.pkl",
            evaluation_config=SimpleNamespace(
                early_termination_settings=EarlyTerminationSettings()
            ),
        )

        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
        }

        result = builder.build_run_config(gene, backtest_config, ga_config)

        assert result is not None
        params = result["strategy_config"]["parameters"]
        assert params["volatility_gate_enabled"] is True
        assert params["volatility_model_path"] == "/tmp/vol-model.pkl"

    def test_build_run_config_applies_defaults(self):
        builder = RunConfigBuilder()
        gene = Mock()
        gene.id = "gene-123456789"

        ga_config = SimpleNamespace(
            volatility_gate_enabled=False,
            volatility_model_path=None,
            evaluation_config=SimpleNamespace(
                early_termination_settings=EarlyTerminationSettings()
            ),
        )

        defaults = {
            "symbol": "ETH/USDT:USDT",
            "timeframe": "4h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }

        result = builder.build_run_config(
            gene,
            {},
            ga_config,
            defaults=defaults,
        )

        assert result is not None
        assert result["symbol"] == "ETH/USDT:USDT"
        assert result["timeframe"] == "4h"
        assert result["start_date"] == "2024-01-01"
        assert result["end_date"] == "2024-01-31"

    def test_inject_external_objects_adds_minute_data_only_when_present(self):
        builder = RunConfigBuilder()
        run_config = {
            "strategy_config": {
                "parameters": {},
            }
        }

        builder.inject_external_objects(run_config, minute_data=None)
        assert "minute_data" not in run_config["strategy_config"]["parameters"]

        builder.inject_external_objects(run_config, minute_data={"rows": 10})
        assert run_config["strategy_config"]["parameters"]["minute_data"] == {
            "rows": 10
        }

    def test_build_run_config_forwards_evaluation_start_when_present(self):
        builder = RunConfigBuilder()
        gene = Mock()
        gene.id = "gene-123456789"

        ga_config = SimpleNamespace(
            volatility_gate_enabled=False,
            volatility_model_path=None,
            evaluation_config=SimpleNamespace(
                early_termination_settings=EarlyTerminationSettings()
            ),
        )

        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "_evaluation_start": "2024-01-02 00:00:00",
        }

        result = builder.build_run_config(gene, backtest_config, ga_config)

        assert result is not None
        assert (
            result["strategy_config"]["parameters"]["evaluation_start"]
            == "2024-01-02 00:00:00"
        )

    def test_build_run_config_injects_early_termination_settings(self):
        builder = RunConfigBuilder()
        gene = Mock()
        gene.id = "gene-early-stop"

        ga_config = SimpleNamespace(
            volatility_gate_enabled=False,
            volatility_model_path=None,
            ml_filter_enabled=False,
            ml_model_path=None,
            evaluation_config=SimpleNamespace(
                early_termination_settings=EarlyTerminationSettings(
                    enabled=True,
                    max_drawdown=0.15,
                    min_trades=20,
                    min_trade_check_progress=0.4,
                    trade_pace_tolerance=0.5,
                    min_expectancy=-0.01,
                    expectancy_min_trades=5,
                    expectancy_progress=0.6,
                )
            ),
        )

        result = builder.build_run_config(
            gene,
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
            },
            ga_config,
        )

        assert result is not None
        params = result["strategy_config"]["parameters"]
        assert params["early_termination_settings"] == {
            "enabled": True,
            "max_drawdown": 0.15,
            "min_trades": 20,
            "min_trade_check_progress": 0.4,
            "trade_pace_tolerance": 0.5,
            "min_expectancy": -0.01,
            "expectancy_min_trades": 5,
            "expectancy_progress": 0.6,
        }

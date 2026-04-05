from unittest.mock import Mock

from app.services.auto_strategy.core.evaluation.run_config_builder import (
    RunConfigBuilder,
)


class TestRunConfigBuilder:
    def test_build_run_config_sets_strategy_payload_and_skip_validation(self):
        builder = RunConfigBuilder()
        gene = Mock()
        gene.id = "gene-123456789"

        ga_config = Mock()
        ga_config.volatility_gate_enabled = True
        ga_config.volatility_model_path = "/tmp/model.pkl"
        ga_config.ml_filter_enabled = True
        ga_config.ml_model_path = "/tmp/model.pkl"

        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
        }

        result = builder.build_run_config(gene, backtest_config, ga_config)

        assert result is not None
        assert result["_skip_validation"] is True
        assert result["strategy_name"] == "GA_Individual_gene-123"
        assert result["strategy_config"]["parameters"]["strategy_gene"] is gene
        assert result["strategy_config"]["parameters"]["volatility_gate_enabled"] is True
        assert (
            result["strategy_config"]["parameters"]["volatility_model_path"]
            == "/tmp/model.pkl"
        )
        assert result["strategy_config"]["parameters"]["ml_filter_enabled"] is True
        assert (
            result["strategy_config"]["parameters"]["ml_model_path"]
            == "/tmp/model.pkl"
        )

    def test_build_run_config_normalizes_legacy_ml_filter_aliases(self):
        builder = RunConfigBuilder()
        gene = Mock()
        gene.id = "gene-123456789"

        ga_config = Mock()
        ga_config.volatility_gate_enabled = False
        ga_config.volatility_model_path = None
        ga_config.ml_filter_enabled = True
        ga_config.ml_model_path = "/tmp/legacy-model.pkl"

        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
        }

        result = builder.build_run_config(gene, backtest_config, ga_config)

        assert result is not None
        params = result["strategy_config"]["parameters"]
        assert params["volatility_gate_enabled"] is True
        assert params["ml_filter_enabled"] is True
        assert params["volatility_model_path"] == "/tmp/legacy-model.pkl"
        assert params["ml_model_path"] == "/tmp/legacy-model.pkl"

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

        ga_config = Mock()
        ga_config.volatility_gate_enabled = False
        ga_config.volatility_model_path = None
        ga_config.ml_filter_enabled = False
        ga_config.ml_model_path = None

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

        ga_config = Mock()
        ga_config.volatility_gate_enabled = False
        ga_config.volatility_model_path = None
        ga_config.ml_filter_enabled = False
        ga_config.ml_model_path = None
        ga_config.enable_early_termination = True
        ga_config.early_termination_max_drawdown = 0.15
        ga_config.early_termination_min_trades = 20
        ga_config.early_termination_min_trade_check_progress = 0.4
        ga_config.early_termination_trade_pace_tolerance = 0.5
        ga_config.early_termination_min_expectancy = -0.01
        ga_config.early_termination_expectancy_min_trades = 5
        ga_config.early_termination_expectancy_progress = 0.6

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
        assert params["enable_early_termination"] is True
        assert params["early_termination_max_drawdown"] == 0.15
        assert params["early_termination_min_trades"] == 20
        assert params["early_termination_min_trade_check_progress"] == 0.4
        assert params["early_termination_trade_pace_tolerance"] == 0.5
        assert params["early_termination_min_expectancy"] == -0.01
        assert params["early_termination_expectancy_min_trades"] == 5
        assert params["early_termination_expectancy_progress"] == 0.6

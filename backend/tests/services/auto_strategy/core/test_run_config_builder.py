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
        assert result["strategy_config"]["parameters"]["ml_filter_enabled"] is True
        assert (
            result["strategy_config"]["parameters"]["ml_model_path"]
            == "/tmp/model.pkl"
        )

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

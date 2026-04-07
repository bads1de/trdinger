"""HybridIndividualEvaluatorのテスト。"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest


class TestHybridIndividualEvaluator:
    @pytest.fixture
    def evaluator(self):
        from app.services.auto_strategy.core.hybrid.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        mock_backtest_service = Mock()
        with patch.object(
            HybridIndividualEvaluator, "_create_feature_adapter", return_value=Mock()
        ):
            yield HybridIndividualEvaluator(mock_backtest_service)

    def test_init_with_defaults(self, evaluator):
        assert evaluator.predictor is None
        assert evaluator.feature_adapter is not None

    def test_prepare_run_config_uses_ga_fallback_values(self, evaluator):
        ga_config = Mock(
            target_symbol="ETHUSDT",
            target_timeframe="4h",
            fallback_start_date="2024-01-01",
            fallback_end_date="2024-01-31",
        )

        gene = Mock()
        gene.id = "gene-123456789"

        result = evaluator._prepare_run_config(gene, {}, ga_config)

        assert result["symbol"] == "ETHUSDT"
        assert result["timeframe"] == "4h"
        assert result["start_date"] == "2024-01-01"
        assert result["end_date"] == "2024-01-31"

    def test_inject_external_objects_loads_runtime_predictor(self, evaluator):
        from app.services.auto_strategy.config.ga import GAConfig

        run_config = {"strategy_config": {"parameters": {}}}
        config = GAConfig(
            volatility_gate_enabled=True,
            volatility_model_path="dummy/model.pkl",
        )

        model_data = {
            "model": Mock(),
            "scaler": None,
            "feature_columns": ["close"],
            "metadata": {
                "task_type": "volatility_regression",
                "target_kind": "log_realized_vol",
                "gate_cutoff_log_rv": 0.1,
            },
        }
        model_data["model"].predict.return_value = pd.Series([0.2]).to_numpy()
        model_data["model"].is_trained = True

        with patch(
            "app.services.auto_strategy.core.hybrid.hybrid_individual_evaluator.model_manager.load_model",
            return_value=model_data,
        ):
            evaluator._inject_external_objects(run_config, {}, config)

        params = run_config["strategy_config"]["parameters"]
        assert params["volatility_gate_enabled"] is True
        assert "ml_predictor" in params
        prediction = params["ml_predictor"].predict(pd.DataFrame({"close": [100.0]}))
        assert prediction["gate_open"] is True

    def test_inject_external_objects_disables_gate_on_load_error(self, evaluator):
        from app.services.auto_strategy.config.ga import GAConfig

        run_config = {"strategy_config": {"parameters": {}}}
        config = GAConfig(
            volatility_gate_enabled=True,
            volatility_model_path="dummy/model.pkl",
        )

        with patch(
            "app.services.auto_strategy.core.hybrid.hybrid_individual_evaluator.model_manager.load_model",
            side_effect=Exception("load failed"),
        ):
            evaluator._inject_external_objects(run_config, {}, config)

        params = run_config["strategy_config"]["parameters"]
        assert params["volatility_gate_enabled"] is False

    def test_inject_external_objects_falls_back_to_runtime_predictor(self, evaluator):
        from app.services.auto_strategy.config.ga import GAConfig

        runtime_predictor = Mock()
        runtime_predictor.is_trained.return_value = True
        evaluator.predictor = runtime_predictor

        run_config = {"strategy_config": {"parameters": {}}}
        config = GAConfig(
            volatility_gate_enabled=True,
            volatility_model_path=None,
        )

        evaluator._inject_external_objects(run_config, {}, config)

        params = run_config["strategy_config"]["parameters"]
        assert params["volatility_gate_enabled"] is True
        assert params["ml_filter_enabled"] is True
        assert params["ml_predictor"] is runtime_predictor

    def test_fetch_ohlcv_data_preserves_non_ohlcv_columns(self, evaluator):
        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
        }

        raw_df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Volume": [10.0],
                "MarketRegime": [1],
                "StrategyName": ["demo"],
            }
        )

        with patch.object(
            evaluator,
            "_get_cached_ohlcv_data",
            return_value=raw_df,
        ):
            result = evaluator._fetch_ohlcv_data(backtest_config, Mock())

        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns
        assert "MarketRegime" in result.columns
        assert "StrategyName" in result.columns
        assert "Marketregime" not in result.columns
        assert "Strategyname" not in result.columns

    def test_build_robustness_cache_key_normalizes_regime_windows(self, evaluator):
        from app.services.auto_strategy.config.ga import GAConfig

        gene = Mock()
        evaluator._build_cache_key = Mock(return_value="gene-key")

        spaced_config = GAConfig(
            robustness_regime_windows=[
                {
                    "name": " bear ",
                    "start_date": " 2024-07-01 00:00:00 ",
                    "end_date": " 2024-08-01 00:00:00 ",
                }
            ]
        )
        trimmed_config = GAConfig(
            robustness_regime_windows=[
                {
                    "name": "bear",
                    "start_date": "2024-07-01 00:00:00",
                    "end_date": "2024-08-01 00:00:00",
                }
            ]
        )

        spaced_key = evaluator._build_robustness_cache_key(gene, spaced_config)
        trimmed_key = evaluator._build_robustness_cache_key(gene, trimmed_config)

        assert spaced_key == trimmed_key

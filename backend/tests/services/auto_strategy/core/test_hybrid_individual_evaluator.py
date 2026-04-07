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

    def test_ensure_backtest_defaults_uses_ga_fallback_values(self, evaluator):
        ga_config = Mock(
            target_symbol="ETHUSDT",
            target_timeframe="4h",
            fallback_start_date="2024-01-01",
            fallback_end_date="2024-01-31",
        )

        result = evaluator._ensure_backtest_defaults({}, ga_config)

        assert result["symbol"] == "ETHUSDT"
        assert result["timeframe"] == "4h"
        assert result["start_date"] == "2024-01-01"
        assert result["end_date"] == "2024-01-31"

    def test_calculate_fitness_delegates_to_base(self, evaluator):
        backtest_result = {
            "performance_metrics": {
                "total_trades": 10,
                "sharpe_ratio": 1.2,
            }
        }
        config = Mock()

        with patch.object(
            evaluator.__class__.__bases__[0], "_calculate_fitness", return_value=0.5
        ) as mock_base:
            result = evaluator._calculate_fitness(backtest_result, config)

        mock_base.assert_called_once()
        assert result == 0.5

    def test_calculate_multi_objective_fitness_delegates_to_base(self, evaluator):
        backtest_result = {"performance_metrics": {"total_trades": 10}}
        config = Mock()

        with patch.object(
            evaluator.__class__.__bases__[0],
            "_calculate_multi_objective_fitness",
            return_value=(0.3, 0.4),
        ) as mock_base:
            result = evaluator._calculate_multi_objective_fitness(
                backtest_result, config
            )

        mock_base.assert_called_once()
        assert result == (0.3, 0.4)

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

import pytest

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.config.sub_configs import (
    EarlyTerminationSettings,
    EvaluationConfig,
)


class TestGAConfig:
    """GAConfigのテスト"""

    def test_default_values(self):
        """デフォルト値が正しく設定されることを確認"""
        config = GAConfig()
        assert config.population_size > 0
        assert config.generations > 0
        assert config.crossover_rate > 0
        assert config.mutation_rate > 0
        assert config.indicator_universe_mode == "curated"

    def test_custom_values(self):
        """カスタム値を設定できることを確認"""
        config = GAConfig(
            population_size=200, generations=100, crossover_rate=0.9, mutation_rate=0.2
        )
        assert config.population_size == 200
        assert config.generations == 100
        assert config.crossover_rate == 0.9
        assert config.mutation_rate == 0.2

    def test_serialize_deserialize(self):
        """シリアライズとデシリアライズが正しく動作することを確認"""
        original = GAConfig(
            population_size=150,
            generations=75,
            indicator_universe_mode="experimental_all",
        )
        data = original.to_dict()
        assert data["population_size"] == 150
        assert data["generations"] == 75
        assert data["indicator_universe_mode"] == "experimental_all"

        restored = GAConfig(**data)
        assert restored.population_size == 150
        assert restored.generations == 75
        assert restored.indicator_universe_mode == "experimental_all"

    def test_two_stage_and_robustness_defaults(self):
        """二段階選抜と robustness のデフォルト設定を確認"""
        config = GAConfig()

        assert config.enable_two_stage_selection is True
        assert config.two_stage_elite_count == 3
        assert config.two_stage_candidate_pool_size == 5
        assert config.two_stage_min_pass_rate == 0.5
        assert config.robustness_validation_symbols is None
        assert config.robustness_stress_slippage == []
        assert config.robustness_stress_commission_multipliers == []
        assert config.robustness_aggregate_method == "robust"
        assert config.enable_multi_fidelity_evaluation is False
        assert config.multi_fidelity_window_ratio == 0.3
        assert config.enable_early_termination is False
        assert config.early_termination_expectancy_min_trades == 5

    def test_two_stage_and_robustness_serialize_deserialize(self):
        """二段階選抜/robustness 設定がシリアライズされることを確認"""
        original = GAConfig(
            enable_two_stage_selection=True,
            two_stage_elite_count=4,
            two_stage_candidate_pool_size=9,
            two_stage_min_pass_rate=0.75,
            robustness_validation_symbols=["ETH/USDT:USDT", "SOL/USDT:USDT"],
            robustness_stress_slippage=[0.0003, 0.0006],
            robustness_stress_commission_multipliers=[1.5, 2.0],
        )

        data = original.to_dict()
        restored = GAConfig(**data)

        assert restored.two_stage_elite_count == 4
        assert restored.two_stage_candidate_pool_size == 9
        assert restored.two_stage_min_pass_rate == 0.75
        assert restored.robustness_validation_symbols == [
            "ETH/USDT:USDT",
            "SOL/USDT:USDT",
        ]
        assert restored.robustness_stress_slippage == [0.0003, 0.0006]
        assert restored.robustness_stress_commission_multipliers == [1.5, 2.0]

    def test_multi_fidelity_and_early_termination_serialize_deserialize(self):
        """高速化設定がシリアライズされることを確認"""
        original = GAConfig(
            enable_multi_fidelity_evaluation=True,
            multi_fidelity_window_ratio=0.4,
            multi_fidelity_oos_ratio=0.15,
            multi_fidelity_candidate_ratio=0.3,
            multi_fidelity_min_candidates=4,
            enable_early_termination=True,
            early_termination_max_drawdown=0.2,
            early_termination_min_trades=12,
            early_termination_min_trade_check_progress=0.45,
            early_termination_trade_pace_tolerance=0.6,
            early_termination_min_expectancy=-0.02,
            early_termination_expectancy_min_trades=6,
            early_termination_expectancy_progress=0.7,
        )

        restored = GAConfig(**original.to_dict())

        assert restored.enable_multi_fidelity_evaluation is True
        assert restored.multi_fidelity_window_ratio == 0.4
        assert restored.multi_fidelity_oos_ratio == 0.15
        assert restored.multi_fidelity_candidate_ratio == 0.3
        assert restored.multi_fidelity_min_candidates == 4
        assert restored.enable_early_termination is True
        assert restored.early_termination_max_drawdown == 0.2
        assert restored.early_termination_min_trades == 12
        assert restored.early_termination_min_trade_check_progress == 0.45
        assert restored.early_termination_trade_pace_tolerance == 0.6
        assert restored.early_termination_min_expectancy == -0.02
        assert restored.early_termination_expectancy_min_trades == 6
        assert restored.early_termination_expectancy_progress == 0.7

    def test_from_dict_expands_nested_two_stage_and_robustness_config(self):
        """ネスト設定からフラット設定へ復元されることを確認"""
        restored = GAConfig.from_dict(
            {
                "two_stage_selection_config": {
                    "enabled": True,
                    "elite_count": 6,
                    "candidate_pool_size": 12,
                    "min_pass_rate": 0.8,
                },
                "robustness_config": {
                    "validation_symbols": ["ETH/USDT:USDT"],
                    "stress_slippage": [0.0004],
                    "stress_commission_multipliers": [1.5],
                    "aggregate_method": "mean",
                },
            }
        )

        assert restored.two_stage_elite_count == 6
        assert restored.two_stage_candidate_pool_size == 12
        assert restored.two_stage_min_pass_rate == 0.8
        assert restored.robustness_validation_symbols == ["ETH/USDT:USDT"]
        assert restored.robustness_stress_slippage == [0.0004]
        assert restored.robustness_stress_commission_multipliers == [1.5]
        assert restored.robustness_aggregate_method == "mean"

    def test_from_dict_expands_all_nested_runtime_configs(self):
        """各種ネスト設定が対応するフラット設定へ復元されることを確認"""
        restored = GAConfig.from_dict(
            {
                "mutation_config": {
                    "rate": 0.25,
                    "indicator_param_range": [0.7, 1.4],
                    "risk_param_range": [0.8, 1.2],
                    "indicator_add_delete_probability": 0.45,
                    "indicator_add_vs_delete_probability": 0.35,
                    "condition_change_multiplier": 1.3,
                    "condition_selection_probability": 0.6,
                    "condition_operator_switch_probability": 0.4,
                    "tpsl_gene_creation_multiplier": 0.55,
                    "position_sizing_gene_creation_multiplier": 0.65,
                    "adaptive_variance_threshold": 0.02,
                    "adaptive_decrease_multiplier": 0.75,
                    "adaptive_increase_multiplier": 1.4,
                    "valid_condition_operators": [">", "<", "CROSS_UP"],
                },
                "evaluation_config": {
                    "enable_parallel": False,
                    "max_workers": 3,
                    "timeout": 120.0,
                    "enable_multi_fidelity_evaluation": True,
                    "multi_fidelity_window_ratio": 0.4,
                    "multi_fidelity_oos_ratio": 0.3,
                    "multi_fidelity_candidate_ratio": 0.5,
                    "multi_fidelity_min_candidates": 7,
                    "oos_split_ratio": 0.25,
                    "oos_fitness_weight": 0.6,
                    "enable_walk_forward": True,
                    "wfa_n_folds": 4,
                    "wfa_train_ratio": 0.8,
                    "wfa_anchored": True,
                },
                "hybrid_config": {
                    "mode": True,
                    "model_type": "xgboost",
                    "model_types": ["xgboost", "lightgbm"],
                    "volatility_gate_enabled": True,
                    "volatility_model_path": "vol.pkl",
                    "ml_filter_enabled": True,
                    "ml_model_path": "ml.pkl",
                    "preprocess_features": False,
                },
                "tuning_config": {
                    "enabled": False,
                    "n_trials": 11,
                    "elite_count": 2,
                    "use_wfa": False,
                    "include_indicators": False,
                    "include_tpsl": False,
                    "include_thresholds": True,
                },
            }
        )

        assert restored.mutation_rate == 0.25
        assert restored.indicator_param_mutation_range == [0.7, 1.4]
        assert restored.risk_param_mutation_range == [0.8, 1.2]
        assert restored.indicator_add_delete_probability == 0.45
        assert restored.indicator_add_vs_delete_probability == 0.35
        assert restored.condition_change_probability_multiplier == 1.3
        assert restored.condition_selection_probability == 0.6
        assert restored.condition_operator_switch_probability == 0.4
        assert restored.tpsl_gene_creation_probability_multiplier == 0.55
        assert restored.position_sizing_gene_creation_probability_multiplier == 0.65
        assert restored.adaptive_mutation_variance_threshold == 0.02
        assert restored.adaptive_mutation_rate_decrease_multiplier == 0.75
        assert restored.adaptive_mutation_rate_increase_multiplier == 1.4
        assert restored.valid_condition_operators == [">", "<", "CROSS_UP"]

        assert restored.enable_parallel_evaluation is False
        assert restored.max_evaluation_workers == 3
        assert restored.evaluation_timeout == 120.0
        assert restored.enable_multi_fidelity_evaluation is True
        assert restored.multi_fidelity_window_ratio == 0.4
        assert restored.multi_fidelity_oos_ratio == 0.3
        assert restored.multi_fidelity_candidate_ratio == 0.5
        assert restored.multi_fidelity_min_candidates == 7
        assert restored.oos_split_ratio == 0.25
        assert restored.oos_fitness_weight == 0.6
        assert restored.enable_walk_forward is True
        assert restored.wfa_n_folds == 4
        assert restored.wfa_train_ratio == 0.8
        assert restored.wfa_anchored is True

        assert restored.hybrid_mode is True
        assert restored.hybrid_model_type == "xgboost"
        assert restored.hybrid_model_types == ["xgboost", "lightgbm"]
        assert restored.volatility_gate_enabled is True
        assert restored.volatility_model_path == "vol.pkl"
        assert restored.ml_filter_enabled is True
        assert restored.ml_model_path == "vol.pkl"
        assert restored.preprocess_features is False

        assert restored.enable_parameter_tuning is False
        assert restored.tuning_n_trials == 11
        assert restored.tuning_elite_count == 2
        assert restored.tuning_use_wfa is False
        assert restored.tuning_include_indicators is False
        assert restored.tuning_include_tpsl is False
        assert restored.tuning_include_thresholds is True

    def test_from_dict_rejects_unknown_keys(self):
        """未知のキーは早期にエラーにすることを確認"""
        with pytest.raises(ValueError, match="未対応の設定キー"):
            GAConfig.from_dict(
                {
                    "population_size": 100,
                    "unknown_field": 1,
                }
            )

    def test_early_termination_sync_does_not_mutate_evaluation_config(self):
        """GAConfig の早期終了設定は evaluation_config へは自動同期しない。"""
        config = GAConfig(
            early_termination_settings=EarlyTerminationSettings(
                enabled=True,
                max_drawdown=0.2,
                min_trades=12,
                min_trade_check_progress=0.45,
                trade_pace_tolerance=0.6,
                min_expectancy=-0.02,
                expectancy_min_trades=6,
                expectancy_progress=0.7,
            ),
            evaluation_config=EvaluationConfig(
                early_termination_settings=EarlyTerminationSettings(
                    enabled=False,
                    max_drawdown=0.9,
                )
            ),
        )

        assert config.enable_early_termination is True
        assert config.early_termination_max_drawdown == 0.2
        assert config.evaluation_config is not None
        assert config.evaluation_config.early_termination_settings.enabled is False
        assert config.evaluation_config.early_termination_settings.max_drawdown == 0.9

    def test_mutation_settings_defaults(self):
        """突然変異関連のデフォルト設定が正しいことを確認"""
        config = GAConfig()
        # GAConfig uses List instead of Tuple
        assert config.indicator_param_mutation_range == [0.8, 1.2]
        assert config.indicator_add_delete_probability == 0.3
        assert config.indicator_add_vs_delete_probability == 0.5
        assert config.crossover_field_selection_probability == 0.5
        assert config.condition_operator_switch_probability == 0.2  # GAConfig default
        assert config.condition_change_probability_multiplier == 1.0  # GAConfig default
        assert config.condition_selection_probability == 0.5
        assert config.risk_param_mutation_range == [0.9, 1.1]  # GAConfig default
        assert config.tpsl_gene_creation_probability_multiplier == 0.2
        assert config.position_sizing_gene_creation_probability_multiplier == 0.2
        assert config.adaptive_mutation_variance_threshold == 0.001  # GAConfig default
        assert (
            config.adaptive_mutation_rate_decrease_multiplier == 0.8
        )  # GAConfig default
        assert (
            config.adaptive_mutation_rate_increase_multiplier == 1.2
        )  # GAConfig default
        # GAConfig has extended operators including CROSS_UP/CROSS_DOWN
        assert ">=" in config.valid_condition_operators
        assert config.numeric_threshold_probability == 0.8  # GAConfig default
        assert config.min_compatibility_score == 0.8  # GAConfig default
        assert config.strict_compatibility_score == 0.9  # GAConfig default
        assert config.enable_multi_timeframe is False
        assert config.mtf_indicator_probability == 0.3
        assert config.available_timeframes is None  # GAConfig default

    def test_mutation_settings_custom_values(self):
        """突然変異関連のカスタム設定が正しく適用されることを確認"""
        custom_range = [0.9, 1.1]  # List for GAConfig
        custom_prob = 0.4
        config = GAConfig(
            indicator_param_mutation_range=custom_range,
            indicator_add_delete_probability=custom_prob,
            indicator_add_vs_delete_probability=0.3,
            crossover_field_selection_probability=0.7,
            condition_operator_switch_probability=0.6,
            condition_change_probability_multiplier=0.7,
            condition_selection_probability=0.8,
            risk_param_mutation_range=[0.7, 1.3],  # List for GAConfig
            tpsl_gene_creation_probability_multiplier=0.1,
            position_sizing_gene_creation_probability_multiplier=0.3,
            adaptive_mutation_variance_threshold=0.05,
            adaptive_mutation_rate_decrease_multiplier=0.6,
            adaptive_mutation_rate_increase_multiplier=1.5,
            valid_condition_operators=["==", "!="],
            numeric_threshold_probability=0.5,
            min_compatibility_score=0.6,
            strict_compatibility_score=0.8,
            enable_multi_timeframe=True,
            mtf_indicator_probability=0.5,
            available_timeframes=["1h", "4h"],
        )
        assert config.indicator_param_mutation_range == custom_range
        assert config.indicator_add_delete_probability == custom_prob
        assert config.condition_operator_switch_probability == 0.6
        assert config.condition_change_probability_multiplier == 0.7
        assert config.risk_param_mutation_range == [0.7, 1.3]
        assert config.tpsl_gene_creation_probability_multiplier == 0.1
        assert config.position_sizing_gene_creation_probability_multiplier == 0.3
        assert config.adaptive_mutation_variance_threshold == 0.05
        assert config.adaptive_mutation_rate_decrease_multiplier == 0.6
        assert config.adaptive_mutation_rate_increase_multiplier == 1.5
        assert config.valid_condition_operators == ["==", "!="]

    def test_parameter_range_defaults_are_isolated(self):
        """parameter_ranges のネスト値がインスタンス間で共有されないことを確認"""
        first = GAConfig()
        second = GAConfig()

        first.parameter_ranges["period"][0] = 999

        assert second.parameter_ranges["period"] == [5, 200]

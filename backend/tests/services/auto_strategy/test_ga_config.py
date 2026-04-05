import pytest

from app.services.auto_strategy.config import GAConfig


class TestGAConfig:
    """GAConfigのテスト"""

    def test_default_values(self):
        """デフォルト値が正しく設定されることを確認"""
        config = GAConfig()
        assert config.population_size > 0
        assert config.generations > 0
        assert config.crossover_rate > 0
        assert config.mutation_rate > 0

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
        original = GAConfig(population_size=150, generations=75)
        data = original.to_dict()
        assert data["population_size"] == 150
        assert data["generations"] == 75

        restored = GAConfig(**data)
        assert restored.population_size == 150
        assert restored.generations == 75

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

    def test_from_dict_rejects_unknown_keys(self):
        """未知のキーは早期にエラーにすることを確認"""
        with pytest.raises(ValueError, match="未対応の設定キー"):
            GAConfig.from_dict(
                {
                    "population_size": 100,
                    "unknown_field": 1,
                }
            )

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

from app.services.auto_strategy.config.ga import GASettings


class TestGAConfig:
    """GASettings（旧GAConfig）のテスト"""

    def test_default_values(self):
        """デフォルト値が正しく設定されることを確認"""
        config = GASettings()
        assert config.population_size > 0
        assert config.generations > 0
        assert config.crossover_rate > 0
        assert config.mutation_rate > 0

    def test_custom_values(self):
        """カスタム値を設定できることを確認"""
        config = GASettings(
            population_size=200, generations=100, crossover_rate=0.9, mutation_rate=0.2
        )
        assert config.population_size == 200
        assert config.generations == 100
        assert config.crossover_rate == 0.9
        assert config.mutation_rate == 0.2

    def test_serialize_deserialize(self):
        """シリアライズとデシリアライズが正しく動作することを確認"""
        original = GASettings(population_size=150, generations=75)
        data = original.to_dict()
        assert data["population_size"] == 150
        assert data["generations"] == 75

        restored = GASettings(**data)
        assert restored.population_size == 150
        assert restored.generations == 75

    def test_mutation_settings_defaults(self):
        """突然変異関連のデフォルト設定が正しいことを確認"""
        config = GASettings()
        assert config.indicator_param_mutation_range == (0.8, 1.2)
        assert config.indicator_add_delete_probability == 0.3
        assert config.indicator_add_vs_delete_probability == 0.5
        assert config.crossover_field_selection_probability == 0.5
        assert config.condition_operator_switch_probability == 0.5
        assert config.condition_change_probability_multiplier == 0.5
        assert config.condition_selection_probability == 0.5
        assert config.risk_param_mutation_range == (0.8, 1.2)
        assert config.tpsl_gene_creation_probability_multiplier == 0.2
        assert config.position_sizing_gene_creation_probability_multiplier == 0.2
        assert config.adaptive_mutation_variance_threshold == 0.1
        assert config.adaptive_mutation_rate_decrease_multiplier == 0.5
        assert config.adaptive_mutation_rate_increase_multiplier == 2.0
        assert config.valid_condition_operators == [">", "<", ">=", "<=", "=="]
        assert config.numeric_threshold_probability == 0.3
        assert config.min_compatibility_score == 0.5
        assert config.strict_compatibility_score == 0.7
        assert config.enable_multi_timeframe is False
        assert config.mtf_indicator_probability == 0.3
        assert config.available_timeframes == []

    def test_mutation_settings_custom_values(self):
        """突然変異関連のカスタム設定が正しく適用されることを確認"""
        custom_range = (0.9, 1.1)
        custom_prob = 0.4
        config = GASettings(
            indicator_param_mutation_range=custom_range,
            indicator_add_delete_probability=custom_prob,
            indicator_add_vs_delete_probability=0.3,
            crossover_field_selection_probability=0.7,
            condition_operator_switch_probability=0.6,
            condition_change_probability_multiplier=0.7,
            condition_selection_probability=0.8,
            risk_param_mutation_range=(0.7, 1.3),
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
            available_timeframes=["1h", "4h"]
        )
        assert config.indicator_param_mutation_range == custom_range
        assert config.indicator_add_delete_probability == custom_prob
        assert config.condition_operator_switch_probability == 0.6
        assert config.condition_change_probability_multiplier == 0.7
        assert config.risk_param_mutation_range == (0.7, 1.3)
        assert config.tpsl_gene_creation_probability_multiplier == 0.1
        assert config.position_sizing_gene_creation_probability_multiplier == 0.3
        assert config.adaptive_mutation_variance_threshold == 0.05
        assert config.adaptive_mutation_rate_decrease_multiplier == 0.6
        assert config.adaptive_mutation_rate_increase_multiplier == 1.5
        assert config.valid_condition_operators == ["==", "!="]




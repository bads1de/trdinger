import pytest

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.config.ga.nested_configs import (
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

        assert config.two_stage_selection_config.enabled is True
        assert config.two_stage_selection_config.elite_count == 3
        assert config.two_stage_selection_config.candidate_pool_size == 5
        assert config.two_stage_selection_config.min_pass_rate == 0.5
        assert config.robustness_config.validation_symbols is None
        assert config.robustness_config.stress_slippage == []
        assert config.robustness_config.stress_commission_multipliers == []
        assert config.robustness_config.aggregate_method == "robust"
        assert config.evaluation_config.enable_multi_fidelity_evaluation is False
        assert config.evaluation_config.multi_fidelity_window_ratio == 0.3
        assert config.evaluation_config.early_termination_settings.enabled is False
        assert config.evaluation_config.early_termination_settings.expectancy_min_trades == 5

    def test_two_stage_and_robustness_serialize_deserialize(self):
        """二段階選抜/robustness 設定がシリアライズされることを確認"""
        from app.services.auto_strategy.config.ga.nested_configs import (
            RobustnessConfig,
            TwoStageSelectionConfig,
        )

        original = GAConfig(
            two_stage_selection_config=TwoStageSelectionConfig(
                enabled=True,
                elite_count=4,
                candidate_pool_size=9,
                min_pass_rate=0.75,
            ),
            robustness_config=RobustnessConfig(
                validation_symbols=["ETH/USDT:USDT", "SOL/USDT:USDT"],
                stress_slippage=[0.0003, 0.0006],
                stress_commission_multipliers=[1.5, 2.0],
            ),
        )

        data = original.to_dict()
        restored = GAConfig.from_dict(data)

        assert restored.two_stage_selection_config.elite_count == 4
        assert restored.two_stage_selection_config.candidate_pool_size == 9
        assert restored.two_stage_selection_config.min_pass_rate == 0.75
        assert restored.robustness_config.validation_symbols == [
            "ETH/USDT:USDT",
            "SOL/USDT:USDT",
        ]
        assert restored.robustness_config.stress_slippage == [0.0003, 0.0006]
        assert restored.robustness_config.stress_commission_multipliers == [1.5, 2.0]

    def test_multi_fidelity_and_early_termination_serialize_deserialize(self):
        """高速化設定がシリアライズされることを確認"""
        from app.services.auto_strategy.config.ga.nested_configs import EvaluationConfig

        original = GAConfig(
            evaluation_config=EvaluationConfig(
                enable_multi_fidelity_evaluation=True,
                multi_fidelity_window_ratio=0.4,
                multi_fidelity_oos_ratio=0.15,
                multi_fidelity_candidate_ratio=0.3,
                multi_fidelity_min_candidates=4,
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
            ),
        )

        restored = GAConfig.from_dict(original.to_dict())

        assert restored.evaluation_config.enable_multi_fidelity_evaluation is True
        assert restored.evaluation_config.multi_fidelity_window_ratio == 0.4
        assert restored.evaluation_config.multi_fidelity_oos_ratio == 0.15
        assert restored.evaluation_config.multi_fidelity_candidate_ratio == 0.3
        assert restored.evaluation_config.multi_fidelity_min_candidates == 4
        assert restored.evaluation_config.early_termination_settings.enabled is True
        assert restored.evaluation_config.early_termination_settings.max_drawdown == 0.2
        assert restored.evaluation_config.early_termination_settings.min_trades == 12
        assert restored.evaluation_config.early_termination_settings.min_trade_check_progress == 0.45
        assert restored.evaluation_config.early_termination_settings.trade_pace_tolerance == 0.6
        assert restored.evaluation_config.early_termination_settings.min_expectancy == -0.02
        assert restored.evaluation_config.early_termination_settings.expectancy_min_trades == 6
        assert restored.evaluation_config.early_termination_settings.expectancy_progress == 0.7

    def test_from_dict_expands_nested_two_stage_and_robustness_config(self):
        """ネスト設定が正しく復元されることを確認"""
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

        assert restored.two_stage_selection_config.elite_count == 6
        assert restored.two_stage_selection_config.candidate_pool_size == 12
        assert restored.two_stage_selection_config.min_pass_rate == 0.8
        assert restored.robustness_config.validation_symbols == ["ETH/USDT:USDT"]
        assert restored.robustness_config.stress_slippage == [0.0004]
        assert restored.robustness_config.stress_commission_multipliers == [1.5]
        assert restored.robustness_config.aggregate_method == "mean"

    def test_from_dict_expands_all_nested_runtime_configs(self):
        """各種ネスト設定が正しく復元されることを確認"""
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
                    "include_thresholds": True,
                },
            }
        )

        assert restored.mutation_config.rate == 0.25
        assert restored.mutation_config.indicator_param_range == [0.7, 1.4]
        assert restored.mutation_config.risk_param_range == [0.8, 1.2]
        assert restored.mutation_config.indicator_add_delete_probability == 0.45
        assert restored.mutation_config.indicator_add_vs_delete_probability == 0.35
        assert restored.mutation_config.condition_change_multiplier == 1.3
        assert restored.mutation_config.condition_selection_probability == 0.6
        assert restored.mutation_config.condition_operator_switch_probability == 0.4
        assert restored.mutation_config.tpsl_gene_creation_multiplier == 0.55
        assert restored.mutation_config.position_sizing_gene_creation_multiplier == 0.65
        assert restored.mutation_config.adaptive_variance_threshold == 0.02
        assert restored.mutation_config.adaptive_decrease_multiplier == 0.75
        assert restored.mutation_config.adaptive_increase_multiplier == 1.4
        assert restored.mutation_config.valid_condition_operators == [">", "<", "CROSS_UP"]

        assert restored.evaluation_config.enable_parallel is False
        assert restored.evaluation_config.max_workers == 3
        assert restored.evaluation_config.timeout == 120.0
        assert restored.evaluation_config.enable_multi_fidelity_evaluation is True
        assert restored.evaluation_config.multi_fidelity_window_ratio == 0.4
        assert restored.evaluation_config.multi_fidelity_oos_ratio == 0.3
        assert restored.evaluation_config.multi_fidelity_candidate_ratio == 0.5
        assert restored.evaluation_config.multi_fidelity_min_candidates == 7
        assert restored.evaluation_config.oos_split_ratio == 0.25
        assert restored.evaluation_config.oos_fitness_weight == 0.6
        assert restored.evaluation_config.enable_walk_forward is True
        assert restored.evaluation_config.wfa_n_folds == 4
        assert restored.evaluation_config.wfa_train_ratio == 0.8
        assert restored.evaluation_config.wfa_anchored is True

        assert restored.hybrid_config.mode is True
        assert restored.hybrid_config.model_type == "xgboost"
        assert restored.hybrid_config.model_types == ["xgboost", "lightgbm"]
        assert restored.hybrid_config.volatility_gate_enabled is True
        assert restored.hybrid_config.volatility_model_path == "vol.pkl"
        assert restored.hybrid_config.ml_filter_enabled is True
        assert restored.hybrid_config.ml_model_path == "ml.pkl"
        assert restored.hybrid_config.preprocess_features is False

        assert restored.tuning_config.enabled is False
        assert restored.tuning_config.n_trials == 11
        assert restored.tuning_config.elite_count == 2
        assert restored.tuning_config.use_wfa is False
        assert restored.tuning_config.include_thresholds is True

    def test_from_dict_ignores_legacy_tuning_flags(self):
        """削除済みの tuning flags は警告付きで無視されることを確認"""
        restored = GAConfig.from_dict(
            {
                "tuning_config": {
                    "n_trials": 9,
                    "use_wfa": False,
                    "include_indicators": False,
                    "include_tpsl": False,
                    "include_thresholds": True,
                }
            }
        )

        assert restored.tuning_config.n_trials == 9
        assert restored.tuning_config.use_wfa is False
        assert restored.tuning_config.include_thresholds is True

    def test_from_dict_rejects_legacy_flat_payload(self):
        """旧フラット形式の GA 設定は受け付けないことを確認"""
        with pytest.raises(ValueError, match="未対応の設定キー"):
            GAConfig.from_dict(
                {
                    "population_size": 128,
                    "enable_parallel_evaluation": False,
                    "max_evaluation_workers": 4,
                    "evaluation_timeout": 90.0,
                    "enable_parameter_tuning": False,
                    "tuning_n_trials": 18,
                    "tuning_elite_count": 6,
                    "enable_two_stage_selection": True,
                    "two_stage_elite_count": 5,
                    "two_stage_candidate_pool_size": 11,
                    "two_stage_min_pass_rate": 0.7,
                    "enable_early_termination": True,
                    "early_termination_max_drawdown": 0.2,
                    "early_termination_min_trades": 14,
                    "enable_fitness_sharing": True,
                    "sharing_radius": 0.25,
                    "sharing_alpha": 1.3,
                    "sampling_threshold": 80,
                    "robustness_validation_symbols": ["BTC/USDT:USDT"],
                    "robustness_stress_slippage": [0.0005],
                    "robustness_aggregate_method": "mean",
                }
            )

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
            evaluation_config=EvaluationConfig(
                early_termination_settings=EarlyTerminationSettings(
                    enabled=True,
                    max_drawdown=0.2,
                    min_trades=12,
                    min_trade_check_progress=0.45,
                    trade_pace_tolerance=0.6,
                    min_expectancy=-0.02,
                    expectancy_min_trades=6,
                    expectancy_progress=0.7,
                )
            ),
        )

        assert config.evaluation_config is not None
        assert config.evaluation_config.early_termination_settings.enabled is True
        assert config.evaluation_config.early_termination_settings.max_drawdown == 0.2

    def test_legacy_early_termination_assignment_is_rejected(self):
        """旧名の早期終了属性代入は拒否されることを確認"""
        config = GAConfig()

        with pytest.raises(AttributeError):
            config.enable_early_termination = True
        with pytest.raises(AttributeError):
            config.early_termination_max_drawdown = 0.25
        with pytest.raises(AttributeError):
            config.early_termination_min_trades = 11
        with pytest.raises(AttributeError):
            config.volatility_gate_enabled = True
        with pytest.raises(AttributeError):
            config.volatility_model_path = "/path/to/model.pkl"
        with pytest.raises(AttributeError):
            config.preprocess_features = False
        with pytest.raises(AttributeError):
            config.enable_parallel_evaluation = True
        with pytest.raises(AttributeError):
            config.max_evaluation_workers = 4
        with pytest.raises(AttributeError):
            config.evaluation_timeout = 60.0
        with pytest.raises(AttributeError):
            config.enable_multi_fidelity_evaluation = True
        with pytest.raises(AttributeError):
            config.multi_fidelity_window_ratio = 0.3
        with pytest.raises(AttributeError):
            config.multi_fidelity_oos_ratio = 0.2
        with pytest.raises(AttributeError):
            config.multi_fidelity_candidate_ratio = 0.25
        with pytest.raises(AttributeError):
            config.multi_fidelity_min_candidates = 3
        with pytest.raises(AttributeError):
            config.hybrid_mode = True
        with pytest.raises(AttributeError):
            config.hybrid_model_type = "lightgbm"
        with pytest.raises(AttributeError):
            config.hybrid_model_types = ["lightgbm", "xgboost"]
        with pytest.raises(AttributeError):
            config.oos_split_ratio = 0.25
        with pytest.raises(AttributeError):
            config.oos_fitness_weight = 0.5
        with pytest.raises(AttributeError):
            config.enable_walk_forward = True
        with pytest.raises(AttributeError):
            config.wfa_n_folds = 4
        with pytest.raises(AttributeError):
            config.wfa_train_ratio = 0.8
        with pytest.raises(AttributeError):
            config.wfa_anchored = True

        assert config.evaluation_config.early_termination_settings.enabled is False
        assert config.evaluation_config.early_termination_settings.max_drawdown is None
        assert config.evaluation_config.early_termination_settings.min_trades is None

    def test_mutation_settings_defaults(self):
        """突然変異関連のデフォルト設定が正しいことを確認"""
        config = GAConfig()
        # mutation_config経由で確認
        assert config.mutation_config.indicator_param_range == [0.8, 1.2]
        assert config.mutation_config.indicator_add_delete_probability == 0.3
        assert config.mutation_config.indicator_add_vs_delete_probability == 0.5
        assert config.mutation_config.crossover_field_selection_probability == 0.5
        assert config.mutation_config.condition_operator_switch_probability == 0.2
        assert config.mutation_config.condition_change_multiplier == 1.0
        assert config.mutation_config.condition_selection_probability == 0.5
        assert config.mutation_config.risk_param_range == [0.9, 1.1]
        assert config.mutation_config.tpsl_gene_creation_multiplier == 0.2
        assert config.mutation_config.position_sizing_gene_creation_multiplier == 0.2
        assert config.mutation_config.adaptive_variance_threshold == 0.001
        assert config.mutation_config.adaptive_decrease_multiplier == 0.8
        assert config.mutation_config.adaptive_increase_multiplier == 1.2
        assert ">=" in config.mutation_config.valid_condition_operators
        assert config.numeric_threshold_probability == 0.8
        assert config.min_compatibility_score == 0.8
        assert config.strict_compatibility_score == 0.9
        assert config.enable_multi_timeframe is False
        assert config.mtf_indicator_probability == 0.3
        assert config.available_timeframes is None

    def test_mutation_settings_custom_values(self):
        """突然変異関連のカスタム設定が正しく適用されることを確認"""
        from app.services.auto_strategy.config.ga.nested_configs import MutationConfig

        custom_range = [0.9, 1.1]
        custom_prob = 0.4
        config = GAConfig(
            mutation_config=MutationConfig(
                indicator_param_range=custom_range,
                indicator_add_delete_probability=custom_prob,
                indicator_add_vs_delete_probability=0.3,
                crossover_field_selection_probability=0.7,
                condition_operator_switch_probability=0.6,
                condition_change_multiplier=0.7,
                condition_selection_probability=0.8,
                risk_param_range=[0.7, 1.3],
                tpsl_gene_creation_multiplier=0.1,
                position_sizing_gene_creation_multiplier=0.3,
                adaptive_variance_threshold=0.05,
                adaptive_decrease_multiplier=0.6,
                adaptive_increase_multiplier=1.5,
                valid_condition_operators=["==", "!="],
            ),
            numeric_threshold_probability=0.5,
            min_compatibility_score=0.6,
            strict_compatibility_score=0.8,
            enable_multi_timeframe=True,
            mtf_indicator_probability=0.5,
            available_timeframes=["1h", "4h"],
        )
        assert config.mutation_config.indicator_param_range == custom_range
        assert config.mutation_config.indicator_add_delete_probability == custom_prob
        assert config.mutation_config.condition_operator_switch_probability == 0.6
        assert config.mutation_config.condition_change_multiplier == 0.7
        assert config.mutation_config.risk_param_range == [0.7, 1.3]
        assert config.mutation_config.tpsl_gene_creation_multiplier == 0.1
        assert config.mutation_config.position_sizing_gene_creation_multiplier == 0.3
        assert config.mutation_config.adaptive_variance_threshold == 0.05
        assert config.mutation_config.adaptive_decrease_multiplier == 0.6
        assert config.mutation_config.adaptive_increase_multiplier == 1.5
        assert config.mutation_config.valid_condition_operators == ["==", "!="]

    def test_parameter_range_defaults_are_isolated(self):
        """parameter_ranges のネスト値がインスタンス間で共有されないことを確認"""
        first = GAConfig()
        second = GAConfig()

        first.parameter_ranges["period"][0] = 999

        assert second.parameter_ranges["period"] == [5, 200]


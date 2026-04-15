from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.config import ConfigValidator


class TestConfigValidator:

    @pytest.fixture
    def base_config(self):
        return SimpleNamespace(
            validation_rules={
                "required_fields": ["field_a"],
                "ranges": {"field_b": (0, 10)},
                "types": {"field_c": int},
            },
            field_a="present",
            field_b=5,
            field_c=100,
        )

    @pytest.fixture
    def ga_config(self):
        # GAConfigは多数の属性を持つので、MagicMockで必要な属性を設定する
        config = MagicMock(spec=GAConfig)
        config.validation_rules = {}

        # 正常なGA設定
        config.population_size = 100
        config.generations = 50
        config.crossover_rate = 0.8
        config.mutation_rate = 0.1
        config.elite_size = 5

        config.fitness_weights = {
            "total_return": 0.4,
            "sharpe_ratio": 0.3,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }
        config.objectives = ["weighted_score"]
        config.objective_weights = [1.0]
        config.max_indicators = 5
        config.parameter_ranges = {"param1": [0, 10]}
        config.log_level = "INFO"
        config.parallel_processes = 4

        # サブ設定クラス
        config.evaluation_config = MagicMock()
        config.evaluation_config.enable_multi_fidelity_evaluation = False
        config.evaluation_config.multi_fidelity_window_ratio = 0.3
        config.evaluation_config.multi_fidelity_oos_ratio = 0.2
        config.evaluation_config.multi_fidelity_candidate_ratio = 0.25
        config.evaluation_config.multi_fidelity_min_candidates = 3
        config.evaluation_config.oos_split_ratio = 0.3
        config.evaluation_config.oos_fitness_weight = 0.6
        config.evaluation_config.enable_walk_forward = False
        config.evaluation_config.wfa_n_folds = 5
        config.evaluation_config.wfa_train_ratio = 0.7
        config.evaluation_config.wfa_anchored = True
        config.evaluation_config.enable_parallel = True
        config.evaluation_config.max_workers = 4
        config.evaluation_config.timeout = 300.0
        config.evaluation_config.early_termination_settings = MagicMock()
        config.evaluation_config.early_termination_settings.enabled = False
        config.evaluation_config.early_termination_settings.max_drawdown = None
        config.evaluation_config.early_termination_settings.min_trades = None
        config.evaluation_config.early_termination_settings.min_trade_check_progress = 0.5
        config.evaluation_config.early_termination_settings.trade_pace_tolerance = 0.5
        config.evaluation_config.early_termination_settings.min_expectancy = None
        config.evaluation_config.early_termination_settings.expectancy_min_trades = 5
        config.evaluation_config.early_termination_settings.expectancy_progress = 0.6

        config.two_stage_selection_config = MagicMock()
        config.two_stage_selection_config.enabled = True
        config.two_stage_selection_config.elite_count = 3
        config.two_stage_selection_config.candidate_pool_size = 5
        config.two_stage_selection_config.min_pass_rate = 0.5

        config.robustness_config = MagicMock()
        config.robustness_config.stress_slippage = [0.0003]
        config.robustness_config.stress_commission_multipliers = [1.5]
        config.robustness_config.aggregate_method = "robust"
        config.robustness_config.validation_symbols = None
        config.robustness_config.regime_windows = []

        return config

    def test_validate_base_success(self, base_config):
        is_valid, errors = ConfigValidator.validate(base_config)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_base_missing_required(self, base_config):
        base_config.field_a = None
        is_valid, errors = ConfigValidator.validate(base_config)
        assert is_valid is False
        assert any("必須フィールド 'field_a'" in e for e in errors)

    def test_validate_base_out_of_range(self, base_config):
        base_config.field_b = 11
        is_valid, errors = ConfigValidator.validate(base_config)
        assert is_valid is False
        assert any("'field_b' は 0 から 10 の範囲" in e for e in errors)

    def test_validate_base_wrong_type(self, base_config):
        base_config.field_c = "string"
        is_valid, errors = ConfigValidator.validate(base_config)
        assert is_valid is False
        assert any("'field_c' は int 型" in e for e in errors)

    def test_validate_base_exception(self, base_config):
        class FailingConfig:
            validation_rules = {"required_fields": ["field_a"]}

            @property
            def field_a(self):
                raise Exception("Test Error")

        # エラーはキャッチされてエラーメッセージに追加されるはず
        is_valid, errors = ConfigValidator.validate(FailingConfig())
        assert is_valid is False
        assert any("検証処理エラー: Test Error" in e for e in errors)

    def test_validate_ga_config_success(self, ga_config):
        # 実際のGAConfigインスタンスを使用
        config = GAConfig()
        is_valid, errors = ConfigValidator.validate(config)
        if not is_valid:
            print(f"Errors: {errors}")
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_ga_config_ranges(self, ga_config):
        # population_size オーバー
        ga_config.population_size = 2000
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any("個体数は1000以下" in e for e in errors)

        # crossover_rate マイナス
        ga_config.population_size = 100  # 戻す
        ga_config.crossover_rate = -0.1
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any("交叉率は0-1の範囲" in e for e in errors)

        # 型エラー
        ga_config.crossover_rate = "string"
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any("交叉率は数値" in e for e in errors)

    def test_validate_ga_config_elite_size(self, ga_config):
        ga_config.elite_size = 100  # population_sizeと同じ
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any("エリート保存数は0以上、個体数未満" in e for e in errors)

    def test_validate_ga_config_fitness_weights(self, ga_config):
        # 合計が1にならない
        ga_config.fitness_weights = {"total_return": 0.5}  # 他が足りない
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        # 合計チェックのエラーと、必須項目不足のエラーが出るはず
        assert any("フィットネス重みの合計は1.0" in e for e in errors)
        assert any("必要なメトリクスが不足" in e for e in errors)

        # 追加の制約が無い場合は、必須メトリクスと重みが揃っていれば有効
        ga_config.fitness_weights = {
            "total_return": 0.4,
            "sharpe_ratio": 0.3,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_ga_config_objective_weights_length(self, ga_config):
        """多目的最適化では objectives と objective_weights の数を一致させる"""
        ga_config.objectives = ["total_return", "max_drawdown"]
        ga_config.objective_weights = [1.0]

        is_valid, errors = ConfigValidator.validate(ga_config)

        assert is_valid is False
        assert any("objective_weights の数は objectives と一致" in e for e in errors)

    def test_validate_ga_config_parameter_ranges(self, ga_config):
        # リストでない
        ga_config.parameter_ranges = {"p1": "invalid"}
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any("範囲は [min, max] の形式" in e for e in errors)

        # min >= max
        ga_config.parameter_ranges = {"p1": [10, 5]}
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any("最小値は最大値より小さい" in e for e in errors)

    def test_validate_ga_config_log_level(self, ga_config):
        ga_config.log_level = "INVALID_LEVEL"
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any("無効なログレベル" in e for e in errors)

    def test_validate_ga_config_parallel_processes(self, ga_config):
        ga_config.parallel_processes = 0
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any("並列プロセス数は正の整数" in e for e in errors)

        ga_config.parallel_processes = 40
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any("並列プロセス数は32以下" in e for e in errors)

    def test_validate_ga_config_multi_fidelity(self, ga_config):
        ga_config.evaluation_config.enable_multi_fidelity_evaluation = True
        ga_config.evaluation_config.multi_fidelity_window_ratio = 0.0

        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any("multi_fidelity_window_ratio は0より大きく" in e for e in errors)

        ga_config.evaluation_config.multi_fidelity_window_ratio = 0.3
        ga_config.evaluation_config.multi_fidelity_candidate_ratio = 0.0

        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any("multi_fidelity_candidate_ratio は0より大きく" in e for e in errors)
        assert any("multi_fidelity_candidate_ratio" in e for e in errors)

    def test_validate_ga_config_early_termination(self, ga_config):
        ga_config.evaluation_config.early_termination_settings.enabled = True
        ga_config.evaluation_config.early_termination_settings.max_drawdown = 1.5

        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any(
            "evaluation_config.early_termination_settings.max_drawdown" in e
            for e in errors
        )

        ga_config.evaluation_config.early_termination_settings.max_drawdown = 0.2
        ga_config.evaluation_config.early_termination_settings.expectancy_min_trades = 0
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any(
            "evaluation_config.early_termination_settings.expectancy_min_trades"
            in e
            for e in errors
        )

    def test_validate_ga_config_two_stage_selection(self, ga_config):
        ga_config.two_stage_selection_config.elite_count = 0
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any("二段階選抜エリート数" in e for e in errors)

        ga_config.two_stage_selection_config.elite_count = 3
        ga_config.two_stage_selection_config.candidate_pool_size = 2
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any("二段階選抜候補数" in e for e in errors)

        ga_config.two_stage_selection_config.candidate_pool_size = 5
        ga_config.two_stage_selection_config.min_pass_rate = 1.5
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any("二段階選抜 pass rate" in e for e in errors)

    def test_validate_ga_config_robustness(self, ga_config):
        ga_config.robustness_config.stress_slippage = [-0.0001]
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any("robustness の slippage" in e for e in errors)

        ga_config.robustness_config.stress_slippage = [0.0002]
        ga_config.robustness_config.stress_commission_multipliers = [0.0]
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any("robustness の commission multiplier" in e for e in errors)

        ga_config.robustness_config.stress_commission_multipliers = [1.5]
        ga_config.robustness_config.aggregate_method = "unsupported"
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any(
            "robustness_config.aggregate_method" in e for e in errors
        )

    def test_validate_robustness_window_supports_z_suffix(self):
        errors = ConfigValidator._validate_robustness_window(
            {
                "name": "regime_z",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-02T00:00:00Z",
            }
        )

        assert errors == []

    def test_validate_robustness_window_supports_mixed_timezone_boundaries(self):
        errors = ConfigValidator._validate_robustness_window(
            {
                "name": "regime_mixed",
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-01-02T00:00:00Z",
            }
        )

        assert errors == []

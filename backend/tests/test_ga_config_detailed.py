"""
GAConfigのテスト
"""
import pytest
from unittest.mock import Mock

from app.services.auto_strategy.config import GAConfig


class TestGAConfig:
    """GAConfigのテストクラス"""

    def test_init_default(self):
        """デフォルト初期化のテスト"""
        config = GAConfig()

        # 実際のデフォルト値の検証 (ga_runtime.pyを参照)
        assert config.population_size == 100
        assert config.generations == 50
        assert config.crossover_rate == 0.8
        assert config.mutation_rate == 0.1
        assert config.elite_size == 10

    def test_init_custom(self):
        """カスタム初期化のテスト"""
        config = GAConfig(
            population_size=100,
            generations=50,
            crossover_rate=0.9,
            mutation_rate=0.05,
            elite_size=10
        )

        assert config.population_size == 100
        assert config.generations == 50
        assert config.crossover_rate == 0.9
        assert config.mutation_rate == 0.05
        assert config.elite_size == 10

    def test_from_dict(self):
        """辞書からの構築テスト"""
        config_dict = {
            "population_size": 75,
            "generations": 150,
            "crossover_rate": 0.85,
            "mutation_rate": 0.12,
            "elite_size": 7
        }

        config = GAConfig.from_dict(config_dict)

        assert config.population_size == 75
        assert config.generations == 150
        assert config.crossover_rate == 0.85
        assert config.mutation_rate == 0.12
        assert config.elite_size == 7

    def test_from_dict_partial(self):
        """部分辞書からの構築テスト"""
        config_dict = {
            "population_size": 120
        }

        config = GAConfig.from_dict(config_dict)

        assert config.population_size == 120
        # 他の値はデフォルト
        assert config.generations == 50

    def test_validate_success(self):
        """検証成功のテスト"""
        config = GAConfig()
        is_valid, errors = config.validate()

        assert is_valid is True
        assert errors == []

    def test_validate_invalid_population_size(self):
        """無効な集団サイズ検証のテスト"""
        config = GAConfig(population_size=0)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("population_size" in error for error in errors)

    def test_validate_invalid_generations(self):
        """無効な世代数検証のテスト"""
        config = GAConfig(generations=0)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("generations" in error for error in errors)

    def test_validate_invalid_crossover_rate(self):
        """無効な交叉率検証のテスト"""
        config = GAConfig(crossover_rate=1.5)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("crossover_rate" in error for error in errors)

    def test_validate_invalid_mutation_rate(self):
        """無効な突然変異率検証のテスト"""
        config = GAConfig(mutation_rate=-0.1)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("mutation_rate" in error for error in errors)

    def test_validate_invalid_elite_size(self):
        """無効なエリートサイズ検証のテスト"""
        config = GAConfig(elite_size=100, population_size=50)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("elite_size" in error for error in errors)

    def test_to_dict(self):
        """辞書変換のテスト"""
        config = GAConfig(
            population_size=80,
            generations=120,
            crossover_rate=0.75,
            mutation_rate=0.15,
            elite_size=8
        )

        config_dict = config.to_dict()

        assert config_dict["population_size"] == 80
        assert config_dict["generations"] == 120
        assert config_dict["crossover_rate"] == 0.75
        assert config_dict["mutation_rate"] == 0.15
        assert config_dict["elite_size"] == 8

    def test_enable_multi_objective(self):
        """多目的最適化有効化のテスト"""
        config = GAConfig()
        config.enable_multi_objective = True
        config.objectives = ["total_return", "sharpe_ratio", "max_drawdown"]

        assert config.enable_multi_objective is True
        assert "total_return" in config.objectives
        assert "sharpe_ratio" in config.objectives
        assert "max_drawdown" in config.objectives

    def test_enable_multi_objective_without_objectives(self):
        """目的なしの多目的最適化テスト"""
        config = GAConfig()
        config.enable_multi_objective = True

        # 目的が設定されていない場合のデフォルト値
        assert config.enable_multi_objective is True
        assert config.objectives == ["total_return"]  # デフォルト目的

    def test_set_fitness_weights(self):
        """フィットネス重み設定のテスト"""
        config = GAConfig()
        weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1
        }
        config.fitness_weights = weights

        assert config.fitness_weights == weights

    def test_set_fitness_constraints(self):
        """フィットネス制約設定のテスト"""
        config = GAConfig()
        constraints = {
            "min_sharpe_ratio": 0.5,
            "max_drawdown_limit": 0.2,
            "min_trades": 10
        }
        config.fitness_constraints = constraints

        assert config.fitness_constraints == constraints

    def test_enable_regime_adaptation(self):
        """レジーム適応有効化のテスト"""
        config = GAConfig()
        config.regime_adaptation_enabled = True

        assert config.regime_adaptation_enabled is True

    def test_set_indicator_constraints(self):
        """インジケータ制約設定のテスト"""
        config = GAConfig()
        constraints = {
            "max_indicators": 5,
            "allowed_indicators": ["SMA", "RSI", "MACD", "BB", "ATR"],
            "indicator_combinations": ["SMA+RSI", "MACD+BB"]
        }
        config.indicator_constraints = constraints

        assert config.indicator_constraints == constraints

    def test_set_position_sizing_method(self):
        """ポジションサイズ方法設定のテスト"""
        config = GAConfig()
        config.position_sizing_method = "fixed"

        assert config.position_sizing_method == "fixed"

    def test_invalid_position_sizing_method(self):
        """無効なポジションサイズ方法のテスト"""
        config = GAConfig()
        config.position_sizing_method = "invalid_method"

        # 無効な方法が設定できること（実際の検証は他の場所で行われる）
        assert config.position_sizing_method == "invalid_method"

    def test_enable_dynamic_parameter_adjustment(self):
        """動的パラメータ調整有効化のテスト"""
        config = GAConfig()
        config.dynamic_parameter_adjustment = True

        assert config.dynamic_parameter_adjustment is True

    def test_set_initial_capital(self):
        """初期資本設定のテスト"""
        config = GAConfig()
        config.initial_capital = 100000

        assert config.initial_capital == 100000

    def test_set_commission_rate(self):
        """手数料率設定のテスト"""
        config = GAConfig()
        config.commission_rate = 0.00055

        assert config.commission_rate == 0.00055

    def test_enable_overfitting_prevention(self):
        """過学習防止有効化のテスト"""
        config = GAConfig()
        config.overfitting_prevention_enabled = True

        assert config.overfitting_prevention_enabled is True

    def test_set_walk_forward_settings(self):
        """ウォークフォワード設定のテスト"""
        config = GAConfig()
        settings = {
            "in_sample_period": "2023-01-01:2023-12-31",
            "out_of_sample_period": "2024-01-01:2024-12-31",
            "rolling_window": True
        }
        config.walk_forward_settings = settings

        assert config.walk_forward_settings == settings

    def test_enable_parallel_evolution(self):
        """並列進化有効化のテスト"""
        config = GAConfig()
        config.parallel_evolution_enabled = True

        assert config.parallel_evolution_enabled is True

    def test_set_max_concurrent_processes(self):
        """最大同時プロセス数設定のテスト"""
        config = GAConfig()
        config.max_concurrent_processes = 4

        assert config.max_concurrent_processes == 4

    def test_set_logging_level(self):
        """ログレベル設定のテスト"""
        config = GAConfig()
        config.logging_level = "DEBUG"

        assert config.logging_level == "DEBUG"

    def test_set_max_runtime_minutes(self):
        """最大実行時間設定のテスト"""
        config = GAConfig()
        config.max_runtime_minutes = 60

        assert config.max_runtime_minutes == 60

    def test_complex_validation_scenario(self):
        """複雑な検証シナリオのテスト"""
        # 複数の無効な値を含む設定
        config = GAConfig(
            population_size=0,  # 無効
            generations=-10,    # 無効
            crossover_rate=2.0, # 無効
            mutation_rate=-0.1, # 無効
            elite_size=200      # 無効（集団サイズより大きい）
        )

        is_valid, errors = config.validate()

        assert is_valid is False
        assert len(errors) >= 5  # 5つの異なるエラーがあるはず

    def test_clone_config(self):
        """設定クローンのテスト"""
        config1 = GAConfig()
        config1.population_size = 100
        config1.generations = 50
        config1.enable_multi_objective = True
        config1.objectives = ["total_return", "sharpe_ratio"]

        config2 = config1.clone()

        assert config2.population_size == 100
        assert config2.generations == 50
        assert config2.enable_multi_objective is True
        assert config2.objectives == ["total_return", "sharpe_ratio"]

        # 別のインスタンスであることを確認
        assert config1 is not config2

    def test_reset_to_defaults(self):
        """デフォルト値リセットのテスト"""
        config = GAConfig()
        config.population_size = 200
        config.generations = 200
        config.crossover_rate = 0.9
        config.mutation_rate = 0.01
        config.elite_size = 20

        config.reset_to_defaults()

        assert config.population_size == 50
        assert config.generations == 50
        assert config.crossover_rate == 0.8
        assert config.mutation_rate == 0.1
        assert config.elite_size == 10

    def test_update_from_dict(self):
        """辞書からの更新テスト"""
        config = GAConfig()
        updates = {
            "population_size": 150,
            "generations": 75,
            "crossover_rate": 0.85
        }

        config.update_from_dict(updates)

        assert config.population_size == 150
        assert config.generations == 75
        assert config.crossover_rate == 0.85
        # 他の値は変わらない
        assert config.mutation_rate == 0.1
        assert config.elite_size == 10
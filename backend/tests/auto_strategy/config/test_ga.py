"""
テスト: GASettingsクラス

GASettingsクラスの機能をテストします。
TDD準拠で、基本機能からバグ検出のためのエッジケースまでテストします。
"""

import pytest
from unittest.mock import patch, Mock
from typing import Dict, Any

# テスト対象のクラス
from backend.app.services.auto_strategy.config.ga import GASettings
from backend.app.services.auto_strategy.config.ga import (
    GA_DEFAULT_CONFIG,
    DEFAULT_FITNESS_WEIGHTS,
    DEFAULT_FITNESS_CONSTRAINTS,
    GA_DEFAULT_FITNESS_SHARING,
    DEFAULT_GA_OBJECTIVES,
    DEFAULT_GA_OBJECTIVE_WEIGHTS,
    GA_PARAMETER_RANGES,
    GA_THRESHOLD_RANGES,
)


class TestGASettings:
    """GASettingsクラスのテスト"""

    def test_initialize_default(self):
        """デフォルト初期化テスト"""
        config = GASettings()

        assert config.population_size == GA_DEFAULT_CONFIG["population_size"]
        assert config.generations == GA_DEFAULT_CONFIG["generations"]
        assert config.crossover_rate == GA_DEFAULT_CONFIG["crossover_rate"]
        assert config.mutation_rate == GA_DEFAULT_CONFIG["mutation_rate"]
        assert config.elite_size == GA_DEFAULT_CONFIG["elite_size"]
        assert config.max_indicators == GA_DEFAULT_CONFIG["max_indicators"]

        # 制約設定
        assert config.min_indicators == 1
        assert config.min_conditions == 1
        assert config.max_conditions == 3

        # 多目的最適化設定
        assert config.enable_multi_objective is False
        assert isinstance(config.ga_objectives, list)
        assert isinstance(config.ga_objective_weights, list)

    def test_post_init_validation_success(self):
        """post_init正常テスト"""
        # 正常値
        config = GASettings(population_size=100, generations=50, elite_size=10, max_indicators=5)

        assert config.population_size == 100
        assert config.generations == 50
        assert config.elite_size == 10
        assert config.max_indicators == 5

    def test_post_init_population_size_invalid(self):
        """population_size無効値テスト"""
        with pytest.raises(ValueError, match="population_size は正の整数である必要があります"):
            GASettings(population_size=0)

    def test_post_init_generations_invalid(self):
        """generations無効値テスト"""
        with pytest.raises(ValueError, match="generations は正の整数である必要があります"):
            GASettings(generations=-1)

    def test_post_init_elite_size_negative(self):
        """elite_size負値テスト"""
        with pytest.raises(ValueError, match="elite_size は負でない整数である必要があります"):
            GASettings(elite_size=-5)

    def test_post_init_max_indicators_invalid(self):
        """max_indicators無効値テスト"""
        with pytest.raises(ValueError, match="max_indicators は正の整数である必要があります"):
            GASettings(max_indicators=0)

    def test_post_init_crossover_rate_invalid(self):
        """crossover_rate範囲外テスト"""
        with pytest.raises(ValueError, match="crossover_rate は0から1の範囲の実数である必要があります"):
            GASettings(crossover_rate=1.5)

    def test_post_init_mutation_rate_invalid(self):
        """mutation_rate範囲外テスト"""
        with pytest.raises(ValueError, match="mutation_rate は0から1の範囲の実数である必要があります"):
            GASettings(mutation_rate=-0.1)

    def test_validate_success(self):
        """正常検証テスト"""
        config = GASettings(population_size=50, generations=20)
        is_valid, errors = config.validate()

        assert is_valid is True
        assert errors == []

    def test_validate_population_size_invalid(self):
        """population_size検証失敗テスト - __post_init__でエラー"""
        with pytest.raises(ValueError, match="population_size は正の整数である必要があります"):
            GASettings(population_size=0)

    def test_validate_crossover_rate_invalid(self):
        """crossover_rate検証失敗テスト - __post_init__でエラー"""
        with pytest.raises(ValueError, match="crossover_rate は0から1の範囲の実数である必要があります"):
            GASettings(crossover_rate=1.5)

    def test_validate_mutation_rate_invalid(self):
        """mutation_rate検証失敗テスト - __post_init__でエラー"""
        with pytest.raises(ValueError, match="mutation_rate は0から1の範囲の実数である必要があります"):
            GASettings(mutation_rate=-0.1)

    def test_validate_elite_size_too_large(self):
        """elite_size過大テスト"""
        config = GASettings(population_size=20, elite_size=25)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("エリートサイズは人口サイズより小さく設定してください" in error for error in errors)

    def test_validate_min_indicators_too_large(self):
        """min_indicators過大テスト"""
        config = GASettings(min_indicators=10, max_indicators=5)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("最小指標数は最大指標数以下である必要があります" in error for error in errors)

    def test_get_default_values(self):
        """get_default_valuesテスト"""
        config = GASettings()
        defaults = config.get_default_values()

        assert isinstance(defaults, dict)
        assert "population_size" in defaults
        assert "generations" in defaults
        assert "crossover_rate" in defaults

        # フィールド自動生成されたデフォルト値を確認
        assert defaults["population_size"] == GA_DEFAULT_CONFIG["population_size"]
        assert defaults["generations"] == GA_DEFAULT_CONFIG["generations"]

    def test_to_dict_success(self):
        """正常な辞書変換テスト"""
        config = GASettings(population_size=100, generations=50)

        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["population_size"] == 100
        assert result["generations"] == 50
        assert "fitness_weights" in result

    def test_from_dict_success(self):
        """正常な辞書からの変換テスト"""
        data = {
            "population_size": 75,
            "generations": 25,
            "crossover_rate": 0.8,
            "fitness_weights": DEFAULT_FITNESS_WEIGHTS.copy(),
        }

        config = GASettings.from_dict(data)

        assert isinstance(config, GASettings)
        assert config.population_size == 75
        assert config.generations == 25
        assert config.crossover_rate == 0.8

    def test_to_json_from_json(self):
        """JSON変換テスト"""
        config = GASettings(population_size=60, generations=30)

        json_str = config.to_json()
        loaded_config = GASettings.from_json(json_str)

        assert isinstance(loaded_config, GASettings)
        assert loaded_config.population_size == 60
        assert loaded_config.generations == 30

    def test_parameters_and_ranges(self):
        """パラメータ範囲テスト"""
        config = GASettings()

        assert isinstance(config.parameter_ranges, dict)
        assert len(config.parameter_ranges) > 0
        assert isinstance(config.threshold_ranges, dict)
        assert len(config.threshold_ranges) > 0

    def test_fitness_settings(self):
        """フィットネス設定テスト"""
        config = GASettings()

        assert isinstance(config.fitness_weights, dict)
        assert len(config.fitness_weights) > 0
        assert isinstance(config.fitness_constraints, dict)
        assert isinstance(config.fitness_sharing, dict)

    def test_multi_objective_settings(self):
        """多目的最適化設定テスト"""
        config = GASettings()

        assert isinstance(config.ga_objectives, list)
        assert isinstance(config.ga_objective_weights, list)
        assert len(config.ga_objectives) > 0
        assert len(config.ga_objective_weights) > 0

    # エッジケーステスト
    @pytest.mark.parametrize("population_size", [0, -10, "invalid"])
    def test_invalid_population_sizes_parametrized(self, population_size):
        """不正なpopulation_sizeパラメータ化テスト"""
        with pytest.raises(ValueError, match="population_size は正の整数である必要があります"):
            GASettings(population_size=population_size)

    @pytest.mark.parametrize("crossover_rate", [-0.1, 1.5, "invalid"])
    def test_invalid_crossover_rates_parametrized(self, crossover_rate):
        """不正なcrossover_rateパラメータ化テスト"""
        if isinstance(crossover_rate, str):
            # 文字列の場合はpost_initでisinstanceチェックで落ちる
            with pytest.raises(ValueError, match="crossover_rate は0から1の範囲の実数である必要があります"):
                GASettings(crossover_rate=crossover_rate)
        else:
            # 数値範囲外はpost_initで落ちる
            with pytest.raises(ValueError, match="crossover_rate は0から1の範囲の実数である必要があります"):
                GASettings(crossover_rate=crossover_rate)

    @pytest.mark.parametrize("mutation_rate", [-0.5, 2.0, "invalid"])
    def test_invalid_mutation_rates_parametrized(self, mutation_rate):
        """不正なmutation_rateパラメータ化テスト"""
        if isinstance(mutation_rate, str):
            # 文字列の場合はpost_initでisinstanceチェックで落ちる
            with pytest.raises(ValueError, match="mutation_rate は0から1の範囲の実数である必要があります"):
                GASettings(mutation_rate=mutation_rate)
        else:
            # 数値範囲外はpost_initで落ちる
            with pytest.raises(ValueError, match="mutation_rate は0から1の範囲の実数である必要があります"):
                GASettings(mutation_rate=mutation_rate)
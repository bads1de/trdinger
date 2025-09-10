"""
テスト: GAConfigおよびGAProgressクラス

GAConfigとGAProgressクラスの機能をテストします。
TDD準拠で、基本機能からバグ検出のためのエッジケースまでテストします。
"""

import pytest
import json
from unittest.mock import patch, Mock
from typing import Dict, Any, List
from dataclasses import dataclass, field

# テスト対象のクラス
from backend.app.services.auto_strategy.config.ga_runtime import GAConfig, GAProgress
from backend.app.services.auto_strategy.config.auto_strategy import AutoStrategyConfig
from backend.app.services.auto_strategy.config.ga import (
    GA_DEFAULT_CONFIG,
    DEFAULT_FITNESS_WEIGHTS,
    GA_PARAMETER_RANGES,
    GA_THRESHOLD_RANGES,
)


class TestGAConfig:
    """GAConfigクラスのテスト"""

    def test_initialize_default(self):
        """デフォルト初期化テスト"""
        config = GAConfig()

        assert config.population_size == GA_DEFAULT_CONFIG["population_size"]
        assert config.generations == GA_DEFAULT_CONFIG["generations"]
        assert config.crossover_rate == GA_DEFAULT_CONFIG["crossover_rate"]
        assert config.mutation_rate == GA_DEFAULT_CONFIG["mutation_rate"]
        assert config.elite_size == GA_DEFAULT_CONFIG["elite_size"]
        assert config.max_indicators == GA_DEFAULT_CONFIG["max_indicators"]
        assert config.log_level == "ERROR"

    def test_post_init_validation_success(self):
        """post_init正常テスト"""
        # 正常値
        config = GAConfig(population_size=50, generations=20, elite_size=5, max_indicators=3)

        assert config.population_size == 50
        assert config.generations == 20
        assert config.elite_size == 5
        assert config.max_indicators == 3

    def test_post_init_population_size_invalid(self):
        """population_size無効値テスト"""
        with pytest.raises(ValueError, match="population_size は正の整数である必要があります"):
            GAConfig(population_size=0)

    def test_post_init_generations_invalid(self):
        """generations無効値テスト"""
        with pytest.raises(ValueError, match="generations は正の整数である必要があります"):
            GAConfig(generations=-1)

    def test_post_init_elite_size_negative(self):
        """elite_size負値テスト"""
        with pytest.raises(ValueError, match="elite_size は負でない整数である必要があります"):
            GAConfig(elite_size=-5)

    def test_post_init_max_indicators_invalid(self):
        """max_indicators無効値テスト"""
        with pytest.raises(ValueError, match="max_indicators は正の整数である必要があります"):
            GAConfig(max_indicators=0)

    def test_post_init_crossover_rate_invalid(self):
        """crossover_rate範囲外テスト"""
        with pytest.raises(ValueError, match="crossover_rate は0から1の範囲の実数である必要があります"):
            GAConfig(crossover_rate=1.5)

    def test_post_init_mutation_rate_invalid(self):
        """mutation_rate範囲外テスト"""
        with pytest.raises(ValueError, match="mutation_rate は0から1の範囲の実数である必要があります"):
            GAConfig(mutation_rate=-0.1)

    def test_validate_success(self):
        """正常検証テスト"""
        config = GAConfig(population_size=50, generations=20)
        is_valid, errors = config.validate()

        assert is_valid is True
        assert errors == []

    def test_validate_population_size_too_large(self):
        """population_size過大テスト"""
        config = GAConfig(population_size=2000)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("1000以下" in error for error in errors)

    def test_validate_generations_too_large(self):
        """generations過大テスト"""
        config = GAConfig(generations=1000)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("500以下" in error for error in errors)

    def test_validate_crossover_rate_invalid(self):
        """crossover_rate無効値テスト"""
        config = GAConfig(crossover_rate="invalid")
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("交叉率は0-1" in error for error in errors)

    def test_validate_mutation_rate_invalid(self):
        """mutation_rate無効値テスト"""
        config = GAConfig(mutation_rate="invalid")
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("突然変異率は0-1" in error for error in errors)

    def test_validate_elite_size_too_large(self):
        """elite_size過大テスト"""
        config = GAConfig(population_size=50, elite_size=60)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("エリート保存数は" in error for error in errors)

    def test_validate_max_indicators_too_large(self):
        """max_indicators過大テスト"""
        config = GAConfig(max_indicators=20)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("10以下" in error for error in errors)

    def test_validate_fitness_weights_sum_invalid(self):
        """fitness_weights合計無効テスト"""
        config = GAConfig(fitness_weights={"total_return": 0.6, "sharpe_ratio": 0.6})
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("1.0である必要があります" in error for error in errors)

    def test_validate_missing_required_metrics(self):
        """必要なメトリクス欠如テスト"""
        config = GAConfig(fitness_weights={"total_return": 1.0})
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("必要なメトリクスが不足" in error for error in errors)

    def test_validate_primary_metric_not_in_weights(self):
        """primary_metric不在テスト"""
        config = GAConfig(primary_metric="invalid_metric", fitness_weights=DEFAULT_FITNESS_WEIGHTS)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("プライマリメトリクス" in error for error in errors)

    def test_validate_allowed_indicators_empty(self):
        """allowed_indicators空テスト"""
        config = GAConfig(allowed_indicators=[])
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("許可された指標リストが空" in error for error in errors)

    def test_validate_parameter_ranges_invalid(self):
        """parameter_ranges無効テスト"""
        config = GAConfig(parameter_ranges={"invalid_param": [5, 3]})  # min > max
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("最小値は最大値より小さい" in error for error in errors)

    def test_validate_log_level_invalid(self):
        """log_level無効テスト"""
        config = GAConfig(log_level="INVALID")
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("無効なログレベル" in error for error in errors)

    def test_validate_parallel_processes_invalid(self):
        """parallel_processes無効テスト"""
        config = GAConfig(parallel_processes=-5)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("並列プロセス数は正の整数" in error for error in errors)

    def test_validate_parallel_processes_too_large(self):
        """parallel_processes過大テスト"""
        config = GAConfig(parallel_processes=50)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert any("32以下" in error for error in errors)

    def test_to_dict_success(self):
        """to_dict正常テスト"""
        config = GAConfig(population_size=100, generations=50)
        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["population_size"] == 100
        assert result["generations"] == 50
        assert "fitness_sharing" in result

    def test_from_dict_success(self):
        """from_dict正常テスト"""
        data = {
            "population_size": 75,
            "generations": 25,
            "crossover_rate": 0.8,
            "allowed_indicators": ["rsi", "macd"],
        }

        config = GAConfig.from_dict(data)

        assert isinstance(config, GAConfig)
        assert config.population_size == 75
        assert config.generations == 25
        assert config.crossover_rate == 0.8

    def test_preprocess_ga_dict_empty_indicators(self):
        """allowed_indicators空前処理テスト"""
        data = {"allowed_indicators": []}

        with patch("backend.app.services.auto_strategy.config.ga_runtime.TacticalIndicatorService", new_callable=Mock) as mock_service:
            mock_service_instance = Mock()
            mock_service_instance.get_supported_indicators.return_value = {"rsi": {}, "macd": {}}
            mock_service.return_value = mock_service_instance

            processed = GAConfig._preprocess_ga_dict(data)

            assert processed["allowed_indicators"] == ["rsi", "macd"]

    def test_preprocess_ga_dict_fitness_weights_empty(self):
        """fitness_weights空前処理テスト"""
        data = {"fitness_weights": {}}
        processed = GAConfig._preprocess_ga_dict(data)

        assert processed["fitness_weights"] == DEFAULT_FITNESS_WEIGHTS

    def test_apply_auto_strategy_config(self):
        """apply_auto_strategy_configテスト"""
        auto_config = Mock(spec=AutoStrategyConfig)
        auto_config.ga = Mock()
        auto_config.ga.population_size = 100
        auto_config.ga.generations = 30
        auto_config.ga.crossover_rate = 0.9

        ga_config = GAConfig()
        ga_config.apply_auto_strategy_config(auto_config)

        assert ga_config.population_size == 100
        assert ga_config.generations == 30

    def test_from_auto_strategy_config(self):
        """from_auto_strategy_configテスト"""
        auto_config = Mock(spec=AutoStrategyConfig)
        auto_config.ga = Mock()
        auto_config.ga.population_size = 150
        auto_config.ga.generations = 40

        ga_config = GAConfig.from_auto_strategy_config(auto_config)

        assert isinstance(ga_config, GAConfig)
        assert ga_config.population_size == 150
        assert ga_config.generations == 40

    def test_get_default_values(self):
        """get_default_valuesテスト"""
        config = GAConfig()
        defaults = config.get_default_values()

        assert isinstance(defaults, dict)
        assert "population_size" in defaults
        assert defaults["population_size"] == GA_DEFAULT_CONFIG["population_size"]

    def test_to_json_from_json(self):
        """JSON変換テスト"""
        config = GAConfig(population_size=60, generations=30)

        json_str = config.to_json()
        loaded_config = GAConfig.from_json(json_str)

        assert isinstance(loaded_config, GAConfig)
        assert loaded_config.population_size == 60
        assert loaded_config.generations == 30

    def test_create_default(self):
        """create_defaultテスト"""
        config = GAConfig.create_default()

        assert isinstance(config, GAConfig)
        assert config.population_size == GA_DEFAULT_CONFIG["population_size"]

    def test_create_fast(self):
        """create_fastテスト"""
        config = GAConfig.create_fast()

        assert config.population_size == 10
        assert config.generations == 5
        assert config.elite_size == 2
        assert config.max_indicators == 3

    def test_create_thorough(self):
        """create_thoroughテスト"""
        config = GAConfig.create_thorough()

        assert config.population_size == 200
        assert config.generations == 100
        assert config.crossover_rate == 0.85
        assert config.mutation_rate == 0.05
        assert config.elite_size == 20
        assert config.max_indicators == 5

    def test_create_multi_objective(self):
        """create_multi_objectiveテスト"""
        config = GAConfig.create_multi_objective()

        assert config.enable_multi_objective is True
        assert config.objectives == ["total_return", "max_drawdown"]
        assert config.objective_weights == [1.0, -1.0]


class TestGAProgress:
    """GAProgressクラスのテスト"""

    def test_initialize(self):
        """初期化テスト"""
        progress = GAProgress(
            experiment_id="test_exp",
            current_generation=5,
            total_generations=100,
            best_fitness=0.85,
            average_fitness=0.72,
            execution_time=120.5,
            estimated_remaining_time=380.0,
            best_strategy_preview={"strategy": "test"}
        )

        assert progress.experiment_id == "test_exp"
        assert progress.current_generation == 5
        assert progress.total_generations == 100
        assert progress.best_fitness == 0.85
        assert progress.average_fitness == 0.72

    def test_progress_percentage_calculation(self):
        """進捗率計算テスト"""
        progress = GAProgress(
            experiment_id="test",
            current_generation=50,
            total_generations=200,
            best_fitness=0.8,
            average_fitness=0.6,
            execution_time=100,
            estimated_remaining_time=100
        )

        assert progress.progress_percentage == 25.0

    def test_progress_percentage_zero_total(self):
        """total_generations0時の進捗率テスト"""
        progress = GAProgress(
            experiment_id="test",
            current_generation=5,
            total_generations=0,
            best_fitness=0.8,
            average_fitness=0.6,
            execution_time=100,
            estimated_remaining_time=100
        )

        assert progress.progress_percentage == 0.0

    def test_to_dict(self):
        """辞書変換テスト"""
        progress = GAProgress(
            experiment_id="test_exp",
            current_generation=10,
            total_generations=50,
            best_fitness=0.9,
            average_fitness=0.7,
            execution_time=60,
            estimated_remaining_time=240,
            status="running"
        )

        result = progress.to_dict()

        assert isinstance(result, dict)
        assert result["experiment_id"] == "test_exp"
        assert result["current_generation"] == 10
        assert result["progress_percentage"] == 20.0
        assert result["status"] == "running"


@pytest.mark.parametrize("population_size", [0, -10])
def test_invalid_population_sizes_value_error(population_size):
    """ValueErrorを投げる無効なpopulation_sizeテスト"""
    with pytest.raises(ValueError, match="正の整数である必要があります"):
        GAConfig(population_size=population_size)


@pytest.mark.parametrize("generations", [0, -5])
def test_invalid_generations_value_error(generations):
    """ValueErrorを投げる無効なgenerationsテスト"""
    with pytest.raises(ValueError, match="正の整数である必要があります"):
        GAConfig(generations=generations)


@pytest.mark.parametrize("crossover_rate", [-0.1, 1.5])
def test_invalid_crossover_rates_value_error(crossover_rate):
    """ValueErrorを投げる無効なcrossover_rateテスト"""
    with pytest.raises(ValueError, match="0から1の範囲の実数である必要があります"):
        GAConfig(crossover_rate=crossover_rate)


@pytest.mark.parametrize("mutation_rate", [-0.5, 2.0])
def test_invalid_mutation_rates_value_error(mutation_rate):
    """ValueErrorを投げる無効なmutation_rateテスト"""
    with pytest.raises(ValueError, match="0から1の範囲の実数である必要があります"):
        GAConfig(mutation_rate=mutation_rate)
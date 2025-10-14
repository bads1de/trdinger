"""
設定クラス群のテストモジュール

GAConfig, TPSL設定クラスなどの設定クラスをテストする。
"""

import pytest
from unittest.mock import Mock, patch

from backend.app.services.auto_strategy.config.ga_runtime import GAConfig, GAProgress


class TestGAConfig:
    """GAConfigクラスのテスト"""

    @pytest.fixture
    def basic_ga_config(self):
        """基本的なGAConfigインスタンス"""
        return GAConfig(
            population_size=50,
            generations=20,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=5,
            max_indicators=3,
        )

    def test_initialization_with_valid_params(self, basic_ga_config):
        """有効なパラメータでの初期化テスト"""
        assert basic_ga_config.population_size == 50
        assert basic_ga_config.generations == 20
        assert basic_ga_config.crossover_rate == 0.8
        assert basic_ga_config.mutation_rate == 0.2
        assert basic_ga_config.elite_size == 5
        assert basic_ga_config.max_indicators == 3

    def test_post_init_validation_invalid_population_size(self):
        """無効なpopulation_sizeでのバリデーション"""
        with pytest.raises(
            ValueError, match="population_size は正の整数である必要があります"
        ):
            GAConfig(population_size=0)

    def test_post_init_validation_invalid_generations(self):
        """無効なgenerationsでのバリデーション"""
        with pytest.raises(
            ValueError, match="generations は正の整数である必要があります"
        ):
            GAConfig(generations=-1)

    def test_post_init_validation_invalid_crossover_rate(self):
        """無効なcrossover_rateでのバリデーション"""
        with pytest.raises(
            ValueError, match="crossover_rate は0から1の範囲の実数である必要があります"
        ):
            GAConfig(crossover_rate=1.5)

    def test_post_init_validation_invalid_mutation_rate(self):
        """無効なmutation_rateでのバリデーション"""
        with pytest.raises(
            ValueError, match="mutation_rate は0から1の範囲の実数である必要があります"
        ):
            GAConfig(mutation_rate=-0.1)

    def test_validate_valid_config(self, basic_ga_config):
        """有効な設定の検証テスト"""
        is_valid, errors = basic_ga_config.validate()
        assert is_valid is True
        assert errors == []

    def test_validate_invalid_population_size_too_large(self):
        """過大なpopulation_sizeの検証"""
        config = GAConfig(population_size=2000)
        is_valid, errors = config.validate()
        assert is_valid is False
        assert "個体数は1000以下である必要があります" in errors[0]

    def test_validate_invalid_generations_too_large(self):
        """過大なgenerationsの検証"""
        config = GAConfig(generations=1000)
        is_valid, errors = config.validate()
        assert is_valid is False
        assert "世代数は500以下である必要があります" in errors[0]

    def test_validate_invalid_fitness_weights_sum(self):
        """フィットネス重みの合計が1.0でない場合の検証"""
        config = GAConfig()
        config.fitness_weights = {"total_return": 0.5, "sharpe_ratio": 0.3}  # 合計0.8
        is_valid, errors = config.validate()
        assert is_valid is False
        assert "フィットネス重みの合計は1.0である必要があります" in errors[0]

    def test_validate_missing_required_metrics(self):
        """必要なメトリクスが不足する場合の検証"""
        config = GAConfig()
        config.fitness_weights = {"total_return": 1.0}  # sharpe_ratioなどが不足
        is_valid, errors = config.validate()
        assert is_valid is False
        assert "必要なメトリクスが不足しています" in errors[0]

    def test_validate_invalid_log_level(self):
        """無効なログレベルの検証"""
        config = GAConfig(log_level="INVALID")
        is_valid, errors = config.validate()
        assert is_valid is False
        assert "無効なログレベル" in errors[0]

    def test_validate_invalid_parallel_processes(self):
        """無効な並列プロセス数の検証"""
        config = GAConfig(parallel_processes=50)
        is_valid, errors = config.validate()
        assert is_valid is False
        assert "並列プロセス数は32以下である必要があります" in errors[0]

    def test_to_dict_conversion(self, basic_ga_config):
        """辞書変換テスト"""
        data = basic_ga_config.to_dict()
        assert isinstance(data, dict)
        assert data["population_size"] == 50
        assert data["generations"] == 20
        assert data["crossover_rate"] == 0.8
        assert data["mutation_rate"] == 0.2

    def test_from_dict_creation(self):
        """辞書からの作成テスト"""
        data = {
            "population_size": 100,
            "generations": 30,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
            "elite_size": 10,
            "max_indicators": 5,
        }
        config = GAConfig.from_dict(data)
        assert config.population_size == 100
        assert config.generations == 30
        assert config.crossover_rate == 0.9
        assert config.mutation_rate == 0.1

    def test_preprocess_ga_dict_defaults(self):
        """GA辞書前処理のデフォルト値設定テスト"""
        data = {}
        processed = GAConfig._preprocess_ga_dict(data)
        assert processed["population_size"] == 50  # デフォルト値
        assert processed["generations"] == 20
        assert processed["crossover_rate"] == 0.8
        assert processed["mutation_rate"] == 0.2

    def test_fitness_weights_validation(self):
        """フィットネス重みの検証テスト"""
        config = GAConfig()
        # 有効な重み
        config.fitness_weights = {
            "total_return": 0.4,
            "sharpe_ratio": 0.3,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }
        is_valid, errors = config.validate()
        assert is_valid is True

    def test_multi_objective_settings(self):
        """多目的最適化設定のテスト"""
        config = GAConfig(
            enable_multi_objective=True,
            objectives=["sharpe_ratio", "total_return"],
            objective_weights=[1.0, -0.5],
        )
        assert config.enable_multi_objective is True
        assert config.objectives == ["sharpe_ratio", "total_return"]
        assert config.objective_weights == [1.0, -0.5]

    def test_parameter_ranges_validation(self):
        """パラメータ範囲の検証テスト"""
        config = GAConfig()
        config.parameter_ranges = {
            "period": [5, 20],
            "multiplier": [1.0, 3.0],
        }
        is_valid, errors = config.validate()
        assert is_valid is True

        # 無効な範囲
        config.parameter_ranges = {"invalid": [10, 5]}  # min > max
        is_valid, errors = config.validate()
        assert is_valid is False
        assert "最小値は最大値より小さい必要があります" in errors[0]

    def test_error_handling_in_validation(self):
        """検証時のエラー処理テスト"""
        config = GAConfig()
        config.population_size = "invalid"  # 文字列を代入
        is_valid, errors = config.validate()
        assert is_valid is False
        assert "個体数は数値である必要があります" in errors[0]


class TestGAProgress:
    """GAProgressクラスのテスト"""

    @pytest.fixture
    def ga_progress(self):
        """GAProgressインスタンス"""
        return GAProgress(
            experiment_id="test_exp_001",
            current_generation=5,
            total_generations=20,
            best_fitness=1.5,
            average_fitness=1.2,
            execution_time=120.5,
            estimated_remaining_time=180.0,
            status="running",
        )

    def test_initialization(self, ga_progress):
        """初期化テスト"""
        assert ga_progress.experiment_id == "test_exp_001"
        assert ga_progress.current_generation == 5
        assert ga_progress.total_generations == 20
        assert ga_progress.best_fitness == 1.5
        assert ga_progress.average_fitness == 1.2
        assert ga_progress.execution_time == 120.5
        assert ga_progress.estimated_remaining_time == 180.0
        assert ga_progress.status == "running"

    def test_progress_percentage_calculation(self, ga_progress):
        """進捗率計算テスト"""
        # 5/20 = 0.25 * 100 = 25.0
        assert ga_progress.progress_percentage == 25.0

    def test_progress_percentage_zero_total_generations(self):
        """総世代数が0の場合の進捗率"""
        progress = GAProgress(
            experiment_id="test",
            current_generation=0,
            total_generations=0,
            best_fitness=0.0,
            average_fitness=0.0,
            execution_time=0.0,
            estimated_remaining_time=0.0,
        )
        assert progress.progress_percentage == 0.0

    def test_to_dict_conversion(self, ga_progress):
        """辞書変換テスト"""
        data = ga_progress.to_dict()
        assert isinstance(data, dict)
        assert data["experiment_id"] == "test_exp_001"
        assert data["current_generation"] == 5
        assert data["total_generations"] == 20
        assert data["best_fitness"] == 1.5
        assert data["average_fitness"] == 1.2
        assert data["progress_percentage"] == 25.0
        assert data["status"] == "running"

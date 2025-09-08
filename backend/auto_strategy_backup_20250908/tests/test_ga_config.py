import pytest
from unittest.mock import Mock, patch
from backend.app.services.auto_strategy.config.auto_strategy_config import GAConfig, GA_DEFAULT_CONFIG, DEFAULT_FITNESS_WEIGHTS
from backend.app.services.auto_strategy.config.constants import GA_DEFAULT_CONFIG, DEFAULT_FITNESS_WEIGHTS


class TestGAConfig:
    """GAConfigの単体テスト"""

    def test_initialization_with_defaults(self):
        """デフォルト初期化テスト"""
        config = GAConfig()

        # 基本GAパラメータ
        assert config.population_size == GA_DEFAULT_CONFIG["population_size"]
        assert config.generations == GA_DEFAULT_CONFIG["generations"]
        assert config.crossover_rate == GA_DEFAULT_CONFIG["crossover_rate"]
        assert config.mutation_rate == GA_DEFAULT_CONFIG["mutation_rate"]

        # フィットネス設定
        assert config.fitness_weights == DEFAULT_FITNESS_WEIGHTS
        assert config.primary_metric == "sharpe_ratio"

        # フィットネス共有設定
        assert config.enable_fitness_sharing == GA_DEFAULT_CONFIG["fitness_sharing"]["enable_fitness_sharing"]
        assert config.sharing_radius == GA_DEFAULT_CONFIG["fitness_sharing"]["sharing_radius"]
        assert config.sharing_alpha == GA_DEFAULT_CONFIG["fitness_sharing"]["sharing_alpha"]

    def test_validation_valid_config(self):
        """有効設定の検証テスト"""
        config = GAConfig()
        is_valid, errors = config.validate()

        assert is_valid == True
        assert errors == []

    def test_validation_invalid_population_size(self):
        """無効人口サイズの検証テスト"""
        config = GAConfig(population_size=-1)
        is_valid, errors = config.validate()

        assert is_valid == False
        assert any("個体数" in error for error in errors)

    def test_validation_invalid_crossover_rate(self):
        """無効交叉率の検証テスト"""
        config = GAConfig(crossover_rate=1.5)
        is_valid, errors = config.validate()

        assert is_valid == False
        assert any("交叉率" in error for error in errors)

    def test_validation_invalid_mutation_rate(self):
        """無効突然変異率の検証テスト"""
        config = GAConfig(mutation_rate=-0.1)
        is_valid, errors = config.validate()

        assert is_valid == False
        assert any("突然変異率" in error for error in errors)

    def test_validation_invalid_elite_size(self):
        """無効エリートサイズの検証テスト"""
        config = GAConfig(population_size=10, elite_size=15)
        is_valid, errors = config.validate()

        assert is_valid == False
        assert any("エリート" in error for error in errors)

    def test_validation_invalid_fitness_weights(self):
        """無効フィットネス重みの検証テスト"""
        config = GAConfig(fitness_weights={"total_return": 0.5})  # 合計が1でない
        is_valid, errors = config.validate()

        assert is_valid == False
        assert any("重み" in error for error in errors)

    def test_validation_missing_fitness_metric(self):
        """プライマリメトリクス欠落の検証テスト"""
        config = GAConfig(
            fitness_weights={"total_return": 1.0},
            primary_metric="sharpe_ratio"  # fitness_weightsにないメトリクス
        )
        is_valid, errors = config.validate()

        assert is_valid == False
        assert any("プライマリ" in error for error in errors)

    def test_validation_invalid_log_level(self):
        """無効ログレベルの検証テスト"""
        config = GAConfig(log_level="INVALID")
        is_valid, errors = config.validate()

        assert is_valid == False
        assert any("ログレベル" in error for error in errors)

    def test_validation_parallel_processes(self):
        """並列プロセス数の検証テスト"""
        # 負の値
        config = GAConfig(parallel_processes=-1)
        is_valid, errors = config.validate()
        assert is_valid == False

        # 過大な値
        config = GAConfig(parallel_processes=50)
        is_valid, errors = config.validate()
        assert is_valid == False

    def test_from_dict_basic(self):
        """基本的な辞書からの変換テスト"""
        data = {
            "population_size": 50,
            "generations": 20,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1
        }

        config = GAConfig.from_dict(data)

        assert config.population_size == 50
        assert config.generations == 20
        assert config.crossover_rate == 0.9
        assert config.mutation_rate == 0.1

    def test_from_dict_with_missing_values(self):
        """欠落した値がある辞書からの変換テスト"""
        data = {"population_size": 100}  # 他の値はデフォルト

        config = GAConfig.from_dict(data)

        assert config.population_size == 100
        assert config.generations == GA_DEFAULT_CONFIG["generations"]  # デフォルト値
        assert config.fitness_weights == DEFAULT_FITNESS_WEIGHTS

    def test_to_dict(self):
        """辞書変換テスト"""
        config = GAConfig(
            population_size=60,
            generations=25,
            crossover_rate=0.85,
            mutation_rate=0.15
        )

        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["population_size"] == 60
        assert result["generations"] == 25
        assert result["crossover_rate"] == 0.85
        assert result["mutation_rate"] == 0.15
        assert result["fitness_weights"] == DEFAULT_FITNESS_WEIGHTS

    def test_to_dict_includes_lists_and_complex_objects(self):
        """リストと複合オブジェクトを含む辞書変換テスト"""
        config = GAConfig()
        result = config.to_dict()

        # リストのパラメータ範囲が含まれる
        assert isinstance(result["parameter_ranges"], dict)
        # フィットネス重みが含まれる
        assert isinstance(result["fitness_weights"], dict)

    @patch('backend.app.services.auto_strategy.config.auto_strategy_config.IndicinService')
    def test_from_dict_with_allowed_indicators_auto_populate(self, mock_indicator_service):
        """許可指標の自動取得テスト"""
        # サービスモック
        mock_service_instance = Mock()
        mock_service_instance.get_supported_indicators.return_value = {
            "SMA": {},
            "EMA": {},
            "RSI": {}
        }
        mock_indicator_service.return_value = mock_service_instance

        data = {"allowed_indicators": []}  # 空リスト
        config = GAConfig.from_dict(data)

        # allowed_indicatorsが自動で取得されたはず
        assert len(config.allowed_indicators) > 0
        assert "SMA" in config.allowed_indicators

    def test_from_dict_fallback_when_service_unavailable(self):
        """サービス利用不可時のフォールバックテスト"""
        with patch('backend.app.services.auto_strategy.config.auto_strategy_config.IndicinService', side_effect=ImportError):
            data = {"allowed_indicators": []}
            config = GAConfig.from_dict(data)

            # フォールバックで空リスト
            assert config.allowed_indicators == []

    def test_json_serialization(self):
        """JSONシリアライゼーションテスト"""
        config = GAConfig(population_size=55, generations=30)
        json_str = config.to_json()

        assert isinstance(json_str, str)
        assert '"population_size": 55' in json_str
        assert '"generations": 30' in json_str

    def test_json_deserialization(self):
        """JSON復元テスト"""
        config1 = GAConfig(population_size=45, generations=40)
        json_str = config1.to_json()

        config2 = GAConfig.from_json(json_str)

        assert config2.population_size == 45
        assert config2.generations == 40

    @patch('backend.app.services.auto_strategy.config.auto_strategy_config.AutoStrategyConfig')
    def test_apply_auto_strategy_config(self, mock_auto_strategy_config_class):
        """AutoStrategyConfig適用テスト"""
        # AutoStrategyConfigモック
        mock_config = Mock()
        mock_config.ga.population_size = 80
        mock_config.ga.generations = 60
        mock_config.ga.crossover_rate = 0.95
        mock_config.ga.enable_fitness_sharing = True
        mock_config.ga.fitness_sharing = {
            "enable_fitness_sharing": True,
            "sharing_radius": 0.8,
            "sharing_alpha": 1.5
        }
        mock_config.ga.fitness_weights = {"total_return": 1.0}
        mock_config.ga.objectives = ["total_return"]
        mock_config.ga.ga_objectives = ["total_return"]
        mock_config.ga.ga_objective_weights = [1.0]

        mock_auto_strategy_config_class.return_value = mock_config

        config = GAConfig()
        config.apply_auto_strategy_config(mock_config)

        # 値が適用されているか確認
        assert config.population_size == 80
        assert config.generations == 60
        assert config.crossover_rate == 0.95
        assert config.enable_fitness_sharing == True
        assert config.sharing_radius == 0.8
        assert config.sharing_alpha == 1.5

    def test_from_auto_strategy_config(self):
        """AutoStrategyConfigからの作成テスト"""
        mock_config = Mock()
        # 同じモック設定
        mock_config.ga.population_size = 90
        mock_config.ga.generations = 70

        config = GAConfig.from_auto_strategy_config(mock_config)

        assert config.population_size == 90
        assert config.generations == 70

    def test_get_default_values(self):
        """デフォルト値取得テスト"""
        config = GAConfig()
        defaults = config.get_default_values()

        assert isinstance(defaults, dict)
        # fitness_weights が含まれている
        assert "fitness_weights" in defaults

    def test_create_default(self):
        """デフォルト設定作成テスト"""
        config = GAConfig.create_default()

        assert isinstance(config, GAConfig)
        assert config.population_size == GA_DEFAULT_CONFIG["population_size"]

    def test_create_fast(self):
        """高速実行設定作成テスト"""
        config = GAConfig.create_fast()

        assert isinstance(config, GAConfig)
        assert config.generations < GA_DEFAULT_CONFIG["generations"]  # 小さいはず

    def test_create_thorough(self):
        """詳細探索設定作成テスト"""
        config = GAConfig.create_thorough()

        assert isinstance(config, GAConfig)
        assert config.generations > GA_DEFAULT_CONFIG["generations"]  # 大きいはず

    def test_create_multi_objective(self):
        """多目的最適化設定作成テスト"""
        objectives = ["total_return", "max_drawdown"]
        weights = [1.0, -1.0]

        config = GAConfig.create_multi_objective(objectives, weights)

        assert isinstance(config, GAConfig)
        assert config.enable_multi_objective == True
        assert config.objectives == objectives
        assert config.objective_weights == weights

    def test_default_multi_objective_config(self):
        """デフォルト多目的最適化設定作成テスト"""
        config = GAConfig.create_multi_objective()

        assert config.objectives == ["total_return", "max_drawdown"]
        assert config.objective_weights == [1.0, -1.0]
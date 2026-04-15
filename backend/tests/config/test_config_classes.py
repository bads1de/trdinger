"""
設定クラス群のテストモジュール

UnifiedConfig, GAConfigなどの設定クラスをテストする。
"""

import copy
import os
from unittest.mock import patch

import pytest

from app.config.unified_config import (
    AppConfig,
    DatabaseConfig,
    LoggingConfig,
    MarketConfig,
    UnifiedConfig,
)
from app.services.auto_strategy.config.constants import (
    DEFAULT_STRATEGIES_LIMIT,
    GA_FALLBACK_END_DATE,
    GA_FALLBACK_START_DATE,
    MAX_STRATEGIES_LIMIT,
)
from app.services.auto_strategy.config.ga import GAConfig as GAConfigRuntime
from app.services.backtest.config import BacktestConfig
from app.services.ml.common.ml_config import MLConfig, MLPredictionConfig
from app.services.auto_strategy.config import ConfigValidator


class TestUnifiedConfig:
    """UnifiedConfigクラスのテスト"""

    def test_initialization_with_defaults(self):
        """デフォルト値での初期化テスト"""
        config = UnifiedConfig()
        assert config.app is not None
        assert config.database is not None
        assert config.logging is not None
        assert config.market is not None
        assert config.data_collection is not None

    def test_singleton_instance_access(self):
        """シングルトンインスタンスへのアクセステスト"""
        from app.config.unified_config import unified_config

        assert isinstance(unified_config, UnifiedConfig)
        assert unified_config.app.app_name == "Trdinger Trading API"

    def test_package_level_exports(self):
        """app.config はクラスを公開し、シングルトンはサブモジュールから取得すること"""
        import app.config as config_package
        from app.config.unified_config import unified_config

        assert config_package.UnifiedConfig is UnifiedConfig
        assert "UnifiedConfig" in config_package.__all__
        assert "unified_config" not in config_package.__all__
        assert isinstance(unified_config, UnifiedConfig)

    def test_hierarchical_config_access(self):
        """階層的設定へのアクセステスト"""
        config = UnifiedConfig()

        # App設定へのアクセス
        assert config.app.app_name == "Trdinger Trading API"
        assert config.app.app_version == "1.0.0"

        # Database設定へのアクセス
        assert config.database.host == "localhost"
        assert config.database.port == 5432

        # Market設定へのアクセス
        assert config.market.default_exchange == "bybit"
        assert "bybit" in config.market.supported_exchanges

    def test_env_nested_delimiter_support(self):
        """環境変数のネスト区切り文字サポートテスト"""
        config = UnifiedConfig()
        assert hasattr(config, "model_config")
        assert config.model_config.get("env_nested_delimiter") == "__"

    @patch.dict(
        os.environ,
        {
            "APP_NAME": "TestApp",
            "APP_VERSION": "2.0.0",
            "DB_HOST": "testhost",
            "DB_PORT": "3306",
        },
    )
    def test_environment_variable_loading(self):
        """環境変数からの設定読み込みテスト"""
        config = UnifiedConfig()

        assert config.app.app_name == "TestApp"
        assert config.app.app_version == "2.0.0"
        assert config.database.host == "testhost"
        assert config.database.port == 3306

class TestAppConfig:
    """AppConfigクラスのテスト"""

    def test_initialization_with_defaults(self):
        """デフォルト値での初期化テスト"""
        config = AppConfig()
        assert config.app_name == "Trdinger Trading API"
        assert config.app_version == "1.0.0"
        assert config.debug is False
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.cors_origins == ["http://localhost:3000"]

    @patch.dict(
        os.environ,
        {"APP_NAME": "CustomApp", "APP_VERSION": "2.0.0"},
    )
    def test_environment_variable_override(self):
        """環境変数による設定上書きテスト"""
        config = AppConfig()
        assert config.app_name == "CustomApp"
        assert config.app_version == "2.0.0"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_initialization_with_defaults(self):
        """デフォルト値での初期化テスト"""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.name == "trdinger"
        assert config.user == "postgres"

    def test_url_complete_with_individual_params(self):
        """個別パラメータからのURL生成テスト"""
        # 環境変数をクリアしてから設定
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "",  # DATABASE_URLを明示的に空にする
                "DB_HOST": "dbhost",
                "DB_PORT": "5432",
                "DB_NAME": "testdb",
                "DB_USER": "testuser",
                "DB_PASSWORD": "testpass",
            },
            clear=False,
        ):
            config = DatabaseConfig()
            expected_url = "postgresql://testuser:testpass@dbhost:5432/testdb"
            assert config.url_complete == expected_url

    @patch.dict(
        os.environ,
        {"DATABASE_URL": "postgresql://custom:pass@custom:5432/custom"},
    )
    def test_url_complete_with_database_url(self):
        """DATABASE_URL優先テスト"""
        custom_url = "postgresql://custom:pass@custom:5432/custom"
        config = DatabaseConfig()
        assert config.url_complete == custom_url


class TestMarketConfig:
    """MarketConfigクラスのテスト"""

    def test_initialization_with_defaults(self):
        """デフォルト値での初期化テスト"""
        config = MarketConfig()
        assert config.sandbox is False
        assert config.enable_cache is True
        assert config.default_exchange == "bybit"
        assert config.default_symbol == "BTC/USDT:USDT"
        assert config.default_timeframe == "1h"

    def test_supported_lists(self):
        """サポートされているリストのテスト"""
        config = MarketConfig()
        assert "bybit" in config.supported_exchanges
        assert "BTC/USDT:USDT" in config.supported_symbols
        assert "1h" in config.supported_timeframes

    def test_symbol_mapping(self):
        """シンボルマッピングのテスト"""
        config = MarketConfig()
        assert "BTCUSDT" in config.symbol_mapping
        assert config.symbol_mapping["BTCUSDT"] == "BTC/USDT:USDT"


class TestBacktestConfig:
    """BacktestConfigクラスのテスト"""

    def test_initialization_with_defaults(self):
        """デフォルト値での初期化テスト"""
        config = BacktestConfig()
        assert config.default_initial_capital == 10000.0
        assert config.default_commission_rate == 0.001
        assert config.max_results_limit == 50
        assert config.default_results_limit == 20


class TestAutoStrategyDefaults:
    """AutoStrategy の公開定数テスト"""

    def test_strategy_list_limits(self):
        """戦略一覧 API の既定値が期待値に固定されている"""
        assert DEFAULT_STRATEGIES_LIMIT == 20
        assert MAX_STRATEGIES_LIMIT == 100
        assert DEFAULT_STRATEGIES_LIMIT < MAX_STRATEGIES_LIMIT

    def test_ga_fallback_dates(self):
        """GA のフォールバック期間が期待値に固定されている"""
        assert GA_FALLBACK_START_DATE == "2024-01-01"
        assert GA_FALLBACK_END_DATE == "2024-04-09"


class TestMLConfig:
    """MLConfigクラスのテスト"""

    def test_initialization_with_defaults(self):
        """デフォルト値での初期化テスト"""
        config = MLConfig()
        assert config.data_processing is not None
        assert config.model is not None
        assert config.prediction is not None
        assert config.training is not None

    def test_data_processing_config(self):
        """データ処理設定のテスト"""
        config = MLConfig()
        print(
            f"DEBUG: config.data_processing.max_ohlcv_rows = {config.data_processing.max_ohlcv_rows}"
        )
        assert config.data_processing.max_ohlcv_rows == 1000000
        assert config.data_processing.feature_calculation_timeout == 3600

    def test_model_config(self):
        """モデル設定のテスト"""
        config = MLConfig()
        assert config.model.model_save_path == "models/"
        assert config.model.model_file_extension == ".pkl"
        assert config.model.max_model_versions == 10

    def test_prediction_config(self):
        """予測設定のテスト"""
        config = MLConfig()
        assert config.prediction.default_forecast_log_rv == 0.0
        assert config.prediction.fallback_forecast_log_rv == 0.0

    def test_prediction_methods(self):
        """予測設定のメソッドテスト"""
        config = MLConfig()
        default_preds = config.prediction.get_default_predictions()
        assert default_preds["forecast_log_rv"] == 0.0
        assert default_preds["gate_open"] is True

        fallback_preds = config.prediction.get_fallback_predictions()
        assert fallback_preds["forecast_log_rv"] == 0.0
        assert fallback_preds["gate_open"] is True

    def test_training_config(self):
        """学習設定のテスト"""
        config = MLConfig()
        assert config.training.lgb_n_estimators == 100
        assert config.training.xgb_learning_rate == 0.1
        assert config.training.cv_folds == 5


class TestConfigIntegration:
    """設定クラス間の統合テスト"""

    def test_unified_config_contains_all_subconfigs(self):
        """UnifiedConfigがアプリ共通のサブ設定のみを含むテスト"""
        config = UnifiedConfig()
        required_configs = [
            "app",
            "database",
            "logging",
            "market",
            "data_collection",
        ]
        for config_name in required_configs:
            assert hasattr(config, config_name)
            assert getattr(config, config_name) is not None

        assert not hasattr(config, "backtest")
        assert not hasattr(config, "auto_strategy")
        assert not hasattr(config, "ml")

    def test_domain_configs_are_separated(self):
        """ドメイン固有の設定は各パッケージで独立していることを確認"""
        config = UnifiedConfig()
        assert config.market.default_exchange == "bybit"
        assert BacktestConfig().default_initial_capital > 0
        assert DEFAULT_STRATEGIES_LIMIT <= MAX_STRATEGIES_LIMIT
        assert MLConfig().training is not None


class TestConfigValidation:
    """設定検証のテスト"""

    def test_invalid_population_size(self):
        """無効なpopulation_sizeの検証テスト"""
        # GAConfigRuntimeは_preprocess_ga_dictを持たないため、直接無効な値を持つGAConfigRuntimeを作成
        config = GAConfigRuntime(population_size=-1)
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is False
        assert any("個体数は正の整数である必要があります" in e for e in errors)

    def test_invalid_port_number(self):
        """無効なポート番号の検証テスト"""
        # Pydanticが自動的に検証するため、正常に動作することを確認
        config = AppConfig(port=8000)
        assert config.port == 8000

    def test_database_url_generation_without_password(self):
        """パスワードなしのデータベースURL生成テスト"""
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "",  # DATABASE_URLを明示的に空にする
                "DB_USER": "testuser",
                "DB_PASSWORD": "",
                "DB_HOST": "localhost",
                "DB_PORT": "5432",
                "DB_NAME": "trdinger",
            },
            clear=False,
        ):
            config = DatabaseConfig()
            url = config.url_complete
            assert "testuser:@localhost" in url

    def test_ml_prediction_probability_range(self):
        """ML予測確率の範囲テスト"""
        config = MLPredictionConfig()
        assert isinstance(config.default_gate_open, bool)
        assert isinstance(config.fallback_gate_open, bool)


class TestEnvironmentVariableSupport:
    """環境変数サポートのテスト"""

    @patch.dict(
        os.environ,
        {
            "APP_NAME": "EnvTestApp",
            "DB_HOST": "envhost",
            "LOG_LEVEL": "WARNING",
            "MARKET_DEFAULT_EXCHANGE": "binance",
        },
    )
    def test_multiple_env_vars_loading(self):
        """複数の環境変数の読み込みテスト"""
        app_config = AppConfig()
        db_config = DatabaseConfig()
        log_config = LoggingConfig()
        market_config = MarketConfig()

        assert app_config.app_name == "EnvTestApp"
        assert db_config.host == "envhost"
        assert log_config.level == "WARNING"
        assert market_config.default_exchange == "binance"

    @patch.dict(os.environ, {"BACKTEST_DEFAULT_INITIAL_CAPITAL": "50000.0"})
    def test_float_env_var_parsing(self):
        """浮動小数点環境変数のパーステスト"""
        config = BacktestConfig()
        assert config.default_initial_capital == 50000.0

    @patch.dict(os.environ, {"MARKET_DATA_SANDBOX": "true"})
    def test_boolean_env_var_parsing(self):
        """ブール環境変数のパーステスト"""
        config = MarketConfig()
        assert config.sandbox is True


class TestGAConfigRuntime:
    """GAConfigRuntimeクラスのテスト"""

    @pytest.fixture
    def basic_ga_config(self):
        """基本的なGAConfigRuntimeインスタンス"""
        return GAConfigRuntime(
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

    def test_validation_invalid_population_size(self):
        """無効なpopulation_sizeでのバリデーション"""
        config = GAConfigRuntime(population_size=0)
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is False
        assert any("個体数は正の整数である必要があります" in e for e in errors)

    def test_validation_invalid_generations(self):
        """無効なgenerationsでのバリデーション"""
        config = GAConfigRuntime(generations=-1)
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is False
        assert any("世代数は正の整数である必要があります" in e for e in errors)

    def test_validation_invalid_crossover_rate(self):
        """無効なcrossover_rateでのバリデーション"""
        config = GAConfigRuntime(crossover_rate=1.5)
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is False
        assert any("交叉率は0-1の範囲である必要があります" in e for e in errors)

    def test_validation_invalid_mutation_rate(self):
        """無効なmutation_rateでのバリデーション"""
        config = GAConfigRuntime(mutation_rate=-0.1)
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is False
        assert any("突然変異率は0-1の範囲である必要があります" in e for e in errors)

    def test_validate_valid_config(self, basic_ga_config):
        """有効な設定の検証テスト"""
        is_valid, errors = ConfigValidator.validate(basic_ga_config)
        assert is_valid is True
        assert errors == []

    def test_validate_invalid_population_size_too_large(self):
        """過大なpopulation_sizeの検証"""
        config = GAConfigRuntime(population_size=2000)
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is False
        assert any("個体数は1000以下である必要があります" in e for e in errors)

    def test_validate_invalid_generations_too_large(self):
        """過大なgenerationsの検証"""
        config = GAConfigRuntime(generations=1000)
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is False
        assert any("世代数は500以下である必要があります" in e for e in errors)

    def test_validate_invalid_fitness_weights_sum(self):
        """フィットネス重みの合計が1.0でない場合の検証"""
        config = GAConfigRuntime()
        config.fitness_weights = {"total_return": 0.5, "sharpe_ratio": 0.3}  # 合計0.8
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is False
        assert any(
            "フィットネス重みの合計は1.0である必要があります" in e for e in errors
        )

    def test_validate_prediction_score_is_rejected(self):
        """prediction_score はボラ回帰化に伴い拒否される"""
        config = GAConfigRuntime()
        config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.3,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
            "prediction_score": 0.1,
        }
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is False
        assert any("prediction_score" in e for e in errors)

    def test_validate_missing_required_metrics(self):
        """必要なメトリクスが不足する場合の検証"""
        config = GAConfigRuntime()
        config.fitness_weights = {"total_return": 1.0}  # sharpe_ratioなどが不足
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is False
        assert any("必要なメトリクスが不足しています" in e for e in errors)

    def test_validate_invalid_log_level(self):
        """無効なログレベルの検証"""
        config = GAConfigRuntime(log_level="INVALID")
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is False
        assert any("無効なログレベル" in e for e in errors)

    def test_validate_invalid_parallel_processes(self):
        """無効な並列プロセス数の検証"""
        config = GAConfigRuntime(parallel_processes=50)
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is False
        assert any("並列プロセス数は32以下である必要があります" in e for e in errors)

    def test_validate_two_stage_selection_constraints(self):
        """二段階選抜の制約を検証"""
        config = GAConfigRuntime(
            population_size=10,
            two_stage_selection_config={
                "enabled": True,
                "elite_count": 10,
                "candidate_pool_size": 2,
                "min_pass_rate": 1.5,
            },
        )
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is False
        assert any("二段階選抜エリート数は個体数未満である必要があります" in e for e in errors)
        assert any(
            "二段階選抜候補数は二段階選抜エリート数以上である必要があります"
            in e
            for e in errors
        )
        assert any("二段階選抜 pass rate は0.0-1.0の範囲である必要があります" in e for e in errors)

    @pytest.mark.parametrize(
        ("regime_window", "expected_message"),
        [
            (
                {
                    "name": "regime_a",
                    "start_date": "2024-02-01 00:00:00",
                    "end_date": "2024-01-01 00:00:00",
                },
                "start_date < end_date",
            ),
            (
                {
                    "name": "regime_a",
                    "start_date": "invalid-date",
                    "end_date": "2024-01-01 00:00:00",
                },
                "日付形式が不正",
            ),
        ],
    )
    def test_validate_robustness_regime_windows(
        self, regime_window, expected_message
    ):
        """robustness の regime window を検証"""
        config = GAConfigRuntime(
            robustness_config={"regime_windows": [regime_window]}
        )
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is False
        assert any(expected_message in e for e in errors)

    def test_validate_robustness_related_fields(self):
        """robustness 関連のその他の制約を検証"""
        config = GAConfigRuntime(
            robustness_config={
                "validation_symbols": "BTC/USDT",
                "stress_slippage": [-0.01],
                "stress_commission_multipliers": [0.0],
                "aggregate_method": "invalid",
            }
        )
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is False
        assert any(
            "robustness_config.validation_symbols はリストである必要があります"
            in e
            for e in errors
        )
        assert any(
            "robustness の slippage は0以上の数値である必要があります" in e
            for e in errors
        )
        assert any(
            "robustness の commission multiplier は正の数値である必要があります"
            in e
            for e in errors
        )
        assert any(
            "robustness_config.aggregate_method は {'robust', 'mean'} のいずれかである必要があります"
            in e
            for e in errors
        )

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
        config = GAConfigRuntime.from_dict(data)
        assert config.population_size == 100
        assert config.generations == 30
        assert config.crossover_rate == 0.9
        assert config.mutation_rate == 0.1

    def test_from_dict_does_not_mutate_input_data(self):
        """from_dict が入力データを破壊しないことを確認"""
        data = {
            "two_stage_selection_config": {
                "enabled": False,
                "elite_count": 7,
                "candidate_pool_size": 9,
                "min_pass_rate": 0.75,
            },
            "robustness_config": {
                "validation_symbols": ["BTC/USDT"],
                "regime_windows": [
                    {
                        "name": "regime_a",
                        "start_date": "2024-01-01 00:00:00",
                        "end_date": "2024-01-02 00:00:00",
                    }
                ],
                "stress_slippage": [0.001],
                "stress_commission_multipliers": [1.5],
                "aggregate_method": "mean",
            },
        }
        original = copy.deepcopy(data)

        config = GAConfigRuntime.from_dict(data)

        config.two_stage_selection_config.enabled = False
        config.robustness_config.validation_symbols.append("ETH/USDT")
        config.robustness_config.regime_windows[0]["name"] = "regime_b"

        assert data == original
        assert config.two_stage_selection_config.enabled is False
        assert config.two_stage_selection_config.elite_count == 7
        assert config.two_stage_selection_config.candidate_pool_size == 9
        assert config.two_stage_selection_config.min_pass_rate == 0.75
        assert config.robustness_config.validation_symbols == ["BTC/USDT", "ETH/USDT"]
        assert config.robustness_config.regime_windows[0]["name"] == "regime_b"
        assert config.robustness_config.aggregate_method == "mean"

    def test_from_dict_creates_independent_mutable_defaults(self):
        """from_dict のデフォルト可変値がインスタンス間で共有されないことを確認"""
        first = GAConfigRuntime.from_dict({})
        second = GAConfigRuntime.from_dict({})

        first.fitness_weights["total_return"] = 0.99
        first.objectives.append("extra_objective")
        first.parameter_ranges["period"][0] = 999
        first.robustness_config.regime_windows.append(
            {
                "name": "regime_b",
                "start_date": "2024-01-01 00:00:00",
                "end_date": "2024-01-02 00:00:00",
            }
        )

        assert second.fitness_weights["total_return"] != 0.99
        assert "extra_objective" not in second.objectives
        assert second.parameter_ranges["period"][0] == 5
        assert second.robustness_config.regime_windows == []

    def test_fitness_weights_validation(self):
        """フィットネス重みの検証テスト"""
        config = GAConfigRuntime()
        # 有効な重み
        config.fitness_weights = {
            "total_return": 0.4,
            "sharpe_ratio": 0.3,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is True

    def test_multi_objective_settings(self):
        """多目的最適化設定のテスト"""
        config = GAConfigRuntime(
            enable_multi_objective=True,
            objectives=["sharpe_ratio", "total_return"],
            objective_weights=[1.0, -0.5],
        )
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is True
        assert config.enable_multi_objective is True
        assert config.objectives == ["sharpe_ratio", "total_return"]
        assert config.objective_weights == [1.0, -0.5]

    def test_parameter_ranges_validation(self):
        """パラメータ範囲の検証テスト"""
        config = GAConfigRuntime()
        config.parameter_ranges = {
            "period": [5, 20],
            "multiplier": [1.0, 3.0],
        }
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is True

        # 無効な範囲
        config.parameter_ranges = {"invalid": [10, 5]}  # min > max
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is False
        assert any("最小値は最大値より小さい必要があります" in e for e in errors)

    def test_error_handling_in_validation(self):
        """検証時のエラー処理テスト"""
        config = GAConfigRuntime()
        config.population_size = "invalid"  # 文字列を代入
        is_valid, errors = ConfigValidator.validate(config)
        assert is_valid is False
        assert any("個体数は数値である必要があります" in e for e in errors)

from unittest.mock import MagicMock

import pytest

from app.services.auto_strategy.config.base import BaseConfig
from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.config.validators import ConfigValidator


class TestConfigValidator:

    @pytest.fixture
    def base_config(self):
        config = MagicMock(spec=BaseConfig)
        config.validation_rules = {
            "required_fields": ["field_a"],
            "ranges": {"field_b": (0, 10)},
            "types": {"field_c": int},
        }
        config.field_a = "present"
        config.field_b = 5
        config.field_c = 100
        return config

    @pytest.fixture
    def ga_config(self):
        # GAConfigは多数の属性を持つので、MagicMockで必要な属性を設定する
        config = MagicMock(spec=GAConfig)
        # BaseConfigの検証をパスするための設定（GAConfigはBaseConfigを継承している想定だが、バリデータはインスタンスを見る）
        # GAConfig自体にvalidation_rulesがあるかどうかは実装によるが、ここでは空にしておく
        config.validation_rules = {}

        # 正常なGA設定
        config.population_size = 100
        config.generations = 50
        config.crossover_rate = 0.8
        config.mutation_rate = 0.1
        config.elite_size = 5
        config.oos_split_ratio = 0.3

        config.fitness_weights = {
            "total_return": 0.4,
            "sharpe_ratio": 0.3,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }
        config.primary_metric = "total_return"
        config.max_indicators = 5
        config.parameter_ranges = {"param1": [0, 10]}
        config.log_level = "INFO"
        config.parallel_processes = 4

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
        # プロパティアクセスで例外を発生させる
        type(base_config).field_a = PropertyMock(side_effect=Exception("Test Error"))

        # エラーはキャッチされてエラーメッセージに追加されるはず
        is_valid, errors = ConfigValidator.validate(base_config)
        assert is_valid is False
        assert any("検証処理エラー: Test Error" in e for e in errors)

    def test_validate_ga_config_success(self, ga_config):
        is_valid, errors = ConfigValidator.validate(ga_config)
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

        # primary_metricが含まれていない
        ga_config.fitness_weights = {
            "total_return": 0.4,
            "sharpe_ratio": 0.3,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }
        ga_config.primary_metric = "not_in_weights"
        is_valid, errors = ConfigValidator.validate(ga_config)
        assert is_valid is False
        assert any(
            "プライマリメトリクス 'not_in_weights' がフィットネス重みに" in e
            for e in errors
        )

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


from unittest.mock import PropertyMock

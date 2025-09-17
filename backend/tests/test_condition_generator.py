import pytest
from unittest.mock import patch
from backend.app.services.auto_strategy.generators.condition_generator import ConditionGenerator
from backend.app.services.auto_strategy.models.strategy_models import IndicatorGene


class TestConditionGenerator:
    def setup_method(self):
        self.generator = ConditionGenerator()

    def test_ema_long_condition_right_operand_is_close(self):
        """EMAのロング条件生成でright_operandが"close"であることをテスト"""
        ema_indicator = IndicatorGene(type="EMA", parameters={"period": 20}, enabled=True)

        long_conditions = self.generator._create_trend_long_conditions(ema_indicator)

        assert len(long_conditions) == 1
        condition = long_conditions[0]
        assert condition.left_operand == "EMA"
        assert condition.operator == ">"
        assert condition.right_operand == "close"

    def test_ema_short_condition_right_operand_is_close(self):
        """EMAのショート条件生成でright_operandが"close"であることをテスト"""
        ema_indicator = IndicatorGene(type="EMA", parameters={"period": 20}, enabled=True)

        short_conditions = self.generator._create_trend_short_conditions(ema_indicator)

        assert len(short_conditions) == 1
        condition = short_conditions[0]
        assert condition.left_operand == "EMA"
        assert condition.operator == "<"
        assert condition.right_operand == "close"

    def test_sma_long_condition_uses_threshold_fallback(self):
        """SMAのロング条件生成でthresholdがない場合fallbackを使うことをテスト"""
        sma_indicator = IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)

        long_conditions = self.generator._create_trend_long_conditions(sma_indicator)

        assert len(long_conditions) == 1
        condition = long_conditions[0]
        assert condition.left_operand == "SMA"
        assert condition.operator == ">"
        assert condition.right_operand == 0

    def test_generate_balanced_conditions_success(self):
        """正常な指標リストで条件生成が成功することをテスト"""
        indicators = [IndicatorGene(type="EMA", parameters={"period": 20}, enabled=True)]
        long_conditions, short_conditions, exit_conditions = self.generator.generate_balanced_conditions(indicators)

        assert isinstance(long_conditions, list)
        assert isinstance(short_conditions, list)
        assert isinstance(exit_conditions, list)

    def test_generate_balanced_conditions_raises_exception_on_error(self):
        """YAML設定読み込みでエラーが発生した場合に例外を投げることをテスト"""
        indicators = [IndicatorGene(type="EMA", parameters={"period": 20}, enabled=True)]

        with patch('backend.app.services.auto_strategy.generators.condition_generator.YamlIndicatorUtils.load_yaml_config_for_indicators') as mock_load:
            mock_load.side_effect = Exception("YAML読み込みエラー")

            # コンストラクタで失敗するので新しいインスタンスを作成
            with pytest.raises(Exception):
                ConditionGenerator()
import pytest
from unittest.mock import MagicMock, patch

from app.services.auto_strategy.generators.condition_generator import ConditionGenerator
from app.services.auto_strategy.models.strategy_models import IndicatorGene, Condition, ConditionGroup
from app.services.auto_strategy.constants import StrategyType, IndicatorType

# --- ヘルパー関数 ---
def create_indicator_gene(type: str, period: int = 14) -> IndicatorGene:
    return IndicatorGene(type=type, parameters={'period': period}, enabled=True)

# --- テストクラス ---
class TestConditionGenerator:

    @pytest.fixture
    def condition_generator(self):
        """ConditionGeneratorのインスタンスと依存サービスのモック"""
        with patch('app.services.auto_strategy.generators.condition_generator.YamlIndicatorUtils') as MockYamlIndicatorUtils, \
             patch('app.services.auto_strategy.generators.condition_generator.indicator_registry') as MockIndicatorRegistry:
            
            # YamlIndicatorUtilsのモック設定
            mock_yaml_config = {
                "indicators": {
                    "RSI": {"type": "momentum", "scale_type": "oscillator_0_100", "thresholds": {"normal": {"long_lt": 30, "short_gt": 70}}},
                    "SMA": {"type": "trend", "scale_type": "price_absolute", "thresholds": {"normal": {"long_gt": 0, "short_lt": 0}}},
                    "MACD": {"type": "momentum", "scale_type": "momentum_zero_centered", "thresholds": {"normal": {"long_gt": 0, "short_lt": 0}}},
                    "ML_UP_PROB": {"type": "ml", "scale_type": "oscillator_0_1", "thresholds": {"normal": {"long_gt": 0.6}}},
                    "ML_DOWN_PROB": {"type": "ml", "scale_type": "oscillator_0_1", "thresholds": {"normal": {"short_lt": 0.4}}},
                }
            }
            MockYamlIndicatorUtils.load_yaml_config_for_indicators.return_value = mock_yaml_config
            MockYamlIndicatorUtils.get_indicator_config_from_yaml.side_effect = lambda yaml_cfg, ind_name: yaml_cfg["indicators"].get(ind_name)
            MockYamlIndicatorUtils.get_threshold_from_yaml.side_effect = lambda yaml_cfg, ind_cfg, side, ctx: \
                ind_cfg["thresholds"]["normal"].get(f"{side}_gt") or ind_cfg["thresholds"]["normal"].get(f"{side}_lt")

            # indicator_registryのモック設定
            # get_indicator_configが呼ばれた際に、モックされたYAML設定から対応するtypeを返す
            def mock_get_indicator_config(ind_name):
                config_data = mock_yaml_config["indicators"].get(ind_name)
                if config_data:
                    # MagicMockのcategory属性にtypeを設定
                    mock_config = MagicMock()
                    mock_config.category = config_data.get("type")
                    mock_config.type = config_data.get("type") # type属性も追加
                    return mock_config
                return None

            MockIndicatorRegistry.get_indicator_config.side_effect = mock_get_indicator_config

            generator = ConditionGenerator(enable_smart_generation=True)
            yield generator

    def test_generate_balanced_conditions_fallback(self, condition_generator):
        """スマート生成が無効または指標がない場合のフォールバック条件生成"""
        generator_fallback = ConditionGenerator(enable_smart_generation=False)
        longs, shorts, exits = generator_fallback.generate_balanced_conditions([])

        assert len(longs) == 1
        assert isinstance(longs[0], Condition)
        assert longs[0].left_operand == "close"
        assert longs[0].operator == ">"
        assert longs[0].right_operand == "open"

        assert len(shorts) == 1
        assert isinstance(shorts[0], Condition)
        assert shorts[0].left_operand == "close"
        assert shorts[0].operator == "<"
        assert shorts[0].right_operand == "open"

        assert len(exits) == 0

    def test_generate_balanced_conditions_different_indicators(self, condition_generator):
        """異なる指標の組み合わせ戦略が正しく条件を生成するか"""
        indicators = [
            create_indicator_gene('SMA', 20),
            create_indicator_gene('RSI', 14),
            create_indicator_gene('MACD', 12)
        ]
        # _select_strategy_typeがDIFFERENT_INDICATORSを返すようにモック
        with patch.object(condition_generator, '_select_strategy_type', return_value=StrategyType.DIFFERENT_INDICATORS):
            longs, shorts, exits = condition_generator.generate_balanced_conditions(indicators)

            assert len(longs) >= 1
            assert len(shorts) >= 1
            assert len(exits) == 0

            # 生成された条件が妥当か（例: SMA > 0, RSI < 30 など）
            # 具体的な値はモックのthresholdsに依存
            # 各カテゴリから少なくとも1つは条件が生成されていることを確認
            generated_operands = {c.left_operand for c in longs}
            assert 'SMA' in generated_operands or 'RSI' in generated_operands or 'MACD' in generated_operands

    def test_generate_balanced_conditions_complex_conditions(self, condition_generator):
        """複合条件戦略が正しく条件を生成するか"""
        indicators = [
            create_indicator_gene('SMA', 20),
            create_indicator_gene('RSI', 14)
        ]
        # _select_strategy_typeがCOMPLEX_CONDITIONSを返すようにモック
        with patch.object(condition_generator, '_select_strategy_type', return_value=StrategyType.COMPLEX_CONDITIONS):
            longs, shorts, exits = condition_generator.generate_balanced_conditions(indicators)

            assert len(longs) >= 1
            assert len(shorts) >= 1
            assert len(exits) == 0

            # 複合条件のロジックは内部で処理されるため、ここでは生成された条件の存在を確認
            assert any(isinstance(c, Condition) for c in longs)

    def test_generate_balanced_conditions_ml_indicators(self, condition_generator):
        """ML指標を含む戦略が正しく条件を生成するか"""
        indicators = [
            create_indicator_gene('ML_UP_PROB'),
            create_indicator_gene('ML_DOWN_PROB')
        ]
        # _select_strategy_typeがINDICATOR_CHARACTERISTICSを返すようにモック
        with patch.object(condition_generator, '_select_strategy_type', return_value=StrategyType.INDICATOR_CHARACTERISTICS):
            longs, shorts, exits = condition_generator.generate_balanced_conditions(indicators)

            assert len(longs) >= 1
            assert len(shorts) >= 1
            assert len(exits) == 0

            assert any(c.left_operand == 'ML_UP_PROB' for c in longs)
            assert any(c.left_operand == 'ML_DOWN_PROB' for c in shorts)

    def test_set_context(self, condition_generator):
        """コンテキスト設定が正しく行われるか"""
        condition_generator.set_context(timeframe="4h", symbol="ETH/USDT", threshold_profile="aggressive")
        assert condition_generator.context["timeframe"] == "4h"
        assert condition_generator.context["symbol"] == "ETH/USDT"
        assert condition_generator.context["threshold_profile"] == "aggressive"

        # 無効なプロファイルはnormalにフォールバック
        condition_generator.set_context(threshold_profile="invalid")
        assert condition_generator.context["threshold_profile"] == "normal"

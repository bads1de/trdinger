import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from backtesting import Strategy

from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
from app.services.auto_strategy.models.strategy_models import StrategyGene, IndicatorGene, Condition, TPSLGene

# --- ヘルパー関数 ---

def create_test_strategy_gene(indicator_count=2, condition_count=1) -> StrategyGene:
    """テスト用の有効なStrategyGeneオブジェクトを生成する"""
    indicators = [IndicatorGene(type='SMA', parameters={'period': 10+i}) for i in range(indicator_count)]
    
    operands = [f"SMA_{10+i}" for i in range(indicator_count)]
    
    entry_conditions = [Condition(left_operand=operands[i], operator='>', right_operand=operands[i+1]) for i in range(condition_count)]
    
    tpsl_gene = TPSLGene(enabled=True, stop_loss_pct=0.05, take_profit_pct=0.1)

    return StrategyGene(
        id="test-gene-123",
        indicators=indicators,
        entry_conditions=entry_conditions,
        long_entry_conditions=entry_conditions,
        short_entry_conditions=[],
        exit_conditions=[],
        risk_management={},
        tpsl_gene=tpsl_gene,
        position_sizing_gene=None
    )

# --- テストクラス ---

class TestStrategyFactory:

    @pytest.fixture
    def strategy_factory(self):
        """StrategyFactoryと依存サービスのモック"""
        with patch('app.services.auto_strategy.generators.strategy_factory.ConditionEvaluator') as MockConditionEvaluator, \
             patch('app.services.auto_strategy.generators.strategy_factory.IndicatorCalculator') as MockIndicatorCalculator, \
             patch('app.services.auto_strategy.generators.strategy_factory.TPSLService'), \
             patch('app.services.auto_strategy.generators.strategy_factory.PositionSizingService') as MockPositionSizingService:
            
            factory = StrategyFactory()
            factory.condition_evaluator = MockConditionEvaluator()
            factory.indicator_calculator = MockIndicatorCalculator()
            factory.position_sizing_service = MockPositionSizingService()
            yield factory

    def test_create_strategy_class_success(self, strategy_factory):
        """有効な遺伝子からStrategyクラスが正しく生成されるか"""
        gene = create_test_strategy_gene()
        StrategyClass = strategy_factory.create_strategy_class(gene)
        assert issubclass(StrategyClass, Strategy)
        assert StrategyClass.__name__ == f"GS_{gene.id.split('-')[0]}"

    def test_create_strategy_class_invalid_gene(self, strategy_factory):
        """無効な遺伝子でValueErrorが発生するか"""
        gene = create_test_strategy_gene()
        with patch.object(gene, 'validate', return_value=(False, ["Invalid parameter"])):
            with pytest.raises(ValueError, match="Invalid strategy gene"):
                strategy_factory.create_strategy_class(gene)

    def test_generated_strategy_init(self, strategy_factory):
        """生成されたクラスのinitメソッドがインジケーターを初期化するか"""
        gene = create_test_strategy_gene(indicator_count=3)
        StrategyClass = strategy_factory.create_strategy_class(gene)
        
        strategy_instance = StrategyClass()
        strategy_instance.init()

        calculator = strategy_factory.indicator_calculator
        assert calculator.init_indicator.call_count == 3
        calculator.init_indicator.assert_any_call(gene.indicators[0], strategy_instance)

    def test_generated_strategy_next_buy_signal(self, strategy_factory):
        """nextメソッドが買いシグナルでbuyを呼び出すか"""
        gene = create_test_strategy_gene()
        StrategyClass = strategy_factory.create_strategy_class(gene)

        strategy_factory.condition_evaluator.evaluate_conditions.side_effect = [True, False] # long=True, short=False
        strategy_factory.position_sizing_service.calculate_position_size.return_value.position_size = 0.1

        mock_data = MagicMock()
        mock_data.Close = [100]

        with patch.object(StrategyClass, 'buy', new_callable=MagicMock) as mock_buy, \
             patch.object(StrategyClass, 'sell', new_callable=MagicMock) as mock_sell, \
             patch.object(StrategyClass, 'position', new_callable=PropertyMock(return_value=None)), \
             patch.object(StrategyClass, 'data', new_callable=PropertyMock(return_value=mock_data)):

            strategy_instance = StrategyClass()
            strategy_instance.next()

            assert strategy_factory.condition_evaluator.evaluate_conditions.call_count == 2
            mock_buy.assert_called_once()
            mock_sell.assert_not_called()

    def test_generated_strategy_next_no_signal(self, strategy_factory):
        """シグナルがない場合に取引しないことを確認"""
        gene = create_test_strategy_gene()
        StrategyClass = strategy_factory.create_strategy_class(gene)

        strategy_factory.condition_evaluator.evaluate_conditions.return_value = False

        with patch.object(StrategyClass, 'buy', new_callable=MagicMock) as mock_buy, \
             patch.object(StrategyClass, 'sell', new_callable=MagicMock) as mock_sell, \
             patch.object(StrategyClass, 'position', new_callable=PropertyMock(return_value=None)):

            strategy_instance = StrategyClass()
            strategy_instance.next()

            mock_buy.assert_not_called()
            mock_sell.assert_not_called()

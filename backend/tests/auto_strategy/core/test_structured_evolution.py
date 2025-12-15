
import pytest
from unittest.mock import MagicMock
from backend.app.services.auto_strategy.core.condition_evolver import ConditionEvolver
from backend.app.services.auto_strategy.models.strategy_models import Condition, ConditionGroup
from backend.app.services.auto_strategy.core.condition_evolver import YamlIndicatorUtils

class TestStructuredEvolution:
    @pytest.fixture
    def evolver(self):
        # 依存関係のモック
        backtest_service = MagicMock()
        yaml_utils = MagicMock()
        yaml_utils.get_available_indicators.return_value = ["SMA", "RSI"]
        yaml_utils.get_indicator_info.return_value = {"scale_type": "oscillator_0_100"}
        
        evolver = ConditionEvolver(backtest_service, yaml_utils)
        return evolver

    def test_crossover_condition_groups(self, evolver):
        """ConditionGroup同士の構造維持交叉をテスト"""
        # 親1: (SMA > 0) AND (RSI > 50)
        parent1 = ConditionGroup(
            operator="AND",
            conditions=[
                Condition("SMA", ">", 0),
                Condition("RSI", ">", 50)
            ]
        )
        
        # 親2: (EMA < 0) AND (MACD < 20)
        parent2 = ConditionGroup(
            operator="AND",
            conditions=[
                Condition("EMA", "<", 0),
                Condition("MACD", "<", 20)
            ]
        )
        
        child1, child2 = evolver.crossover(parent1, parent2)
        
        # 子はConditionGroupであるべき
        assert isinstance(child1, ConditionGroup)
        assert isinstance(child2, ConditionGroup)
        
        # 構造（要素数）が維持されているか
        assert len(child1.conditions) == 2
        assert len(child2.conditions) == 2
        
        # 再帰的な交叉が行われているか（要素が混ざっているか、あるいは値が変わっているか）
        # 完全な検証は確率に依存するが、型チェックは可能
        assert isinstance(child1.conditions[0], Condition)

    def test_mutate_condition_group(self, evolver):
        """ConditionGroupの突然変異をテスト"""
        group = ConditionGroup(
            operator="AND",
            conditions=[
                Condition("SMA", ">", 0),
                Condition("RSI", ">", 50)
            ]
        )
        
        mutated = evolver.mutate(group)
        
        assert isinstance(mutated, ConditionGroup)
        # 内容が何かしら変わっている可能性がある（確率依存なのでアサートは緩く）
        # 少なくともエラーにならずにオブジェクトが返ってくることを確認

    def test_crossover_mixed_types(self, evolver):
        """異なる型の交叉（Condition vs Group）"""
        c1 = Condition("SMA", ">", 0)
        g1 = ConditionGroup(operator="AND", conditions=[Condition("RSI", ">", 50)])
        
        # 構造が違う場合は交叉せず、そのまま返す（あるいは親のコピー）
        child1, child2 = evolver.crossover(c1, g1)
        
        # 簡単な実装では、型が違う場合は交叉しない
        assert isinstance(child1, Condition)
        assert isinstance(child2, ConditionGroup)

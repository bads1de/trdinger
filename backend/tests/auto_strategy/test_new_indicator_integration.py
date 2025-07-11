"""
新しいインジケータカテゴリのオートストラテジー統合テスト

101個の新しいテクニカルインジケータがオートストラテジー機能で
正しく動作することを確認するテストスイート
"""

import pytest
import random
from unittest.mock import patch

from app.core.services.auto_strategy.generators.smart_condition_generator import (
    SmartConditionGenerator,
    IndicatorType,
    INDICATOR_CHARACTERISTICS
)
from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.models.gene_strategy import IndicatorGene
from app.core.services.auto_strategy.models.ga_config import GAConfig


class TestNewIndicatorIntegration:
    """新しいインジケータカテゴリの統合テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.smart_generator = SmartConditionGenerator(enable_smart_generation=True)
        
        # テスト用のGA設定
        self.config = GAConfig(
            population_size=10,
            generations=5,
            max_indicators=3,
            min_indicators=1,
            max_conditions=2,
            min_conditions=1
        )
        self.random_generator = RandomGeneGenerator(self.config)

    def test_new_indicator_types_in_enum(self):
        """新しいインジケータタイプがEnumに追加されていることを確認"""
        expected_types = [
            IndicatorType.MOMENTUM,
            IndicatorType.TREND,
            IndicatorType.VOLATILITY,
            IndicatorType.CYCLE,
            IndicatorType.STATISTICS,
            IndicatorType.MATH_TRANSFORM,
            IndicatorType.MATH_OPERATORS,
            IndicatorType.PATTERN_RECOGNITION
        ]
        
        for indicator_type in expected_types:
            assert indicator_type in IndicatorType

    def test_new_indicators_in_characteristics(self):
        """新しいインジケータがINDICATOR_CHARACTERISTICSに定義されていることを確認"""
        # サイクル系
        cycle_indicators = ["HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR", "HT_SINE", "HT_TRENDMODE"]
        for indicator in cycle_indicators:
            assert indicator in INDICATOR_CHARACTERISTICS
            assert INDICATOR_CHARACTERISTICS[indicator]["type"] == IndicatorType.CYCLE

        # 統計系
        stats_indicators = ["BETA", "CORREL", "LINEARREG", "LINEARREG_ANGLE", "STDDEV", "VAR"]
        for indicator in stats_indicators:
            assert indicator in INDICATOR_CHARACTERISTICS
            assert INDICATOR_CHARACTERISTICS[indicator]["type"] == IndicatorType.STATISTICS

        # 数学変換系
        math_transform_indicators = ["ACOS", "ASIN", "ATAN", "COS", "SIN", "TAN", "CEIL", "FLOOR", "SQRT"]
        for indicator in math_transform_indicators:
            assert indicator in INDICATOR_CHARACTERISTICS
            assert INDICATOR_CHARACTERISTICS[indicator]["type"] == IndicatorType.MATH_TRANSFORM

        # 数学演算子系
        math_operators = ["ADD", "SUB", "MULT", "DIV", "MAX", "MIN"]
        for indicator in math_operators:
            assert indicator in INDICATOR_CHARACTERISTICS
            assert INDICATOR_CHARACTERISTICS[indicator]["type"] == IndicatorType.MATH_OPERATORS

        # パターン認識系
        pattern_indicators = ["CDL_DOJI", "CDL_HAMMER", "CDL_HANGING_MAN", "CDL_SHOOTING_STAR"]
        for indicator in pattern_indicators:
            assert indicator in INDICATOR_CHARACTERISTICS
            assert INDICATOR_CHARACTERISTICS[indicator]["type"] == IndicatorType.PATTERN_RECOGNITION

    def test_cycle_indicator_condition_generation(self):
        """サイクル系インジケータの条件生成テスト"""
        # HT_DCPHASEのテスト
        indicator = IndicatorGene(type="HT_DCPHASE", parameters={"period": 14}, enabled=True)
        
        long_conditions = self.smart_generator._create_cycle_long_conditions(indicator)
        short_conditions = self.smart_generator._create_cycle_short_conditions(indicator)
        
        assert len(long_conditions) > 0
        assert len(short_conditions) > 0
        assert long_conditions[0].left_operand == "HT_DCPHASE_14"
        assert short_conditions[0].left_operand == "HT_DCPHASE_14"

    def test_statistics_indicator_condition_generation(self):
        """統計系インジケータの条件生成テスト"""
        # CORRELのテスト
        indicator = IndicatorGene(type="CORREL", parameters={"period": 30}, enabled=True)
        
        long_conditions = self.smart_generator._create_statistics_long_conditions(indicator)
        short_conditions = self.smart_generator._create_statistics_short_conditions(indicator)
        
        assert len(long_conditions) > 0
        assert len(short_conditions) > 0
        assert long_conditions[0].left_operand == "CORREL_30"
        assert short_conditions[0].left_operand == "CORREL_30"

    def test_pattern_recognition_condition_generation(self):
        """パターン認識系インジケータの条件生成テスト"""
        # CDL_HAMMERのテスト
        indicator = IndicatorGene(type="CDL_HAMMER", parameters={}, enabled=True)
        
        long_conditions = self.smart_generator._create_pattern_long_conditions(indicator)
        short_conditions = self.smart_generator._create_pattern_short_conditions(indicator)
        
        assert len(long_conditions) > 0
        # CDL_HAMMERは強気パターンなのでショート条件は生成されない
        assert len(short_conditions) == 0
        assert long_conditions[0].left_operand == "CDL_HAMMER"

    def test_balanced_conditions_with_new_indicators(self):
        """新しいインジケータを含むバランス条件生成テスト"""
        indicators = [
            IndicatorGene(type="HT_DCPHASE", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="CORREL", parameters={"period": 30}, enabled=True),
            IndicatorGene(type="CDL_DOJI", parameters={}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
        ]
        
        long_conditions, short_conditions, exit_conditions = (
            self.smart_generator.generate_balanced_conditions(indicators)
        )
        
        assert len(long_conditions) > 0
        assert len(short_conditions) > 0
        # 新しいインジケータが条件に含まれていることを確認
        all_operands = [cond.left_operand for cond in long_conditions + short_conditions]
        assert any("HT_DCPHASE" in operand or "CORREL" in operand or "CDL_DOJI" in operand 
                  for operand in all_operands)

    def test_random_gene_generator_includes_new_indicators(self):
        """RandomGeneGeneratorが新しいインジケータを含むことを確認"""
        # 新しいインジケータが利用可能リストに含まれていることを確認
        available_indicators = self.random_generator.available_indicators
        
        # 各カテゴリから少なくとも1つのインジケータが含まれていることを確認
        cycle_indicators = ["HT_DCPERIOD", "HT_DCPHASE", "HT_SINE"]
        stats_indicators = ["BETA", "CORREL", "STDDEV"]
        pattern_indicators = ["CDL_DOJI", "CDL_HAMMER"]
        
        assert any(indicator in available_indicators for indicator in cycle_indicators)
        assert any(indicator in available_indicators for indicator in stats_indicators)
        assert any(indicator in available_indicators for indicator in pattern_indicators)

    def test_strategy_gene_generation_with_new_indicators(self):
        """新しいインジケータを使った戦略遺伝子生成テスト"""
        # 複数回生成して新しいインジケータが選ばれることを確認
        new_indicator_found = False
        new_indicators = [
            "HT_DCPERIOD", "HT_DCPHASE", "HT_SINE", "BETA", "CORREL", "STDDEV",
            "ACOS", "ASIN", "COS", "SIN", "ADD", "SUB", "CDL_DOJI", "CDL_HAMMER"
        ]
        
        for _ in range(10):  # 10回試行
            strategy_gene = self.random_generator.generate_random_gene()
            
            # 生成された指標に新しいインジケータが含まれているかチェック
            for indicator in strategy_gene.indicators:
                if indicator.type in new_indicators:
                    new_indicator_found = True
                    break
            
            if new_indicator_found:
                break
        
        assert new_indicator_found, "新しいインジケータが戦略生成で選ばれませんでした"

    def test_error_handling_with_new_indicators(self):
        """新しいインジケータでのエラーハンドリングテスト"""
        # 存在しないインジケータタイプでのテスト
        invalid_indicator = IndicatorGene(type="INVALID_INDICATOR", parameters={}, enabled=True)
        
        # エラーが発生してもフォールバック条件が生成されることを確認
        long_conditions, short_conditions, exit_conditions = (
            self.smart_generator.generate_balanced_conditions([invalid_indicator])
        )
        
        assert len(long_conditions) > 0
        assert len(short_conditions) > 0

    def test_complex_strategy_with_new_indicators(self):
        """複合戦略での新しいインジケータ活用テスト"""
        indicators = [
            IndicatorGene(type="HT_TRENDMODE", parameters={}, enabled=True),
            IndicatorGene(type="LINEARREG_SLOPE", parameters={"period": 20}, enabled=True)
        ]
        
        long_conditions, short_conditions, exit_conditions = (
            self.smart_generator._generate_complex_conditions_strategy(indicators)
        )
        
        assert len(long_conditions) > 0
        assert len(short_conditions) > 0
        
        # 新しいインジケータが使用されていることを確認
        all_operands = [cond.left_operand for cond in long_conditions + short_conditions]
        assert any("HT_TRENDMODE" in operand or "LINEARREG_SLOPE" in operand 
                  for operand in all_operands)

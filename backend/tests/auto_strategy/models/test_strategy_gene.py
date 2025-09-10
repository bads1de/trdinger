"""
strategy_gene.py のユニットテスト
"""
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import replace

from backend.app.services.auto_strategy.models.strategy_gene import StrategyGene
from backend.app.services.auto_strategy.models.condition import Condition, ConditionGroup
from backend.app.services.auto_strategy.models.indicator_gene import IndicatorGene
from backend.app.services.auto_strategy.models.tpsl_gene import TPSLGene
from backend.app.services.auto_strategy.models.position_sizing_gene import PositionSizingGene
from backend.app.services.auto_strategy.models.enums import PositionSizingMethod, TPSLMethod


class TestStrategyGene:
    """StrategyGene クラステスト"""

    def test_strategy_gene_initialization(self):
        """StrategyGene の初期化テスト"""
        # 基本的な初期化
        gene = StrategyGene(
            id="test_strategy",
            indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
            entry_conditions=[Condition(left_operand="close", operator=">", right_operand="open")],
            exit_conditions=[Condition(left_operand="close", operator="<", right_operand="open")],
            long_entry_conditions=[Condition(left_operand="close", operator=">", right_operand="open")],
            short_entry_conditions=[Condition(left_operand="close", operator="<", right_operand="open")],
            risk_management={"max_drawdown": 0.1},
            tpsl_gene=TPSLGene(stop_loss_pct=0.05, take_profit_pct=0.1),
            position_sizing_gene=PositionSizingGene(method=PositionSizingMethod.FIXED_RATIO),
            metadata={"source": "test"}
        )

        assert gene.id == "test_strategy"
        assert len(gene.indicators) == 1
        assert len(gene.entry_conditions) == 1
        assert len(gene.exit_conditions) == 1
        assert len(gene.long_entry_conditions) == 1
        assert len(gene.short_entry_conditions) == 1
        assert gene.risk_management == {"max_drawdown": 0.1}
        assert gene.tpsl_gene is not None
        assert gene.position_sizing_gene is not None
        assert gene.metadata == {"source": "test"}

    def test_strategy_gene_default_values(self):
        """デフォルト値のテスト"""
        gene = StrategyGene()

        assert gene.id == ""
        assert gene.indicators == []
        assert gene.entry_conditions == []
        assert gene.exit_conditions == []
        assert gene.long_entry_conditions == []
        assert gene.short_entry_conditions == []
        assert gene.risk_management == {}
        assert gene.tpsl_gene is None
        assert gene.position_sizing_gene is None
        assert gene.metadata == {}

    def test_strategy_gene_max_indicators_constant(self):
        """MAX_INDICATORS 定数のテスト"""
        assert hasattr(StrategyGene, 'MAX_INDICATORS')
        assert StrategyGene.MAX_INDICATORS == 5

    def test_get_effective_long_conditions_with_long_conditions(self):
        """get_effective_long_conditions: long_entry_conditions が存在する場合"""
        gene = StrategyGene()
        gene.long_entry_conditions = [
            Condition(left_operand="close", operator=">", right_operand="sma_20"),
            Condition(left_operand="volume", operator=">", right_operand="1000000")
        ]

        result = gene.get_effective_long_conditions()
        assert len(result) == 2
        assert result[0].operator == ">"
        assert result[1].operator == ">"

    def test_get_effective_long_conditions_with_entry_conditions(self):
        """get_effective_long_conditions: entry_conditions のみ存在する場合"""
        gene = StrategyGene()
        gene.entry_conditions = [
            Condition(left_operand="close", operator="<", right_operand="sma_20")
        ]
        gene.long_entry_conditions = []

        result = gene.get_effective_long_conditions()
        assert len(result) == 1
        assert result[0].operator == "<"

    def test_get_effective_long_conditions_empty(self):
        """get_effective_long_conditions: 条件がない場合"""
        gene = StrategyGene()

        result = gene.get_effective_long_conditions()
        assert result == []

    def test_get_effective_short_conditions_with_short_conditions(self):
        """get_effective_short_conditions: short_entry_conditions が存在する場合"""
        gene = StrategyGene()
        gene.short_entry_conditions = [
            Condition(left_operand="close", operator="<", right_operand="sma_20")
        ]

        result = gene.get_effective_short_conditions()
        assert len(result) == 1
        assert result[0].operator == "<"

    def test_get_effective_short_conditions_with_entry_conditions(self):
        """get_effective_short_conditions: entry_conditions のみ存在する場合"""
        gene = StrategyGene()
        gene.entry_conditions = [
            Condition(left_operand="close", operator=">", right_operand="sma_20")
        ]
        gene.long_entry_conditions = []  # long条件が存在してshort選択を強制
        gene.short_entry_conditions = []

        result = gene.get_effective_short_conditions()
        assert len(result) == 1
        assert result[0].operator == ">"

    def test_get_effective_short_conditions_empty(self):
        """get_effective_short_conditions: 条件がない場合"""
        gene = StrategyGene()

        result = gene.get_effective_short_conditions()
        assert result == []

    def test_has_long_short_separation_true(self):
        """has_long_short_separation: 条件が分離されている場合"""
        gene_with_separation = StrategyGene()
        gene_with_separation.long_entry_conditions = [
            Condition(left_operand="close", operator=">", right_operand="sma_20")
        ]

        assert gene_with_separation.has_long_short_separation() is True

    def test_has_long_short_separation_false(self):
        """has_long_short_separation: 条件が分離されていない場合"""
        gene_without_separation = StrategyGene()
        gene_without_separation.entry_conditions = [
            Condition(left_operand="close", operator=">", right_operand="sma_20")
        ]

        assert gene_without_separation.has_long_short_separation() is False

    def test_method_property_with_position_sizing_gene(self):
        """method プロパティ: position_sizing_gene が存在する場合"""
        gene = StrategyGene()
        gene.position_sizing_gene = PositionSizingGene(method=PositionSizingMethod.VOLATILITY_BASED)

        assert gene.method == PositionSizingMethod.VOLATILITY_BASED

    def test_method_property_without_position_sizing_gene(self):
        """method プロパティ: position_sizing_gene が存在しない場合"""
        gene = StrategyGene()

        assert gene.method == PositionSizingMethod.FIXED_RATIO

    def test_validate_method_with_gene_validator(self):
        """validate() メソッド: GeneValidator 使用テスト"""
        gene = StrategyGene()
        gene.indicators = [IndicatorGene(type="SMA", parameters={"period": 20})]
        gene.entry_conditions = [Condition(left_operand="close", operator=">", right_operand="open")]

        # GeneValidator をモック
        with patch('backend.app.services.auto_strategy.models.strategy_gene.GeneValidator') as mock_validator:
            mock_validator_instance = MagicMock()
            mock_validator_instance.validate_strategy_gene.return_value = (True, [])
            mock_validator.return_value = mock_validator_instance

            result = gene.validate()

            # GeneValidator が正しく呼び出されたか確認
            mock_validator.assert_called_once()
            mock_validator_instance.validate_strategy_gene.assert_called_once_with(gene)

    def test_validate_method_with_validation_errors(self):
        """validate() メソッド: 検証エラーテスト"""
        gene = StrategyGene()
        # 空の条件でエラーを発生させる

        # GeneValidator をモックしてエラーを返却
        with patch('backend.app.services.auto_strategy.models.strategy_gene.GeneValidator') as mock_validator:
            mock_validator_instance = MagicMock()
            mock_validator_instance.validate_strategy_gene.return_value = (False, ["エントリー条件が設定されていません"])
            mock_validator.return_value = mock_validator_instance

            is_valid, errors = gene.validate()

            assert is_valid is False
            assert "エントリー条件が設定されていません" in errors

    def test_validate_method_with_max_indicators_violation(self):
        """validate() メソッド: MAX_INDICATORS 違反テスト"""
        gene = StrategyGene()
        # MAX_INDICATORS (5) を超えた指標を追加
        gene.indicators = [IndicatorGene(type=f"INDICATOR_{i}") for i in range(8)]

        with patch('backend.app.services.auto_strategy.models.strategy_gene.GeneValidator') as mock_validator:
            mock_validator_instance = MagicMock()
            mock_validator_instance.validate_strategy_gene.return_value = (True, [])
            mock_validator.return_value = mock_validator_instance

            is_valid, errors = gene.validate()

            # 検証が呼ばれていることを確認
            mock_validator_instance.validate_strategy_gene.assert_called_once_with(gene)


class TestStrategyGeneWithConditions:
    """StrategyGene と条件関連のテスト"""

    def test_strategy_gene_with_condition_groups(self):
        """ConditionGroup を使った StrategyGene テスト"""
        gene = StrategyGene()

        # ConditionGroup の作成
        group = ConditionGroup()
        group.conditions = [
            Condition(left_operand="close", operator=">", right_operand="open"),
            Condition(left_operand="volume", operator=">", right_operand="1000000")
        ]

        gene.long_entry_conditions = [group, Condition(left_operand="rsi", operator="<", right_operand="30")]

        result = gene.get_effective_long_conditions()
        assert len(result) == 2
        assert isinstance(result[0], ConditionGroup)
        assert isinstance(result[1], Condition)

    def test_strategy_gene_with_mixed_conditions(self):
        """混合条件タイプのテスト"""
        gene = StrategyGene()

        simple_condition = Condition(left_operand="close", operator=">", right_operand="sma_20")

        # ConditionGroupを含む
        gene.entry_conditions = [simple_condition]

        result = gene.get_effective_long_conditions()
        assert len(result) == 1
        assert result[0] == simple_condition


class TestStrategyGeneEdgeCases:
    """エッジケーステスト"""

    def test_strategy_gene_with_none_values(self):
        """None 値のテスト"""
        gene = StrategyGene()
        gene.id = None
        gene.risk_management = None
        gene.tpsl_gene = None

        # None が設定されていることを確認
        assert gene.id is None
        assert gene.risk_management is None
        assert gene.tpsl_gene is None

    def test_strategy_gene_with_empty_lists(self):
        """空リストのテスト"""
        gene = StrategyGene()
        gene.indicators = []
        gene.entry_conditions = []
        gene.exit_conditions = []
        gene.long_entry_conditions = []
        gene.short_entry_conditions = []

        assert len(gene.indicators) == 0
        assert len(gene.entry_conditions) == 0
        assert len(gene.exit_conditions) == 0

    def test_strategy_gene_with_invalid_id(self):
        """無効なIDのテスト"""
        gene = StrategyGene()
        gene.id = "invalid<id>test"  # 特殊文字を含む

        assert gene.id == "invalid<id>test"

    def test_strategy_gene_with_large_metadata(self):
        """大きなメタデータを持つテスト"""
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(100)}
        gene = StrategyGene(metadata=large_metadata)

        assert len(gene.metadata) == 100
        assert gene.metadata["key_0"] == "value_0"
        assert gene.metadata["key_99"] == "value_99"

    def test_strategy_gene_method_property_edge_cases(self):
        """method プロパティのエッジケース"""
        gene = StrategyGene()

        # position_sizing_gene なし
        assert gene.method == PositionSizingMethod.FIXED_RATIO

        # position_sizing_gene の method が None
        gene.position_sizing_gene = PositionSizingGene()
        assert gene.method == PositionSizingMethod.VOLATILITY_BASED  # デフォルト値

    def test_strategy_gene_has_long_short_separation_edge_cases(self):
        """has_long_short_separation のエッジケース"""
        gene = StrategyGene()

        # 空のリストの場合
        gene.long_entry_conditions = []
        assert gene.has_long_short_separation() is False

        # None の場合
        gene.long_entry_conditions = None
        assert gene.has_long_short_separation() is False


class TestStrategyGeneComplexScenarios:
    """複雑なシナリオテスト"""

    def test_strategy_gene_complete_workflow(self):
        """完全なワークフローシナリオテスト"""
        # 完全な戦略遺伝子の作成
        gene = StrategyGene()
        gene.id = "complete_strategy"
        gene.indicators = [
            IndicatorGene(type="SMA", parameters={"period": 20}),
            IndicatorGene(type="RSI", parameters={"period": 14}),
            IndicatorGene(type="MACD", parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9})
        ]
        gene.entry_conditions = [
            Condition(left_operand="close", operator=">", right_operand="sma_20")
        ]
        gene.exit_conditions = [
            Condition(left_operand="close", operator="<", right_operand="open")
        ]
        gene.tpsl_gene = TPSLGene(stop_loss_pct=0.05, take_profit_pct=0.1)
        gene.position_sizing_gene = PositionSizingGene(risk_per_trade=0.02)

        # 基本検証
        assert gene.id == "complete_strategy"
        assert len(gene.indicators) == 3
        assert len(gene.entry_conditions) == 1
        assert len(gene.exit_conditions) == 1
        assert gene.tpsl_gene is not None
        assert gene.position_sizing_gene is not None

        # メソッド呼び出しのテスト
        long_conditions = gene.get_effective_long_conditions()
        assert len(long_conditions) == 1

        short_conditions = gene.get_effective_short_conditions()
        assert len(short_conditions) == 1

        assert gene.has_long_short_separation() is False
        assert gene.method == PositionSizingMethod.FIXED_RATIO  # position_sizing_geneのデフォルトのため

    def test_strategy_gene_condition_mutation_scenarios(self):
        """条件操作のシナリオテスト"""
        gene = StrategyGene()
        gene.entry_conditions = [Condition(left_operand="close", operator=">", right_operand="open")]

        # 条件の動的変更
        new_condition = Condition(left_operand="rsi", operator="<", right_operand="30")
        gene.entry_conditions.append(new_condition)

        assert len(gene.entry_conditions) == 2
        assert gene.entry_conditions[0].left_operand == "close"
        assert gene.entry_conditions[1].left_operand == "rsi"
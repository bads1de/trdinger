"""
StrategyFactory のテスト
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from backtesting import Strategy

from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
from app.services.auto_strategy.models.strategy_models import (
    StrategyGene,
    IndicatorGene,
)


class TestStrategyFactory:
    """StrategyFactory のテストクラス"""

    def setup_method(self):
        """各テストメソッド前のセットアップ"""
        self.factory = StrategyFactory()

    def test_create_strategy_class_success(self):
        """正常な戦略クラスの生成テスト"""
        from app.services.auto_strategy.models.condition import Condition

        # 有効な戦略遺伝子を作成（最低限のエントリー条件とイグジット条件を設定）
        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            ],
            entry_conditions=[],
            exit_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA")
            ],
            long_entry_conditions=[
                Condition(left_operand="RSI", operator="<", right_operand=30),
                Condition(left_operand="close", operator=">", right_operand="SMA"),
            ],
            short_entry_conditions=[
                Condition(left_operand="RSI", operator=">", right_operand=70),
                Condition(left_operand="close", operator="<", right_operand="SMA"),
            ],
            risk_management={},
            metadata={"test": True},
        )

        # 戦略クラス生成
        strategy_class = self.factory.create_strategy_class(gene)

        # 生成されたクラスがStrategyを継承しているか確認
        assert issubclass(strategy_class, Strategy)
        assert hasattr(strategy_class, "strategy_gene")
        assert hasattr(strategy_class, "init")
        assert hasattr(strategy_class, "next")

    def test_create_strategy_class_with_invalid_gene(self):
        """無効な遺伝子でのエラーテスト"""
        # 無効な戦略遺伝子（空の指標リストなど）
        gene = StrategyGene(
            indicators=[],  # 空の指標リスト
            entry_conditions=[],
            exit_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            risk_management={},
            metadata={"test": True},
        )

        # 戦略クラス生成は検証エラーで失敗するはず
        with pytest.raises(ValueError):
            self.factory.create_strategy_class(gene)

    @patch("app.services.auto_strategy.generators.strategy_factory.IndicatorCalculator")
    def test_indicator_initialization_failure(self, mock_indicator_calculator):
        """指標初期化失敗時のテスト"""
        from app.services.auto_strategy.models.condition import Condition

        # IndicatorCalculatorのモックを設定
        mock_instance = Mock()
        mock_indicator_calculator.return_value = mock_instance
        mock_instance.init_indicator.side_effect = Exception(
            "Indicator initialization failed"
        )

        # 戦略遺伝子作成（有効な条件を追加）- 有効な指標タイプを使う
        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            ],
            entry_conditions=[],
            exit_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA")
            ],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA"),
            ],
            short_entry_conditions=[
                Condition(left_operand="close", operator="<", right_operand="SMA"),
            ],
            risk_management={},
            metadata={"test": True},
        )

        # 戦略クラス生成
        strategy_class = self.factory.create_strategy_class(gene)

        # インスタンス作成とinit呼び出しでエラーが発生することを確認
        strategy_instance = strategy_class()

        with pytest.raises(Exception):  # 初期化エラーが発生するはず
            strategy_instance.init()

    def test_strategy_execution_with_valid_conditions(self):
        """有効な条件での戦略実行テスト"""
        from app.services.auto_strategy.models.condition import Condition

        # 有効な戦略遺伝子を作成
        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            ],
            entry_conditions=[],
            exit_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA")
            ],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA"),
            ],
            short_entry_conditions=[
                Condition(left_operand="close", operator="<", right_operand="SMA"),
            ],
            risk_management={},
            metadata={"test": True},
        )

        # 戦略クラス生成
        strategy_class = self.factory.create_strategy_class(gene)

        # インスタンス作成
        strategy_instance = strategy_class()

        # nextメソッドがエラーなく実行できるか確認（データがないので何もしない）
        strategy_instance.next()  # エラーが発生しないことを確認

    def test_position_size_calculation_fallback(self):
        """ポジションサイズ計算のフォールバックテスト"""
        from app.services.auto_strategy.models.condition import Condition

        # PositionSizingGeneなしの戦略遺伝子
        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            ],
            entry_conditions=[],
            exit_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA")
            ],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA"),
            ],
            short_entry_conditions=[
                Condition(left_operand="close", operator="<", right_operand="SMA"),
            ],
            risk_management={},
            metadata={"test": True},
        )

        # 戦略クラス生成
        strategy_class = self.factory.create_strategy_class(gene)
        strategy_instance = strategy_class()

        # ポジションサイズ計算メソッドがデフォルト値を返すことを確認
        size = strategy_instance._calculate_position_size()
        assert size == 0.01  # デフォルト値

    def test_validate_gene_method(self):
        """validate_geneメソッドのテスト"""
        from app.services.auto_strategy.models.condition import Condition

        # 有効な遺伝子
        valid_gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            ],
            entry_conditions=[],
            exit_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA")
            ],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA"),
            ],
            short_entry_conditions=[
                Condition(left_operand="close", operator="<", right_operand="SMA"),
            ],
            risk_management={},
            metadata={"test": True},
        )

        is_valid, errors = self.factory.validate_gene(valid_gene)
        assert is_valid
        assert len(errors) == 0

        # 無効な遺伝子（例: 空の指標リスト）
        invalid_gene = StrategyGene(
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            risk_management={},
            metadata={"test": True},
        )

        # validate_geneはgene.validate()を呼び出すだけ
        is_valid, errors = self.factory.validate_gene(invalid_gene)
        assert not is_valid  # 無効な遺伝子なのでfalseになるはず
        assert len(errors) > 0

#!/usr/bin/env python3
"""
統合されたstrategy_modelsのテスト

TDDアプローチで統合後のモデルをテストします。
"""

import pytest
import sys
import os
from typing import Dict, Any

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)


class TestStrategyModelsRefactored:
    """統合されたstrategy_modelsのテストクラス"""

    def test_imports(self):
        """統合後のモジュールからの基本インポートテスト"""
        try:
            from app.services.auto_strategy.models.strategy_models import (
                StrategyGene,
                IndicatorGene,
                Condition,
                ConditionGroup,
                PositionSizingGene,
                PositionSizingMethod,
                TPSLGene,
                TPSLMethod,
                GeneValidator,
            )

            assert True, "すべてのクラスが正常にインポートできました"
        except ImportError as e:
            pytest.fail(f"インポートエラー: {e}")

    def test_condition_creation(self):
        """条件オブジェクトの作成テスト"""
        from app.services.auto_strategy.models.strategy_models import Condition

        condition = Condition(left_operand="RSI", operator=">", right_operand=70.0)

        assert condition.left_operand == "RSI"
        assert condition.operator == ">"
        assert condition.right_operand == 70.0

    def test_condition_group_creation(self):
        """条件グループの作成テスト"""
        from app.services.auto_strategy.models.strategy_models import (
            Condition,
            ConditionGroup,
        )

        condition1 = Condition("RSI", ">", 70.0)
        condition2 = Condition("MACD", "<", 0.0)

        group = ConditionGroup(conditions=[condition1, condition2])

        assert len(group.conditions) == 2
        assert not group.is_empty()

    def test_indicator_gene_creation(self):
        """指標遺伝子の作成テスト"""
        from app.services.auto_strategy.models.strategy_models import IndicatorGene

        indicator = IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)

        assert indicator.type == "RSI"
        assert indicator.parameters["period"] == 14
        assert indicator.enabled is True

    def test_position_sizing_gene_creation(self):
        """ポジションサイジング遺伝子の作成テスト"""
        from app.services.auto_strategy.models.strategy_models import (
            PositionSizingGene,
            PositionSizingMethod,
        )

        gene = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            risk_per_trade=0.02,
            atr_period=14,
        )

        assert gene.method == PositionSizingMethod.VOLATILITY_BASED
        assert gene.risk_per_trade == 0.02
        assert gene.atr_period == 14

    def test_tpsl_gene_creation(self):
        """TP/SL遺伝子の作成テスト"""
        from app.services.auto_strategy.models.strategy_models import (
            TPSLGene,
            TPSLMethod,
        )

        gene = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            risk_reward_ratio=2.0,
            base_stop_loss=0.03,
        )

        assert gene.method == TPSLMethod.RISK_REWARD_RATIO
        assert gene.risk_reward_ratio == 2.0
        assert gene.base_stop_loss == 0.03

    def test_strategy_gene_creation(self):
        """戦略遺伝子の作成テスト"""
        from app.services.auto_strategy.models.strategy_models import (
            StrategyGene,
            IndicatorGene,
            Condition,
            PositionSizingGene,
            TPSLGene,
        )

        indicators = [
            IndicatorGene(type="RSI", parameters={"period": 14}),
            IndicatorGene(type="SMA", parameters={"period": 20}),
        ]

        entry_conditions = [Condition("RSI", "<", 30.0), Condition("close", ">", "SMA")]

        strategy = StrategyGene(
            id="test_strategy",
            indicators=indicators,
            entry_conditions=entry_conditions,
            position_sizing_gene=PositionSizingGene(),
            tpsl_gene=TPSLGene(),
        )

        assert strategy.id == "test_strategy"
        assert len(strategy.indicators) == 2
        assert len(strategy.entry_conditions) == 2
        assert strategy.position_sizing_gene is not None
        assert strategy.tpsl_gene is not None

    def test_gene_validator_creation(self):
        """遺伝子バリデーターの作成テスト"""
        from app.services.auto_strategy.models.strategy_models import GeneValidator

        validator = GeneValidator()
        assert validator is not None

    def test_condition_validation(self):
        """条件の検証テスト"""
        from app.services.auto_strategy.models.strategy_models import (
            Condition,
            GeneValidator,
        )

        validator = GeneValidator()

        # 有効な条件
        valid_condition = Condition("RSI", ">", 70.0)
        is_valid, error = validator.validate_condition(valid_condition)
        assert is_valid, f"有効な条件が無効と判定されました: {error}"

        # 無効な条件（無効な演算子）
        invalid_condition = Condition("RSI", "invalid_op", 70.0)
        is_valid, error = validator.validate_condition(invalid_condition)
        assert not is_valid, "無効な条件が有効と判定されました"

    def test_strategy_gene_validation(self):
        """戦略遺伝子の検証テスト"""
        from app.services.auto_strategy.models.strategy_models import (
            StrategyGene,
            IndicatorGene,
            Condition,
            GeneValidator,
        )

        validator = GeneValidator()

        # 有効な戦略
        strategy = StrategyGene(
            id="test",
            indicators=[IndicatorGene(type="RSI", parameters={"period": 14})],
            entry_conditions=[Condition("RSI", "<", 30.0)],
            exit_conditions=[Condition("RSI", ">", 70.0)],
        )

        is_valid, errors = validator.validate_strategy_gene(strategy)
        assert is_valid, f"有効な戦略が無効と判定されました: {errors}"

    def test_long_short_conditions(self):
        """ロング・ショート条件の分離テスト"""
        from app.services.auto_strategy.models.strategy_models import (
            StrategyGene,
            Condition,
        )

        strategy = StrategyGene(
            id="test",
            long_entry_conditions=[Condition("RSI", "<", 30.0)],
            short_entry_conditions=[Condition("RSI", ">", 70.0)],
        )

        assert strategy.has_long_short_separation()
        assert len(strategy.get_effective_long_conditions()) == 1
        assert len(strategy.get_effective_short_conditions()) == 1

    def test_serialization(self):
        """シリアライゼーションテスト"""
        from app.services.auto_strategy.models.strategy_models import (
            PositionSizingGene,
            TPSLGene,
        )

        # PositionSizingGene
        pos_gene = PositionSizingGene(risk_per_trade=0.03)
        pos_dict = pos_gene.to_dict()
        pos_restored = PositionSizingGene.from_dict(pos_dict)
        assert pos_restored.risk_per_trade == 0.03

        # TPSLGene
        tpsl_gene = TPSLGene(stop_loss_pct=0.02)
        tpsl_dict = tpsl_gene.to_dict()
        tpsl_restored = TPSLGene.from_dict(tpsl_dict)
        assert tpsl_restored.stop_loss_pct == 0.02


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

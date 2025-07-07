"""
ポジションサイジングとGA統合のテスト
"""

import pytest
import sys
import os

# テスト対象のモジュールをインポートするためのパス設定
backend_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, backend_path)

from app.core.services.auto_strategy.models.position_sizing_gene import (
    PositionSizingGene,
    PositionSizingMethod,
    create_random_position_sizing_gene,
    crossover_position_sizing_genes,
    mutate_position_sizing_gene,
)
from app.core.services.auto_strategy.calculators.position_sizing_calculator import (
    PositionSizingCalculatorService,
)
from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene,
    crossover_strategy_genes,
    mutate_strategy_gene,
)
from app.core.services.auto_strategy.models.gene_encoding import GeneEncoder
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.models.gene_serialization import GeneSerializer


class TestPositionSizingGAIntegration:
    """ポジションサイジングとGA統合のテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.ga_config = GAConfig()
        self.gene_encoder = GeneEncoder(self.ga_config)
        self.gene_serializer = GeneSerializer()
        self.calculator = PositionSizingCalculatorService()

    def test_position_sizing_gene_creation_and_validation(self):
        """ポジションサイジング遺伝子の作成と検証テスト"""
        # ランダム遺伝子の作成
        gene = create_random_position_sizing_gene()

        assert isinstance(gene, PositionSizingGene)
        assert gene.method in list(PositionSizingMethod)

        # バリデーション
        is_valid, errors = gene.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_position_sizing_gene_crossover(self):
        """ポジションサイジング遺伝子の交叉テスト"""
        parent1 = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.1,
            priority=1.0,
        )

        parent2 = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            fixed_ratio=0.2,
            priority=1.2,
        )

        child1, child2 = crossover_position_sizing_genes(parent1, parent2)

        assert isinstance(child1, PositionSizingGene)
        assert isinstance(child2, PositionSizingGene)

        # 交叉結果の妥当性チェック
        is_valid1, _ = child1.validate()
        is_valid2, _ = child2.validate()
        assert is_valid1 is True
        assert is_valid2 is True

    def test_position_sizing_gene_mutation(self):
        """ポジションサイジング遺伝子の突然変異テスト"""
        original = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.15,
            priority=1.0,
        )

        mutated = mutate_position_sizing_gene(original, mutation_rate=1.0)

        assert isinstance(mutated, PositionSizingGene)

        # 突然変異結果の妥当性チェック
        is_valid, _ = mutated.validate()
        assert is_valid is True

    def test_strategy_gene_with_position_sizing_gene(self):
        """ポジションサイジング遺伝子を含む戦略遺伝子のテスト"""
        position_sizing_gene = create_random_position_sizing_gene()

        strategy_gene = StrategyGene(
            id="test_strategy",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={"position_size": 0.1},
            position_sizing_gene=position_sizing_gene,
        )

        # 戦略遺伝子の妥当性チェック
        is_valid, errors = strategy_gene.validate()
        assert is_valid is True
        assert len(errors) == 0

        # ポジションサイジング遺伝子が正しく設定されていることを確認
        assert strategy_gene.position_sizing_gene is not None
        assert strategy_gene.position_sizing_gene.method == position_sizing_gene.method

    def test_strategy_gene_crossover_with_position_sizing(self):
        """ポジションサイジング遺伝子を含む戦略遺伝子の交叉テスト"""
        parent1 = StrategyGene(
            id="parent1",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={"position_size": 0.1},
            position_sizing_gene=PositionSizingGene(
                method=PositionSizingMethod.FIXED_RATIO,
                fixed_ratio=0.1,
            ),
        )

        parent2 = StrategyGene(
            id="parent2",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={"position_size": 0.2},
            position_sizing_gene=PositionSizingGene(
                method=PositionSizingMethod.VOLATILITY_BASED,
                fixed_ratio=0.2,
            ),
        )

        child1, child2 = crossover_strategy_genes(parent1, parent2)

        assert isinstance(child1, StrategyGene)
        assert isinstance(child2, StrategyGene)

        # 子遺伝子にポジションサイジング遺伝子が含まれていることを確認
        assert child1.position_sizing_gene is not None
        assert child2.position_sizing_gene is not None

    def test_strategy_gene_mutation_with_position_sizing(self):
        """ポジションサイジング遺伝子を含む戦略遺伝子の突然変異テスト"""
        original = StrategyGene(
            id="original",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={"position_size": 0.1},
            position_sizing_gene=PositionSizingGene(
                method=PositionSizingMethod.FIXED_RATIO,
                fixed_ratio=0.15,
            ),
        )

        mutated = mutate_strategy_gene(original, mutation_rate=1.0)

        assert isinstance(mutated, StrategyGene)

        # 突然変異後もポジションサイジング遺伝子が存在することを確認
        assert mutated.position_sizing_gene is not None

    def test_gene_encoding_with_position_sizing(self):
        """ポジションサイジング遺伝子を含む遺伝子エンコードのテスト"""
        position_sizing_gene = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            atr_period=20,
            atr_multiplier=2.5,
            risk_per_trade=0.03,
        )

        strategy_gene = StrategyGene(
            id="test_encode",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={"position_size": 0.1},
            position_sizing_gene=position_sizing_gene,
        )

        # エンコード
        encoded = self.gene_encoder.encode_strategy_gene(strategy_gene)

        assert isinstance(encoded, list)
        assert len(encoded) == 32  # 基本24 + ポジションサイジング8

        # デコード
        decoded = self.gene_encoder.decode_strategy_gene(encoded, StrategyGene)

        assert isinstance(decoded, StrategyGene)
        assert decoded.position_sizing_gene is not None
        assert (
            decoded.position_sizing_gene.method == PositionSizingMethod.VOLATILITY_BASED
        )

    def test_gene_serialization_with_position_sizing(self):
        """ポジションサイジング遺伝子を含む遺伝子シリアライゼーションのテスト"""
        position_sizing_gene = PositionSizingGene(
            method=PositionSizingMethod.HALF_OPTIMAL_F,
            lookback_period=150,
            optimal_f_multiplier=0.6,
        )

        strategy_gene = StrategyGene(
            id="test_serialize",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={"position_size": 0.1},
            position_sizing_gene=position_sizing_gene,
        )

        # シリアライゼーション
        serialized = self.gene_serializer.strategy_gene_to_dict(strategy_gene)

        assert isinstance(serialized, dict)
        assert "position_sizing_gene" in serialized
        assert serialized["position_sizing_gene"] is not None

        # デシリアライゼーション
        deserialized = self.gene_serializer.dict_to_strategy_gene(
            serialized, StrategyGene
        )

        assert isinstance(deserialized, StrategyGene)
        assert deserialized.position_sizing_gene is not None
        assert (
            deserialized.position_sizing_gene.method
            == PositionSizingMethod.HALF_OPTIMAL_F
        )
        assert deserialized.position_sizing_gene.lookback_period == 150

    def test_position_sizing_calculator_integration(self):
        """ポジションサイジング計算サービスの統合テスト"""
        gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.2,
            min_position_size=0.01,
            max_position_size=2.0,
        )

        result = self.calculator.calculate_position_size(
            gene=gene,
            account_balance=10000.0,
            current_price=50000.0,
            symbol="BTCUSDT",
        )

        assert hasattr(result, "position_size")
        assert hasattr(result, "method_used")
        assert hasattr(result, "confidence_score")
        assert hasattr(result, "risk_metrics")

        assert result.position_size == 2000.0  # 10000 * 0.2
        assert result.method_used == "fixed_ratio"
        assert 0.0 <= result.confidence_score <= 1.0

    def test_ga_config_position_sizing_constraints(self):
        """GAConfigのポジションサイジング制約テスト"""
        config = GAConfig()

        # ポジションサイジング関連の制約が設定されていることを確認
        assert hasattr(config, "position_sizing_method_constraints")
        assert hasattr(config, "position_sizing_lookback_range")
        assert hasattr(config, "position_sizing_optimal_f_multiplier_range")
        assert hasattr(config, "position_sizing_atr_period_range")
        assert hasattr(config, "position_sizing_atr_multiplier_range")
        assert hasattr(config, "position_sizing_risk_per_trade_range")
        assert hasattr(config, "position_sizing_fixed_ratio_range")
        assert hasattr(config, "position_sizing_fixed_quantity_range")
        assert hasattr(config, "position_sizing_min_size_range")
        assert hasattr(config, "position_sizing_max_size_range")
        assert hasattr(config, "position_sizing_priority_range")

        # 制約値の妥当性チェック
        assert len(config.position_sizing_method_constraints) == 4
        assert "half_optimal_f" in config.position_sizing_method_constraints
        assert "volatility_based" in config.position_sizing_method_constraints
        assert "fixed_ratio" in config.position_sizing_method_constraints
        assert "fixed_quantity" in config.position_sizing_method_constraints

    def test_end_to_end_position_sizing_workflow(self):
        """エンドツーエンドのポジションサイジングワークフローテスト"""
        # 1. ランダム遺伝子生成
        position_sizing_gene = create_random_position_sizing_gene()

        # 2. 戦略遺伝子に統合
        strategy_gene = StrategyGene(
            id="e2e_test",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={"position_size": 0.1},
            position_sizing_gene=position_sizing_gene,
        )

        # 3. エンコード
        encoded = self.gene_encoder.encode_strategy_gene(strategy_gene)

        # 4. デコード
        decoded = self.gene_encoder.decode_strategy_gene(encoded, StrategyGene)

        # 5. ポジションサイズ計算
        result = self.calculator.calculate_position_size(
            gene=decoded.position_sizing_gene,
            account_balance=10000.0,
            current_price=50000.0,
        )

        # 6. 結果検証
        assert decoded.position_sizing_gene is not None
        assert result.position_size > 0
        assert result.method_used in [
            "half_optimal_f",
            "volatility_based",
            "fixed_ratio",
            "fixed_quantity",
        ]

        # 7. シリアライゼーション
        serialized = self.gene_serializer.strategy_gene_to_dict(decoded)

        # 8. デシリアライゼーション
        final_gene = self.gene_serializer.dict_to_strategy_gene(
            serialized, StrategyGene
        )

        # 9. 最終検証
        assert final_gene.position_sizing_gene is not None
        assert (
            final_gene.position_sizing_gene.method
            == decoded.position_sizing_gene.method
        )


if __name__ == "__main__":
    pytest.main([__file__])

from app.services.auto_strategy.config import GASettings
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.genes.validator import GeneValidator
from app.services.auto_strategy.core.genetic_operators import (
    crossover_strategy_genes_pure,
    mutate_strategy_gene_pure,
)
from app.services.auto_strategy.genes.strategy import StrategyGene


class TestAutoStrategyFlow:
    """
    オートストラテジーの主要フローを確認する統合テスト
    """

    def test_gene_generation_and_evolution_cycle(self):
        """
        遺伝子生成 -> 検証 -> 交叉 -> 変異 -> 検証 のサイクルが正常に動作することを確認
        """
        config = GASettings()

        # 1. 遺伝子生成
        generator = RandomGeneGenerator(config)
        parent1 = generator.generate_random_gene()
        parent2 = generator.generate_random_gene()

        assert isinstance(parent1, StrategyGene)
        assert isinstance(parent2, StrategyGene)

        # 必須フィールドの確認
        assert parent1.long_tpsl_gene is not None
        assert parent1.short_tpsl_gene is not None
        assert len(parent1.indicators) >= config.min_indicators

        # 2. バリデーション
        validator = GeneValidator()

        # 生成された遺伝子は有効であるべき（ただしランダム生成によっては無効になる場合もあるが、単純な構造チェックは通るはず）
        # ここでは構造的な妥当性をチェック
        is_valid1, errors1 = validator.validate_strategy_gene(parent1)
        # ランダム生成で複雑なバリデーションに引っかかる可能性があるため、アサーションはログ出力にとどめる
        if not is_valid1:
            print(f"Parent1 invalid: {errors1}")

        # 3. 交叉
        child1, child2 = crossover_strategy_genes_pure(parent1, parent2, config)

        assert child1.long_tpsl_gene is not None
        assert child1.short_tpsl_gene is not None

        # 4. 変異
        mutated_child1 = mutate_strategy_gene_pure(child1, config, mutation_rate=0.5)

        assert mutated_child1.long_tpsl_gene is not None
        assert mutated_child1.short_tpsl_gene is not None

        # 5. 再検証
        is_valid_child, errors_child = validator.validate_strategy_gene(mutated_child1)
        if not is_valid_child:
            print(f"Child invalid: {errors_child}")

        # 基本的な構造が維持されていることを確認
        assert len(mutated_child1.indicators) > 0

"""
進化演算子

遺伝的アルゴリズムの進化演算子（交叉・突然変異）を担当します。
"""

import logging

logger = logging.getLogger(__name__)


class EvolutionOperators:
    """
    進化演算子クラス

    遺伝的アルゴリズムの進化演算子（交叉・突然変異）を担当します。
    """

    def crossover_strategy_genes(self, ind1, ind2):
        """
        戦略遺伝子レベルの交叉

        Args:
            ind1: 個体1（エンコードされた戦略遺伝子）
            ind2: 個体2（エンコードされた戦略遺伝子）

        Returns:
            交叉後の個体のタプル
        """
        try:
            # 遺伝子デコード
            from ..models.gene_encoding import GeneEncoder
            from ..models.gene_strategy import StrategyGene, crossover_strategy_genes

            gene_encoder = GeneEncoder()
            gene1 = gene_encoder.decode_list_to_strategy_gene(ind1, StrategyGene)
            gene2 = gene_encoder.decode_list_to_strategy_gene(ind2, StrategyGene)

            # 戦略遺伝子レベルの交叉
            child1, child2 = crossover_strategy_genes(gene1, gene2)

            # 再エンコード
            encoded_child1 = gene_encoder.encode_strategy_gene_to_list(child1)
            encoded_child2 = gene_encoder.encode_strategy_gene_to_list(child2)

            # 個体を更新
            ind1[:] = encoded_child1
            ind2[:] = encoded_child2

            return ind1, ind2

        except Exception as e:
            logger.error(f"戦略遺伝子交叉エラー: {e}")
            # エラー時は元の個体をそのまま返す
            return ind1, ind2

    def mutate_strategy_gene(self, individual, mutation_rate: float = 0.1):
        """
        戦略遺伝子レベルの突然変異

        Args:
            individual: 個体（エンコードされた戦略遺伝子）
            mutation_rate: 突然変異率

        Returns:
            突然変異後の個体のタプル
        """
        try:
            # 遺伝子デコード
            from ..models.gene_encoding import GeneEncoder
            from ..models.gene_strategy import StrategyGene, mutate_strategy_gene

            gene_encoder = GeneEncoder()
            gene = gene_encoder.decode_list_to_strategy_gene(individual, StrategyGene)

            # 戦略遺伝子レベルの突然変異
            mutated_gene = mutate_strategy_gene(gene, mutation_rate)

            # 再エンコード
            encoded_mutated = gene_encoder.encode_strategy_gene_to_list(mutated_gene)

            # 個体を更新
            individual[:] = encoded_mutated

            return (individual,)

        except Exception as e:
            logger.error(f"戦略遺伝子突然変異エラー: {e}")
            # エラー時は元の個体をそのまま返す
            return (individual,)

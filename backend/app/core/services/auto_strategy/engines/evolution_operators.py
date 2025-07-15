"""
進化演算子

遺伝的アルゴリズムの進化演算子（交叉・突然変異）を担当します。
"""

import logging
import random

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

    def mutate_with_short_bias(
        self, individual, mutation_rate: float = 0.1, short_bias_rate: float = 0.3
    ):
        """
        ショートバイアス付き突然変異

        Args:
            individual: 個体（エンコードされた戦略遺伝子）
            mutation_rate: 基本突然変異率
            short_bias_rate: ショートバイアス適用率

        Returns:
            突然変異後の個体のタプル
        """
        try:
            # 通常の突然変異を実行
            mutated_individual = self.mutate_strategy_gene(individual, mutation_rate)[0]

            # ショートバイアスを適用
            if random.random() < short_bias_rate:
                mutated_individual = self._apply_short_bias_mutation(mutated_individual)

            return (mutated_individual,)

        except Exception as e:
            logger.error(f"ショートバイアス突然変異エラー: {e}")
            return (individual,)

    def _apply_short_bias_mutation(self, individual):
        """
        ショートバイアス突然変異を適用

        Args:
            individual: 個体（エンコードされた戦略遺伝子）

        Returns:
            ショートバイアスが適用された個体
        """
        try:
            from ..models.gene_encoding import GeneEncoder
            from ..models.gene_strategy import StrategyGene
            from ..generators.smart_condition_generator import SmartConditionGenerator

            gene_encoder = GeneEncoder()
            gene = gene_encoder.decode_list_to_strategy_gene(individual, StrategyGene)

            # SmartConditionGeneratorを使用してショート特化条件を生成
            smart_generator = SmartConditionGenerator()

            # 既存のショート条件を拡張
            enhanced_short_conditions = (
                smart_generator.generate_enhanced_short_conditions(gene.indicators)
            )

            if enhanced_short_conditions:
                # 既存のショート条件を一部置き換え
                if (
                    hasattr(gene, "short_entry_conditions")
                    and gene.short_entry_conditions
                ):
                    # 既存のショート条件の一部を拡張条件で置き換え
                    replacement_count = min(
                        len(enhanced_short_conditions), len(gene.short_entry_conditions)
                    )
                    for i in range(replacement_count):
                        if random.random() < 0.5:  # 50%の確率で置き換え
                            gene.short_entry_conditions[i] = enhanced_short_conditions[
                                i
                            ]

                    # 追加の条件を加える
                    if len(enhanced_short_conditions) > replacement_count:
                        gene.short_entry_conditions.extend(
                            enhanced_short_conditions[
                                replacement_count : replacement_count + 1
                            ]
                        )
                else:
                    # ショート条件が存在しない場合は新規追加
                    gene.short_entry_conditions = enhanced_short_conditions[
                        :2
                    ]  # 最大2つの条件

            # 既存の条件にショートバイアスを適用
            if hasattr(gene, "entry_conditions") and gene.entry_conditions:
                gene.entry_conditions = smart_generator.apply_short_bias_mutation(
                    gene.entry_conditions, mutation_rate=0.4
                )

            # 再エンコード
            encoded_mutated = gene_encoder.encode_strategy_gene_to_list(gene)
            individual[:] = encoded_mutated

            return individual

        except Exception as e:
            logger.error(f"ショートバイアス適用エラー: {e}")
            return individual

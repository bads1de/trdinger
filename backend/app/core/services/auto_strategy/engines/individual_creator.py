"""
個体生成器

遺伝的アルゴリズムの個体生成を担当します。
"""

import logging

from ..generators.random_gene_generator import RandomGeneGenerator

logger = logging.getLogger(__name__)


class IndividualCreator:
    """
    個体生成器
    
    遺伝的アルゴリズムの個体生成を担当します。
    """

    def __init__(self, gene_generator: RandomGeneGenerator, individual_class):
        """初期化"""
        self.gene_generator = gene_generator
        self.Individual = individual_class

    def create_individual(self):
        """個体生成"""
        try:
            # RandomGeneGeneratorを使用して遺伝子を生成
            gene = self.gene_generator.generate_random_gene()

            # 遺伝子をエンコード
            from ..models.gene_encoding import GeneEncoder

            gene_encoder = GeneEncoder()
            encoded_gene = gene_encoder.encode_strategy_gene_to_list(gene)

            if not self.Individual:
                raise TypeError("個体クラス 'Individual' が初期化されていません。")
            return self.Individual(encoded_gene)

        except Exception as e:
            logger.error(f"個体生成中に致命的なエラーが発生しました: {e}")
            # 遺伝子生成はGAの根幹部分であり、失敗した場合は例外をスローして処理を停止するのが安全
            raise

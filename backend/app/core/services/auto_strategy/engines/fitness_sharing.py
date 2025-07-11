"""
フィットネス共有（Fitness Sharing）

遺伝的アルゴリズムにおけるニッチ形成を実現するためのフィットネス共有機能。
類似した個体のフィットネス値を調整することで、多様な戦略の共存を促進します。
"""

import logging
import math
from typing import List, Dict, Any
from ..models.gene_encoding import GeneEncoder
from ..models.gene_strategy import StrategyGene

logger = logging.getLogger(__name__)


class FitnessSharing:
    """
    フィットネス共有クラス
    
    個体間の類似度を計算し、類似した個体のフィットネス値を調整することで
    多様な戦略の共存を促進します。
    """

    def __init__(self, sharing_radius: float = 0.1, alpha: float = 1.0):
        """
        初期化

        Args:
            sharing_radius: 共有半径（類似度の閾値）
            alpha: 共有関数の形状パラメータ
        """
        self.sharing_radius = sharing_radius
        self.alpha = alpha
        self.gene_encoder = GeneEncoder()

    def apply_fitness_sharing(self, population: List[Any]) -> List[Any]:
        """
        個体群にフィットネス共有を適用

        Args:
            population: 個体群

        Returns:
            フィットネス共有適用後の個体群
        """
        try:
            if len(population) <= 1:
                return population

            # 各個体の戦略遺伝子を取得
            genes = []
            for individual in population:
                try:
                    gene = self.gene_encoder.decode_list_to_strategy_gene(
                        individual, StrategyGene
                    )
                    genes.append(gene)
                except Exception as e:
                    logger.warning(f"個体のデコードに失敗: {e}")
                    genes.append(None)

            # 各個体のニッチカウントを計算
            niche_counts = []
            for i, gene_i in enumerate(genes):
                if gene_i is None:
                    niche_counts.append(1.0)
                    continue

                niche_count = 0.0
                for j, gene_j in enumerate(genes):
                    if gene_j is None:
                        continue

                    # 類似度を計算
                    similarity = self._calculate_similarity(gene_i, gene_j)
                    
                    # 共有関数を適用
                    sharing_value = self._sharing_function(similarity)
                    niche_count += sharing_value

                niche_counts.append(max(1.0, niche_count))

            # フィットネス値を調整
            for i, individual in enumerate(population):
                if hasattr(individual, 'fitness') and individual.fitness.valid:
                    original_fitness = individual.fitness.values[0]
                    shared_fitness = original_fitness / niche_counts[i]
                    individual.fitness.values = (shared_fitness,)

            return population

        except Exception as e:
            logger.error(f"フィットネス共有適用エラー: {e}")
            return population

    def _calculate_similarity(self, gene1: StrategyGene, gene2: StrategyGene) -> float:
        """
        2つの戦略遺伝子間の類似度を計算

        Args:
            gene1: 戦略遺伝子1
            gene2: 戦略遺伝子2

        Returns:
            類似度（0.0-1.0）
        """
        try:
            similarity_scores = []

            # 指標の類似度
            indicator_similarity = self._calculate_indicator_similarity(
                gene1.indicators, gene2.indicators
            )
            similarity_scores.append(indicator_similarity)

            # 買い条件の類似度
            buy_similarity = self._calculate_condition_similarity(
                gene1.buy_conditions, gene2.buy_conditions
            )
            similarity_scores.append(buy_similarity)

            # 売り条件の類似度
            sell_similarity = self._calculate_condition_similarity(
                gene1.sell_conditions, gene2.sell_conditions
            )
            similarity_scores.append(sell_similarity)

            # リスク管理の類似度
            risk_similarity = self._calculate_risk_management_similarity(
                gene1.risk_management, gene2.risk_management
            )
            similarity_scores.append(risk_similarity)

            # 重み付き平均（指標とリスク管理を重視）
            weights = [0.3, 0.25, 0.25, 0.2]
            weighted_similarity = sum(
                score * weight for score, weight in zip(similarity_scores, weights)
            )

            return max(0.0, min(1.0, weighted_similarity))

        except Exception as e:
            logger.error(f"類似度計算エラー: {e}")
            return 0.0

    def _calculate_indicator_similarity(self, indicators1: List[Any], indicators2: List[Any]) -> float:
        """指標の類似度を計算"""
        try:
            if not indicators1 or not indicators2:
                return 0.0 if indicators1 != indicators2 else 1.0

            # 指標タイプの集合を作成
            types1 = {ind.type for ind in indicators1}
            types2 = {ind.type for ind in indicators2}

            # Jaccard係数を計算
            intersection = len(types1 & types2)
            union = len(types1 | types2)

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def _calculate_condition_similarity(self, conditions1: List[Any], conditions2: List[Any]) -> float:
        """条件の類似度を計算"""
        try:
            if not conditions1 or not conditions2:
                return 0.0 if conditions1 != conditions2 else 1.0

            # 条件の構造的類似度を計算
            similar_conditions = 0
            total_conditions = max(len(conditions1), len(conditions2))

            for i in range(min(len(conditions1), len(conditions2))):
                cond1 = conditions1[i]
                cond2 = conditions2[i]

                # 演算子の一致
                if hasattr(cond1, 'operator') and hasattr(cond2, 'operator'):
                    if cond1.operator == cond2.operator:
                        similar_conditions += 0.5

                # オペランドタイプの一致（簡易版）
                if hasattr(cond1, 'left_operand') and hasattr(cond2, 'left_operand'):
                    if str(type(cond1.left_operand)) == str(type(cond2.left_operand)):
                        similar_conditions += 0.5

            return similar_conditions / total_conditions if total_conditions > 0 else 0.0

        except Exception:
            return 0.0

    def _calculate_risk_management_similarity(self, risk1: Dict[str, Any], risk2: Dict[str, Any]) -> float:
        """リスク管理の類似度を計算"""
        try:
            if not risk1 or not risk2:
                return 0.0 if risk1 != risk2 else 1.0

            similarity_score = 0.0
            total_fields = 0

            # 共通フィールドの類似度を計算
            common_fields = set(risk1.keys()) & set(risk2.keys())
            
            for field in common_fields:
                total_fields += 1
                val1 = risk1[field]
                val2 = risk2[field]

                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # 数値の場合は相対差を計算
                    if val1 == 0 and val2 == 0:
                        similarity_score += 1.0
                    elif val1 != 0 or val2 != 0:
                        max_val = max(abs(val1), abs(val2))
                        if max_val > 0:
                            diff = abs(val1 - val2) / max_val
                            similarity_score += max(0.0, 1.0 - diff)
                else:
                    # その他の場合は完全一致
                    if val1 == val2:
                        similarity_score += 1.0

            return similarity_score / total_fields if total_fields > 0 else 0.0

        except Exception:
            return 0.0

    def _sharing_function(self, similarity: float) -> float:
        """
        共有関数

        Args:
            similarity: 類似度

        Returns:
            共有値
        """
        if similarity < self.sharing_radius:
            return 0.0
        else:
            return 1.0 - (similarity / self.sharing_radius) ** self.alpha

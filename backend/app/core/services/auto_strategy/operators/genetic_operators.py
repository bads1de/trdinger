"""
遺伝的演算子

戦略遺伝子の交叉・突然変異ロジックを担当します。
"""

import random
import copy
import uuid
import logging
from typing import Tuple

from ..models.strategy_gene import StrategyGene

logger = logging.getLogger(__name__)


def crossover_strategy_genes(
    parent1: StrategyGene, parent2: StrategyGene
) -> Tuple[StrategyGene, StrategyGene]:
    """
    戦略遺伝子の交叉

    指標遺伝子、条件、TP/SL遺伝子すべてを含む完全な交叉を実行します。

    Args:
        parent1: 親1の戦略遺伝子
        parent2: 親2の戦略遺伝子

    Returns:
        交叉後の子1、子2の戦略遺伝子のタプル
    """
    try:
        # 指標遺伝子の交叉（単純な一点交叉）
        crossover_point = random.randint(
            1, min(len(parent1.indicators), len(parent2.indicators))
        )

        child1_indicators = (
            parent1.indicators[:crossover_point] + parent2.indicators[crossover_point:]
        )
        child2_indicators = (
            parent2.indicators[:crossover_point] + parent1.indicators[crossover_point:]
        )

        # 最大指標数制限
        max_indicators = getattr(parent1, "MAX_INDICATORS", 5)
        child1_indicators = child1_indicators[:max_indicators]
        child2_indicators = child2_indicators[:max_indicators]

        # 条件の交叉（ランダム選択）
        if random.random() < 0.5:
            child1_entry = parent1.entry_conditions.copy()
            child2_entry = parent2.entry_conditions.copy()
        else:
            child1_entry = parent2.entry_conditions.copy()
            child2_entry = parent1.entry_conditions.copy()

        if random.random() < 0.5:
            child1_exit = parent1.exit_conditions.copy()
            child2_exit = parent2.exit_conditions.copy()
        else:
            child1_exit = parent2.exit_conditions.copy()
            child2_exit = parent1.exit_conditions.copy()

        # リスク管理設定の交叉（平均値）
        child1_risk = {}
        child2_risk = {}

        all_keys = set(parent1.risk_management.keys()) | set(
            parent2.risk_management.keys()
        )
        for key in all_keys:
            val1 = parent1.risk_management.get(key, 0)
            val2 = parent2.risk_management.get(key, 0)

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                child1_risk[key] = (val1 + val2) / 2
                child2_risk[key] = (val1 + val2) / 2
            else:
                child1_risk[key] = val1 if random.random() < 0.5 else val2
                child2_risk[key] = val2 if random.random() < 0.5 else val1

        # TP/SL遺伝子の交叉
        child1_tpsl = None
        child2_tpsl = None

        if parent1.tpsl_gene and parent2.tpsl_gene:
            from ..models.tpsl_gene import crossover_tpsl_genes
            child1_tpsl, child2_tpsl = crossover_tpsl_genes(
                parent1.tpsl_gene, parent2.tpsl_gene
            )
        elif parent1.tpsl_gene:
            child1_tpsl = parent1.tpsl_gene
            child2_tpsl = parent1.tpsl_gene  # コピー
        elif parent2.tpsl_gene:
            child1_tpsl = parent2.tpsl_gene
            child2_tpsl = parent2.tpsl_gene  # コピー

        # ポジションサイジング遺伝子の交叉
        child1_position_sizing = None
        child2_position_sizing = None

        if getattr(parent1, "position_sizing_gene", None) and getattr(
            parent2, "position_sizing_gene", None
        ):
            from ..models.position_sizing_gene import crossover_position_sizing_genes

            child1_position_sizing, child2_position_sizing = (
                crossover_position_sizing_genes(
                    parent1.position_sizing_gene, parent2.position_sizing_gene
                )
            )
        elif getattr(parent1, "position_sizing_gene", None):
            child1_position_sizing = parent1.position_sizing_gene
            child2_position_sizing = parent1.position_sizing_gene  # コピー
        elif getattr(parent2, "position_sizing_gene", None):
            child1_position_sizing = parent2.position_sizing_gene
            child2_position_sizing = parent2.position_sizing_gene  # コピー

        # メタデータの継承
        child1_metadata = parent1.metadata.copy()
        child1_metadata["crossover_parent1"] = parent1.id
        child1_metadata["crossover_parent2"] = parent2.id

        child2_metadata = parent2.metadata.copy()
        child2_metadata["crossover_parent1"] = parent1.id
        child2_metadata["crossover_parent2"] = parent2.id

        # 子遺伝子の作成
        child1 = StrategyGene(
            id=str(uuid.uuid4()),
            indicators=child1_indicators,
            entry_conditions=child1_entry,
            exit_conditions=child1_exit,
            risk_management=child1_risk,
            tpsl_gene=child1_tpsl,
            position_sizing_gene=child1_position_sizing,
            metadata=child1_metadata,
        )

        child2 = StrategyGene(
            id=str(uuid.uuid4()),
            indicators=child2_indicators,
            entry_conditions=child2_entry,
            exit_conditions=child2_exit,
            risk_management=child2_risk,
            tpsl_gene=child2_tpsl,
            position_sizing_gene=child2_position_sizing,
            metadata=child2_metadata,
        )

        return child1, child2

    except Exception as e:
        logger.error(f"戦略遺伝子交叉エラー: {e}")
        # エラー時は親をそのまま返す
        return parent1, parent2


def mutate_strategy_gene(
    gene: StrategyGene, mutation_rate: float = 0.1
) -> StrategyGene:
    """
    戦略遺伝子の突然変異

    指標遺伝子、条件、TP/SL遺伝子すべてを含む完全な突然変異を実行します。

    Args:
        gene: 突然変異対象の戦略遺伝子
        mutation_rate: 突然変異率

    Returns:
        突然変異後の戦略遺伝子
    """
    try:
        # 深いコピーを作成
        mutated = copy.deepcopy(gene)

        # 指標遺伝子の突然変異
        for i, indicator in enumerate(mutated.indicators):
            if random.random() < mutation_rate:
                # パラメータの突然変異
                for param_name, param_value in indicator.parameters.items():
                    if (
                        isinstance(param_value, (int, float))
                        and random.random() < mutation_rate
                    ):
                        if param_name == "period":
                            # 期間パラメータの場合
                            mutated.indicators[i].parameters[param_name] = max(
                                1, min(200, int(param_value * random.uniform(0.8, 1.2)))
                            )
                        else:
                            # その他の数値パラメータ
                            mutated.indicators[i].parameters[param_name] = (
                                param_value * random.uniform(0.8, 1.2)
                            )

        # 指標の追加・削除（低確率）
        if random.random() < mutation_rate * 0.3:
            max_indicators = getattr(gene, "MAX_INDICATORS", 5)

            if len(mutated.indicators) < max_indicators and random.random() < 0.5:
                # 新しい指標を追加
                from ..generators.random_gene_generator import RandomGeneGenerator
                from ..models.ga_config import GAConfig

                generator = RandomGeneGenerator(GAConfig())
                new_indicators = generator._generate_random_indicators()
                if new_indicators:
                    mutated.indicators.append(random.choice(new_indicators))

            elif len(mutated.indicators) > 1 and random.random() < 0.5:
                # 指標を削除
                mutated.indicators.pop(random.randint(0, len(mutated.indicators) - 1))

        # 条件の突然変異（低確率）
        if random.random() < mutation_rate * 0.5:
            # エントリー条件の変更
            if mutated.entry_conditions and random.random() < 0.5:
                condition_idx = random.randint(0, len(mutated.entry_conditions) - 1)
                condition = mutated.entry_conditions[condition_idx]

                # オペレーターの変更
                operators = [">", "<", ">=", "<=", "=="]
                condition.operator = random.choice(operators)

        if random.random() < mutation_rate * 0.5:
            # エグジット条件の変更
            if mutated.exit_conditions and random.random() < 0.5:
                condition_idx = random.randint(0, len(mutated.exit_conditions) - 1)
                condition = mutated.exit_conditions[condition_idx]

                # オペレーターの変更
                operators = [">", "<", ">=", "<=", "=="]
                condition.operator = random.choice(operators)

        # リスク管理設定の突然変異
        for key, value in mutated.risk_management.items():
            if isinstance(value, (int, float)) and random.random() < mutation_rate:
                if key == "position_size":
                    # ポジションサイズの場合
                    mutated.risk_management[key] = max(
                        0.01, min(1.0, value * random.uniform(0.8, 1.2))
                    )
                else:
                    # その他の数値設定
                    mutated.risk_management[key] = value * random.uniform(0.8, 1.2)

        # TP/SL遺伝子の突然変異
        if mutated.tpsl_gene:
            if random.random() < mutation_rate:
                from ..models.tpsl_gene import mutate_tpsl_gene
                mutated.tpsl_gene = mutate_tpsl_gene(mutated.tpsl_gene, mutation_rate)
        else:
            # TP/SL遺伝子が存在しない場合、低確率で新規作成
            if random.random() < mutation_rate * 0.2:
                from ..models.tpsl_gene import create_random_tpsl_gene
                mutated.tpsl_gene = create_random_tpsl_gene()

        # ポジションサイジング遺伝子の突然変異
        if getattr(mutated, "position_sizing_gene", None):
            if random.random() < mutation_rate:
                from ..models.position_sizing_gene import mutate_position_sizing_gene

                mutated.position_sizing_gene = mutate_position_sizing_gene(
                    mutated.position_sizing_gene, mutation_rate
                )
        else:
            # ポジションサイジング遺伝子が存在しない場合、低確率で新規作成
            if random.random() < mutation_rate * 0.2:
                from ..models.position_sizing_gene import create_random_position_sizing_gene

                mutated.position_sizing_gene = create_random_position_sizing_gene()

        # メタデータの更新
        mutated.metadata["mutated"] = True
        mutated.metadata["mutation_rate"] = mutation_rate

        return mutated

    except Exception as e:
        logger.error(f"戦略遺伝子突然変異エラー: {e}")
        # エラー時は元の遺伝子をそのまま返す
        return gene

"""
遺伝的アルゴリズム関連ユーティリティ

遺伝子操作（交叉・変異）のための汎用機能を提供します。
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union


class GeneticUtils:
    """遺伝的アルゴリズム関連ユーティリティ"""

    @staticmethod
    def create_child_metadata(
        parent1_metadata: Dict[str, Any],
        parent2_metadata: Dict[str, Any],
        parent1_id: str,
        parent2_id: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        交叉子のメタデータを生成

        Args:
            parent1_metadata: 親1のメタデータ
            parent2_metadata: 親2のメタデータ
            parent1_id: 親1のID
            parent2_id: 親2のID

        Returns:
            (child1_metadata, child2_metadata) のタプル
        """
        # 子1のメタデータ：親1のメタデータを継承し、親情報を追加
        child1_metadata = parent1_metadata.copy()
        child1_metadata["crossover_parent1"] = parent1_id
        child1_metadata["crossover_parent2"] = parent2_id

        # 子2のメタデータ：親2のメタデータを継承し、親情報を追加
        child2_metadata = parent2_metadata.copy()
        child2_metadata["crossover_parent1"] = parent1_id
        child2_metadata["crossover_parent2"] = parent2_id

        return child1_metadata, child2_metadata

    @staticmethod
    def prepare_crossover_metadata(strategy_parent1, strategy_parent2):
        """
        戦略遺伝子交叉用のメタデータ作成ユーティリティ

        Args:
            strategy_parent1: 親1のStrategyGene
            strategy_parent2: 親2のStrategyGene

        Returns:
            (child1_metadata, child2_metadata) のタプル
        """
        return GeneticUtils.create_child_metadata(
            parent1_metadata=strategy_parent1.metadata,
            parent2_metadata=strategy_parent2.metadata,
            parent1_id=strategy_parent1.id,
            parent2_id=strategy_parent2.id,
        )

    @staticmethod
    def crossover_generic_genes(
        parent1_gene,
        parent2_gene,
        gene_class,
        numeric_fields: Optional[List[str]] = None,
        enum_fields: Optional[List[str]] = None,
        choice_fields: Optional[List[str]] = None,
    ):
        """
        ジェネリック遺伝子交叉関数

        Args:
            parent1_gene: 親1遺伝子
            parent2_gene: 親2遺伝子
            gene_class: 遺伝子クラス
            numeric_fields: 数値フィールドのリスト（平均化）
            enum_fields: Enumフィールドのリスト（ランダム選択）
            choice_fields: どちらかを選択するフィールドのリスト

        Returns:
            (child1_gene, child2_gene) のタプル
        """
        if numeric_fields is None:
            numeric_fields = []
        if enum_fields is None:
            enum_fields = []
        if choice_fields is None:
            choice_fields = []

        # 共通のフィールド処理
        parent1_dict = parent1_gene.__dict__.copy()
        parent2_dict = parent2_gene.__dict__.copy()

        child1_params = {}
        child2_params = {}

        # 全フィールドを取得
        all_fields = set(parent1_dict.keys()) & set(parent2_dict.keys())
        all_fields.discard("_")  # プライベート属性を除外

        for field in all_fields:
            if field in choice_fields:
                # どちらかを選択
                child1_params[field] = random.choice(
                    [parent1_dict[field], parent2_dict[field]]
                )
                child2_params[field] = random.choice(
                    [parent1_dict[field], parent2_dict[field]]
                )
            elif field in numeric_fields:
                # 数値フィールドは平均化
                val1 = parent1_dict[field]
                val2 = parent2_dict[field]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    child1_params[field] = (val1 + val2) / 2
                    child2_params[field] = (val1 + val2) / 2
                else:
                    child1_params[field] = val1
                    child2_params[field] = val2
            elif field in enum_fields:
                # Enumフィールドはランダム選択
                child1_params[field] = random.choice(
                    [parent1_dict[field], parent2_dict[field]]
                )
                child2_params[field] = random.choice(
                    [parent1_dict[field], parent2_dict[field]]
                )
            else:
                # デフォルト：交互に選択
                if random.random() < 0.5:
                    child1_params[field] = parent1_dict[field]
                    child2_params[field] = parent2_dict[field]
                else:
                    child1_params[field] = parent2_dict[field]
                    child2_params[field] = parent1_dict[field]

        # 子遺伝子作成
        child1 = gene_class(**child1_params)
        child2 = gene_class(**child2_params)

        return child1, child2

    @staticmethod
    def mutate_generic_gene(
        gene,
        gene_class,
        mutation_rate: float = 0.1,
        numeric_fields: Optional[List[str]] = None,
        enum_fields: Optional[List[str]] = None,
        numeric_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        ジェネリック遺伝子突然変異関数

        Args:
            gene: 突然変異対象の遺伝子
            gene_class: 遺伝子クラス
            mutation_rate: 突然変異率
            numeric_fields: 数値フィールドのリスト
            enum_fields: Enumフィールドのリスト
            numeric_ranges: 数値フィールドの範囲 {"field_name": (min_val, max_val)}

        Returns:
            突然変異後の遺伝子
        """
        import random

        if numeric_fields is None:
            numeric_fields = []
        if enum_fields is None:
            enum_fields = []
        if numeric_ranges is None:
            numeric_ranges = {}

        # 遺伝子のコピーを作成
        mutated_params = gene.__dict__.copy()

        # 数値フィールドの突然変異
        for field in numeric_fields:
            if random.random() < mutation_rate and field in mutated_params:
                current_value = mutated_params[field]
                if isinstance(current_value, (int, float)):
                    # 値の範囲を取得（指定がない場合はデフォルト範囲）
                    min_val, max_val = numeric_ranges.get(field, (0, 100))

                    # 現在の値を中心とした範囲で突然変異
                    mutation_factor = random.uniform(0.8, 1.2)
                    new_value = current_value * mutation_factor

                    # 範囲内に制限
                    new_value = max(min_val, min(max_val, new_value))

                    mutated_params[field] = new_value

                    # int型だった場合はintに変換
                    if isinstance(current_value, int):
                        mutated_params[field] = int(new_value)

        # Enumフィールドの突然変異
        for field in enum_fields:
            if random.random() < mutation_rate and field in mutated_params:
                current_enum = mutated_params[field]
                if hasattr(current_enum, "__class__"):
                    enum_class = current_enum.__class__
                    # ランダムに別の値を選択
                    mutated_params[field] = random.choice(list(enum_class))

        return gene_class(**mutated_params)

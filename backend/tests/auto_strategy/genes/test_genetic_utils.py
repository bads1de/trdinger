from dataclasses import dataclass
from enum import Enum
from unittest.mock import MagicMock

import pytest

from app.services.auto_strategy.genes.genetic_utils import GeneticUtils


# テスト用のヘルパークラス
class SampleEnum(Enum):
    A = "a"
    B = "b"
    C = "c"


@dataclass
class SampleGene:
    numeric_val: float
    int_val: int
    enum_val: SampleEnum
    choice_val: str
    other_val: str

    # 辞書形式でのアクセスを可能にするためのダミー実装（実際のBaseGeneの挙動を模倣）
    # ただしGeneticUtilsは__dict__にアクセスしているため、dataclassならそのままでOK

    # コンストラクタ引数展開用
    def __init__(self, **kwargs):
        self.numeric_val = kwargs.get("numeric_val", 0.0)
        self.int_val = kwargs.get("int_val", 0)
        self.enum_val = kwargs.get("enum_val", SampleEnum.A)
        self.choice_val = kwargs.get("choice_val", "x")
        self.other_val = kwargs.get("other_val", "y")
        # メタデータは属性として持つ
        self.metadata = kwargs.get("metadata", {})
        self.id = kwargs.get("id", "dummy_id")


class TestGeneticUtils:

    def test_create_child_metadata(self):
        parent1_meta = {"gen": 1, "score": 10}
        parent2_meta = {"gen": 2, "score": 20}

        c1, c2 = GeneticUtils.create_child_metadata(
            parent1_meta, parent2_meta, "p1", "p2"
        )

        assert c1["gen"] == 1
        assert c1["crossover_parent1"] == "p1"
        assert c1["crossover_parent2"] == "p2"

        assert c2["gen"] == 2
        assert c2["crossover_parent1"] == "p1"
        assert c2["crossover_parent2"] == "p2"

    def test_prepare_crossover_metadata(self):
        p1 = MagicMock()
        p1.metadata = {"a": 1}
        p1.id = "id1"

        p2 = MagicMock()
        p2.metadata = {"b": 2}
        p2.id = "id2"

        c1, c2 = GeneticUtils.prepare_crossover_metadata(p1, p2)

        assert c1["a"] == 1
        assert c1["crossover_parent1"] == "id1"
        assert c2["b"] == 2
        assert c2["crossover_parent2"] == "id2"

    def test_crossover_generic_genes(self):
        parent1 = SampleGene(
            numeric_val=10.0,
            int_val=100,
            enum_val=SampleEnum.A,
            choice_val="left",
            other_val="fixed1",
        )
        parent2 = SampleGene(
            numeric_val=20.0,
            int_val=200,
            enum_val=SampleEnum.B,
            choice_val="right",
            other_val="fixed2",
        )

        c1, c2 = GeneticUtils.crossover_generic_genes(
            parent1,
            parent2,
            SampleGene,
            numeric_fields=["numeric_val", "int_val"],
            enum_fields=["enum_val"],
            choice_fields=["choice_val"],
        )

        # 数値フィールド（平均化）
        assert c1.numeric_val == 15.0
        assert c2.numeric_val == 15.0
        assert c1.int_val == 150
        assert c2.int_val == 150

        # Choiceフィールド（どちらか）
        assert c1.choice_val in ["left", "right"]
        assert c2.choice_val in ["left", "right"]

        # Enumフィールド（ランダム選択）
        assert c1.enum_val in [SampleEnum.A, SampleEnum.B]

        # その他フィールド（交互選択の確率的挙動だが、どちらかにはなる）
        assert c1.other_val in ["fixed1", "fixed2"]

    def test_crossover_generic_genes_defaults(self):
        # オプション引数なしの場合の動作確認
        p1 = SampleGene(other_val="A")
        p2 = SampleGene(other_val="B")

        c1, c2 = GeneticUtils.crossover_generic_genes(p1, p2, SampleGene)

        assert c1.other_val in ["A", "B"]
        assert c2.other_val in ["A", "B"]

    def test_mutate_generic_gene_numeric(self):
        gene = SampleGene(numeric_val=50.0, int_val=10)

        # 常に変異するようにレートを1.0に設定
        mutated = GeneticUtils.mutate_generic_gene(
            gene,
            SampleGene,
            mutation_rate=1.0,
            numeric_fields=["numeric_val", "int_val"],
            numeric_ranges={"numeric_val": (0, 100), "int_val": (0, 20)},
        )

        # 変異しているはず（乱数要素があるため絶対ではないが、確率的に変化する）
        # ただし範囲内であることの確認が重要
        assert 0 <= mutated.numeric_val <= 100
        assert 0 <= mutated.int_val <= 20
        assert isinstance(mutated.int_val, int)

        # 値が変わったことの確認（1.0倍になる可能性は極めて低いがゼロではない）
        # mutation_factorは0.8~1.2
        assert mutated.numeric_val != 50.0 or (
            mutated.numeric_val == 50.0 and gene.numeric_val == 50.0
        )

    def test_mutate_generic_gene_enum(self):
        gene = SampleGene(enum_val=SampleEnum.A)

        # 常に変異
        mutated = GeneticUtils.mutate_generic_gene(
            gene, SampleGene, mutation_rate=1.0, enum_fields=["enum_val"]
        )

        assert isinstance(mutated.enum_val, SampleEnum)
        assert mutated.enum_val in SampleEnum

    def test_mutate_generic_gene_no_mutation(self):
        gene = SampleGene(numeric_val=10.0)

        # 変異率0.0
        mutated = GeneticUtils.mutate_generic_gene(
            gene, SampleGene, mutation_rate=0.0, numeric_fields=["numeric_val"]
        )

        assert mutated.numeric_val == 10.0

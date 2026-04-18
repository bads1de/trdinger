"""
ExitGene の単体テスト
"""

from unittest.mock import MagicMock

import pytest

from app.services.auto_strategy.config.constants import ExitType
from app.services.auto_strategy.genes.exit import ExitGene, create_random_exit_gene


class TestExitGene:

    def test_init_default(self):
        gene = ExitGene()
        assert gene.exit_type == ExitType.FULL
        assert gene.partial_exit_pct == 0.5
        assert gene.partial_exit_enabled is False
        assert gene.trailing_stop_activation is False
        assert gene.enabled is True
        assert gene.priority == 1.0

    def test_validate_valid(self):
        gene = ExitGene(
            exit_type=ExitType.PARTIAL,
            partial_exit_pct=0.3,
            partial_exit_enabled=True,
        )
        is_valid, errors = gene.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_invalid_partial_exit_pct_low(self):
        gene = ExitGene(partial_exit_pct=0.05)
        is_valid, errors = gene.validate()
        assert is_valid is False
        assert "partial_exit_pct" in errors[0]

    def test_validate_invalid_partial_exit_pct_high(self):
        gene = ExitGene(partial_exit_pct=0.95)
        is_valid, errors = gene.validate()
        assert is_valid is False
        assert "partial_exit_pct" in errors[0]

    def test_validate_invalid_exit_type(self):
        gene = ExitGene()
        gene.exit_type = "invalid_type"
        is_valid, errors = gene.validate()
        assert is_valid is False
        assert "exit_type" in errors[0]

    def test_validate_boundary_values(self):
        """境界値のテスト"""
        gene = ExitGene(partial_exit_pct=0.1)
        is_valid, errors = gene.validate()
        assert is_valid is True

        gene = ExitGene(partial_exit_pct=0.9)
        is_valid, errors = gene.validate()
        assert is_valid is True

    def test_to_dict(self):
        gene = ExitGene(
            exit_type=ExitType.TRAILING,
            partial_exit_pct=0.4,
            trailing_stop_activation=True,
        )
        data = gene.to_dict()
        assert data["exit_type"] == "trailing"
        assert data["partial_exit_pct"] == 0.4
        assert data["trailing_stop_activation"] is True

    def test_from_dict(self):
        data = {
            "exit_type": "partial",
            "partial_exit_pct": 0.6,
            "partial_exit_enabled": True,
            "trailing_stop_activation": False,
            "enabled": False,
            "priority": 1.5,
        }
        gene = ExitGene.from_dict(data)
        assert gene.exit_type == ExitType.PARTIAL
        assert gene.partial_exit_pct == 0.6
        assert gene.partial_exit_enabled is True
        assert gene.trailing_stop_activation is False
        assert gene.enabled is False
        assert gene.priority == 1.5

    def test_from_dict_enum_handling(self):
        data = {"exit_type": ExitType.FULL}
        gene = ExitGene.from_dict(data)
        assert gene.exit_type == ExitType.FULL

        with pytest.raises(ValueError):
            ExitGene.from_dict({"exit_type": "unknown"})

    def test_clone(self):
        gene = ExitGene(
            exit_type=ExitType.PARTIAL,
            partial_exit_pct=0.4,
            partial_exit_enabled=True,
        )
        cloned = gene.clone()
        assert cloned.exit_type == gene.exit_type
        assert cloned.partial_exit_pct == gene.partial_exit_pct
        assert cloned is not gene

    def test_mutate_type_change(self):
        gene = ExitGene(exit_type=ExitType.FULL)
        # 十分な回数の突然変異を試行して、少なくとも1回はタイプが変わることを確認
        changed = False
        for _ in range(100):
            mutated = gene.mutate(mutation_rate=1.0)
            if mutated.exit_type != ExitType.FULL:
                changed = True
                break
        assert changed

    def test_mutate_partial_exit_pct_range(self):
        gene = ExitGene(partial_exit_pct=0.5)
        for _ in range(50):
            mutated = gene.mutate(mutation_rate=1.0)
            assert 0.1 <= mutated.partial_exit_pct <= 0.9

    def test_mutate_flag_toggle(self):
        gene = ExitGene(
            partial_exit_enabled=False,
            trailing_stop_activation=False,
        )
        # フラグがトグルされることを確認
        for _ in range(100):
            mutated = gene.mutate(mutation_rate=1.0)
            if mutated.partial_exit_enabled or mutated.trailing_stop_activation:
                break
        else:
            pytest.fail("フラグがトグルされなかった")

    def test_crossover(self):
        parent1 = ExitGene(exit_type=ExitType.FULL, partial_exit_pct=0.3)
        parent2 = ExitGene(exit_type=ExitType.PARTIAL, partial_exit_pct=0.7)

        child1, child2 = ExitGene.crossover(parent1, parent2)

        assert isinstance(child1, ExitGene)
        assert isinstance(child2, ExitGene)
        assert child1 is not parent1
        assert child2 is not parent2
        # 数値フィールドは平均化される: (0.3 + 0.7) / 2 = 0.5
        assert child1.partial_exit_pct == 0.5
        assert child2.partial_exit_pct == 0.5

    def test_create_random_exit_gene_defaults(self):
        gene = create_random_exit_gene()
        assert isinstance(gene, ExitGene)
        assert isinstance(gene.exit_type, ExitType)
        assert 0.2 <= gene.partial_exit_pct <= 0.8
        assert gene.enabled is True

    def test_create_random_exit_gene_with_config(self):
        config = MagicMock()
        config.exit_type_weights = {"full": 1.0, "partial": 0.0, "trailing": 0.0}

        gene = create_random_exit_gene(config)
        assert gene.exit_type == ExitType.FULL

    def test_create_random_exit_gene_fallback(self):
        config = MagicMock()
        config.exit_type_weights = {"invalid_type": 1.0}

        gene = create_random_exit_gene(config)
        assert gene.exit_type == ExitType.FULL
        assert gene.partial_exit_pct == 0.5

"""
指標パラメータ制約の修復組み合わせテスト

GAのランダム生成・変異・交叉はパラメータを独立に操作するため、
MACD の fast >= slow のような依存関係違反が生じ得る。本テストは

1. discovery が fast/slow 系指標に制約を登録すること
2. StrategyGene.repair_indicator_parameters が違反を修復すること
3. 変異・交叉パイプラインが修復を通して常に有効な個体を生成すること

を検証します。
"""

import pytest

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.genes import IndicatorGene, StrategyGene
from app.services.auto_strategy.genes.operators.crossover import (
    crossover_strategy_genes,
)
from app.services.auto_strategy.genes.operators.mutation import (
    _is_integer_parameter,
    mutate_strategy_gene,
)
from app.services.indicators.config.discovery import DynamicIndicatorDiscovery
from app.services.indicators.config.indicator_config import IndicatorConfig


def _make_gene(indicator: IndicatorGene) -> StrategyGene:
    return StrategyGene(
        id="gene-parameter-repair",
        indicators=[indicator],
        long_entry_conditions=[],
        short_entry_conditions=[],
        stateful_conditions=[],
        risk_management={"position_size": 0.1},
        metadata={},
    )


def _macd_gene(fast: int, slow: int, signal: int = 9) -> IndicatorGene:
    return IndicatorGene(
        type="MACD", parameters={"fast": fast, "slow": slow, "signal": signal}
    )


class TestDiscoveryConstraintRegistration:
    """discovery のパラメータ依存関係制約登録のテスト"""

    @pytest.mark.parametrize(
        ("indicator_name", "expected_count"),
        [
            ("MACD", 1),
            ("PPO", 1),
            ("UO", 2),  # fast<medium<slow の連鎖
        ],
    )
    def test_apply_special_overrides_registers_constraints(
        self, indicator_name: str, expected_count: int
    ):
        """fast/slow 系指標に制約が付与される"""
        config = IndicatorConfig(indicator_name=indicator_name)
        DynamicIndicatorDiscovery._apply_special_overrides(config)

        assert config.parameter_constraints is not None
        assert len(config.parameter_constraints) == expected_count
        assert all(c["type"] == "less_than" for c in config.parameter_constraints)

    def test_apply_special_overrides_no_constraints_for_simple_indicator(self):
        """fast/slow を持たない指標には制約が付かない"""
        config = IndicatorConfig(indicator_name="RSI")
        DynamicIndicatorDiscovery._apply_special_overrides(config)

        assert not config.parameter_constraints

    def test_registry_macd_generation_always_satisfies_constraints(self):
        """実レジストリの MACD ランダム生成が常に fast < slow を満たす"""
        from app.services.indicators.config import indicator_registry

        indicator_registry.ensure_initialized()
        config = indicator_registry.get_indicator_config("MACD")
        assert config is not None
        assert config.parameter_constraints

        for _ in range(100):
            params = config.generate_random_parameters()
            is_valid, _ = config.validate_constraints(params)
            assert is_valid, f"違反: {params}"


class TestStrategyGeneParameterRepair:
    """StrategyGene.repair_indicator_parameters のテスト"""

    def test_repairs_macd_fast_slow_violation(self):
        """fast >= slow の MACD がスワップで修復される"""
        gene = _make_gene(_macd_gene(fast=50, slow=10))

        repaired = gene.repair_indicator_parameters()

        params = gene.indicators[0].parameters
        assert params["fast"] < params["slow"]
        assert repaired == 1

    def test_repairs_uo_chain_violation(self):
        """UO の連鎖制約（fast<medium<slow）違反が修復される"""
        gene = _make_gene(
            IndicatorGene(type="UO", parameters={"fast": 60, "medium": 10, "slow": 5})
        )

        gene.repair_indicator_parameters()

        params = gene.indicators[0].parameters
        assert params["fast"] < params["medium"] < params["slow"]

    def test_no_repair_for_valid_parameters(self):
        """有効なパラメータは変更されない"""
        gene = _make_gene(_macd_gene(fast=12, slow=26))

        repaired = gene.repair_indicator_parameters()

        assert gene.indicators[0].parameters == {"fast": 12, "slow": 26, "signal": 9}
        assert repaired == 0

    def test_unknown_indicator_type_does_not_crash(self):
        """未知の指標タイプはスキップされクラッシュしない"""
        gene = _make_gene(
            IndicatorGene(type="UNKNOWN_XYZ", parameters={"fast": 50, "slow": 10})
        )

        repaired = gene.repair_indicator_parameters()

        assert repaired == 0
        assert gene.indicators[0].parameters == {"fast": 50, "slow": 10}

    def test_registry_error_does_not_crash(self, monkeypatch: pytest.MonkeyPatch):
        """レジストリ照会で例外が発生してもクラッシュしない"""
        from app.services.indicators.config import indicator_registry

        def _raise(_name: str) -> IndicatorConfig:
            raise RuntimeError("registry broken")

        monkeypatch.setattr(indicator_registry, "get_indicator_config", _raise)
        gene = _make_gene(_macd_gene(fast=50, slow=10))

        assert gene.repair_indicator_parameters() == 0


class TestMutationIntegerParameterDetection:
    """変異の整数パラメータ判定（レジストリ由来）のテスト"""

    def test_macd_fast_is_integer_parameter(self):
        """実パラメータ名 fast/slow が整数扱いされる（旧リストは fast_period のみ）"""
        from app.services.indicators.config import indicator_registry

        indicator_registry.ensure_initialized()

        assert _is_integer_parameter("MACD", "fast") is True
        assert _is_integer_parameter("MACD", "slow") is True

    def test_float_param_is_not_integer_parameter(self):
        """scalar 等の実数パラメータや未知の名前は整数扱いされない"""
        from app.services.indicators.config import indicator_registry

        indicator_registry.ensure_initialized()

        assert _is_integer_parameter("PVO", "scalar") is False
        assert _is_integer_parameter("MACD", "unknown_name") is False

    def test_unknown_indicator_falls_back_to_legacy_names(self):
        """未知指標はフォールバックリストで判定される"""
        assert _is_integer_parameter("UNKNOWN_XYZ", "period") is True
        assert _is_integer_parameter("UNKNOWN_XYZ", "scalar") is False


class TestOperatorPipelineConstraint:
    """変異・交叉パイプライン全体の制約保持テスト"""

    def test_mutation_keeps_fast_below_slow(self):
        """高変異レートでも fast < slow と整数性が保たれる"""
        config = GAConfig()
        config.mutation_config.indicator_param_range = (3.0, 6.0)

        checked = 0
        for _ in range(30):
            mutated = mutate_strategy_gene(
                _make_gene(_macd_gene(fast=12, slow=26)), config, mutation_rate=1.0
            )
            for indicator in mutated.indicators:
                if indicator.type != "MACD":
                    continue
                params = indicator.parameters
                assert params["fast"] < params["slow"], f"違反: {params}"
                assert all(isinstance(v, int) for v in params.values()), (
                    f"小数化: {params}"
                )
                checked += 1
        assert checked > 0

    def test_mutation_coerces_fractional_params(self):
        """実数化したパラメータ（12.5 等）でも変異後に整数へ戻る"""
        config = GAConfig()
        config.mutation_config.indicator_param_range = (3.0, 6.0)

        fractional = IndicatorGene(
            type="MACD", parameters={"fast": 12.5, "slow": 26.7, "signal": 9.3}
        )

        checked = 0
        for _ in range(30):
            mutated = mutate_strategy_gene(
                _make_gene(fractional), config, mutation_rate=1.0
            )
            for indicator in mutated.indicators:
                if indicator.type != "MACD":
                    continue
                assert all(isinstance(v, int) for v in indicator.parameters.values()), (
                    f"小数化: {indicator.parameters}"
                )
                checked += 1
        assert checked > 0

    def test_crossover_children_repaired(self):
        """交叉の子は親由来の制約違反が修復されている"""
        from app.services.indicators.config import indicator_registry

        indicator_registry.ensure_initialized()
        config = GAConfig()

        parent1 = _make_gene(_macd_gene(fast=50, slow=10))
        parent2 = _make_gene(
            IndicatorGene(type="UO", parameters={"fast": 60, "medium": 10, "slow": 5})
        )

        child1, child2 = crossover_strategy_genes(
            StrategyGene, parent1, parent2, config
        )

        for child in (child1, child2):
            for indicator in child.indicators:
                params = indicator.parameters
                if indicator.type == "MACD":
                    assert params["fast"] < params["slow"], f"違反: {params}"
                elif indicator.type == "UO":
                    assert params["fast"] < params["medium"] < params["slow"], (
                        f"違反: {params}"
                    )

"""パラメータ摂動頑健性ゲートのテスト"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.config.ga_config import (
    EvaluationConfig,
    RobustnessConfig,
    ValidationConfig,
)
from app.services.auto_strategy.core.evaluation.evaluation_report import (
    EvaluationReport,
    ScenarioEvaluation,
)
from app.services.auto_strategy.services.parameter_perturbation import (
    build_perturbed_genes,
)
from app.services.auto_strategy.services.strategy_validation_service import (
    StrategyValidationService,
)


def _make_gene():
    """摂動テスト用の遺伝子（生きた指標2つ＋死んだ指標1つ）"""
    return SimpleNamespace(
        indicators=[
            SimpleNamespace(
                type="BBANDS",
                parameters={"length": 20},
                enabled=True,
                timeframe=None,
                id="aaaa1111",
            ),
            SimpleNamespace(
                type="RSI",
                parameters={"period": 14},
                enabled=True,
                timeframe=None,
                id="bbbb2222",
            ),
            # 条件式から参照されない死んだ遺伝子
            SimpleNamespace(
                type="SQUEEZE",
                parameters={"lookback": 50},
                enabled=True,
                timeframe=None,
                id="cccc3333",
            ),
        ],
        long_entry_conditions=[
            {
                "direction": "long",
                "left_operand": "close",
                "operator": ">",
                "right_operand": {"type": "indicator", "name": "BBANDS_aaaa1111_2"},
            }
        ],
        short_entry_conditions=[
            {
                "direction": "short",
                "left_operand": "close",
                "operator": "<",
                "right_operand": {"type": "indicator", "name": "RSI_bbbb2222_0"},
            }
        ],
        long_exit_conditions=[],
        short_exit_conditions=[],
        risk_management={},
        tpsl_gene=None,
        position_sizing_gene=None,
        entry_gene=None,
        exit_gene=None,
        tool_genes=[],
    )


class TestBuildPerturbedGenes:
    """変異体生成ロジックのテスト"""

    def test_one_at_a_time_int_perturbation(self):
        """intパラメータが±10%ずつ個別に揺らされる"""
        labels = [label for label, _ in build_perturbed_genes(_make_gene(), 0.10, 10)]
        assert any("BBANDS.length: 20 -> 22" in x for x in labels)
        assert any("BBANDS.length: 20 -> 18" in x for x in labels)
        assert any("RSI.period: 14 -> 15" in x for x in labels)
        assert any("RSI.period: 14 -> 13" in x for x in labels)

    def test_dead_indicator_is_not_perturbed(self):
        """条件から参照されない指標は変異対象にならない"""
        labels = [label for label, _ in build_perturbed_genes(_make_gene(), 0.10, 10)]
        assert not any("SQUEEZE" in x for x in labels)

    def test_max_variants_caps_generation(self):
        """max_variants で生成数が打ち止めになる"""
        variants = build_perturbed_genes(_make_gene(), 0.10, 2)
        assert len(variants) == 2

    def test_no_numeric_params_returns_empty(self):
        """数値パラメータが無い場合は空リスト"""
        gene = _make_gene()
        gene.indicators = [
            SimpleNamespace(
                type="X",
                parameters={"mode": "fast"},
                enabled=True,
                timeframe=None,
                id="dddd4444",
            )
        ]
        gene.long_entry_conditions = [{"type": "indicator", "name": "X_dddd444"}]
        gene.short_entry_conditions = []
        assert build_perturbed_genes(gene, 0.10, 6) == []

    def test_bool_params_are_skipped(self):
        """bool は数値として扱わない"""
        gene = _make_gene()
        gene.indicators[0].parameters = {"length": 20, "use_mid": True}
        labels = [label for label, _ in build_perturbed_genes(gene, 0.10, 20)]
        assert not any("use_mid" in x for x in labels)

    def test_float_param_perturbation(self):
        """float パラメータも比率で揺らされる"""
        gene = _make_gene()
        gene.indicators[0].parameters = {"std": 2.0}
        gene.short_entry_conditions = []
        labels = [label for label, _ in build_perturbed_genes(gene, 0.10, 10)]
        assert any("std: 2.0 -> 2.2" in x for x in labels)
        assert any("std: 2.0 -> 1.8" in x for x in labels)


@pytest.fixture
def service():
    return StrategyValidationService(MagicMock())


def _base_args():
    config = GAConfig(
        validation_config=ValidationConfig(),
        evaluation_config=EvaluationConfig(evaluation_mode="walk_forward"),
        objectives=["total_return"],
    )
    robustness_config = RobustnessConfig(
        param_perturbation_enabled=True,
        param_delta=0.10,
        param_max_variants=4,
        param_min_survival_rate=0.5,
        param_min_retention_ratio=0.5,
    )
    return config, {}, robustness_config


def _passed_validation():
    return {"passed": True, "reasons": []}


class TestParameterPerturbationGate:
    """摂動ゲート判定のテスト"""

    def test_all_variants_survive_keeps_passed(self, service):
        """全変異体が耐久なら合格を維持する"""
        ga_config, bt_config, rconfig = _base_args()

        def fake_original(*args, **kwargs):
            return {"total_return": 0.10, "buy_hold_return": 0.02}

        def fake_variant(variant, bt, cfg):
            return {"total_return": 0.08, "buy_hold_return": 0.03}

        service._get_full_period_metrics = fake_original
        service._evaluate_variant_metrics = fake_variant

        validation = service._apply_parameter_perturbation_gate(
            _passed_validation(), _make_gene(), bt_config, ga_config, rconfig
        )

        assert validation["passed"] is True
        result = validation["param_perturbation"]
        assert result["variant_count"] == 4
        assert result["survived"] == 4
        assert result["survival_rate"] == pytest.approx(1.0)

    def test_majority_collapse_fails_closed(self, service):
        """過半数の変異体が崩壊すれば不合格"""
        ga_config, bt_config, rconfig = _base_args()

        service._get_full_period_metrics = lambda *a, **k: {
            "total_return": 0.10,
            "buy_hold_return": 0.02,
        }
        # 元市場比 +8%、retention 0.5 → 変異体は市場比 +4% 以上が必要
        responses = [
            {"total_return": 0.09, "buy_hold_return": 0.03},  # excess +6% 耐久
            {"total_return": -0.05, "buy_hold_return": 0.03},  # 赤字 不耐
            {"total_return": 0.05, "buy_hold_return": 0.03},  # excess +2% 不耐
            {"total_return": 0.07, "buy_hold_return": 0.05},  # excess +2% 不耐
        ]
        iterator = iter(responses)
        service._evaluate_variant_metrics = lambda *a, **k: next(iterator)

        validation = service._apply_parameter_perturbation_gate(
            _passed_validation(), _make_gene(), bt_config, ga_config, rconfig
        )

        assert validation["passed"] is False
        assert any(
            "パラメータ摂動への耐性不足" in reason for reason in validation["reasons"]
        )
        result = validation["param_perturbation"]
        assert result["survived"] == 1
        assert result["survival_rate"] == pytest.approx(0.25)

    def test_variant_evaluation_failure_counts_as_non_survivor(self, service):
        """変異体評価の失敗は不耐扱い（fail-closed）"""
        ga_config, bt_config, rconfig = _base_args()

        service._get_full_period_metrics = lambda *a, **k: {
            "total_return": 0.10,
            "buy_hold_return": 0.02,
        }

        def broken_variant(variant, bt, cfg):
            raise RuntimeError("boom")

        service._evaluate_variant_metrics = broken_variant

        validation = service._apply_parameter_perturbation_gate(
            _passed_validation(), _make_gene(), bt_config, ga_config, rconfig
        )

        assert validation["passed"] is False
        result = validation["param_perturbation"]
        assert result["survived"] == 0

    def test_no_variants_passes_through(self, service):
        """変異体が生成できない場合は判定なしで通過"""
        ga_config, bt_config, rconfig = _base_args()
        gene = _make_gene()
        for ind in gene.indicators:
            ind.parameters = {}
        gene.long_entry_conditions = []
        gene.short_entry_conditions = []

        service._get_full_period_metrics = lambda *a, **k: {
            "total_return": 0.10,
            "buy_hold_return": 0.02,
        }

        validation = service._apply_parameter_perturbation_gate(
            _passed_validation(), gene, bt_config, ga_config, rconfig
        )

        assert validation["passed"] is True
        assert validation["param_perturbation"]["variant_count"] == 0

    def test_missing_original_metrics_skips_judgement(self, service):
        """元戦略の連続指標が無い場合は判定をスキップする"""
        ga_config, bt_config, rconfig = _base_args()

        service._get_full_period_metrics = lambda *a, **k: None
        service._evaluate_variant_metrics = lambda *a, **k: {
            "total_return": -1.0,
            "buy_hold_return": 0.0,
        }

        validation = service._apply_parameter_perturbation_gate(
            _passed_validation(), _make_gene(), bt_config, ga_config, rconfig
        )

        assert validation["passed"] is True
        result = validation["param_perturbation"]
        assert result["variant_count"] == 4
        assert result["survived"] is None


class TestValidationIntegration:
    """_validate_strategy 経由の組み込み確認"""

    def test_validate_strategy_applies_perturbation_gate_on_pass(self):
        """WFA合格時に摂動ゲートが実行される"""
        evaluator = MagicMock()
        service = StrategyValidationService(evaluator)

        report = EvaluationReport(
            mode="walk_forward",
            objectives=("total_return",),
            aggregated_fitness=(0.5,),
            scenarios=[
                ScenarioEvaluation(
                    name=f"fold_{i}",
                    fitness=(0.5,),
                    passed=True,
                    performance_metrics={
                        "total_trades": 40.0,
                        "max_drawdown": 0.05,
                        "total_return": 0.1,
                        "excess_return": 0.05,
                        "buy_hold_return": 0.05,
                    },
                )
                for i in range(3)
            ],
            aggregate_method="robust",
            metadata={},
        )

        service._evaluation_strategy.execute_report = MagicMock(return_value=report)
        service._evaluation_strategy.execute_single_report = MagicMock(
            side_effect=RuntimeError("no single expected here")
        )
        service._get_full_period_metrics = MagicMock(
            return_value={"total_return": 0.10, "buy_hold_return": 0.02}
        )
        service._evaluate_variant_metrics = MagicMock(
            return_value={"total_return": 0.08, "buy_hold_return": 0.03}
        )

        ga_config = GAConfig(
            validation_config=ValidationConfig(),
            evaluation_config=EvaluationConfig(evaluation_mode="walk_forward"),
            objectives=["total_return"],
        )
        strategy = _make_gene()

        validation = service._validate_strategy(
            strategy,
            {},
            ga_config,
            ga_config.validation_config,
            ga_config.robustness_config,
        )

        assert validation.get("passed") is True
        assert "param_perturbation" in validation
        assert validation["param_perturbation"]["variant_count"] > 0

"""
ParameterTuningManager の拡張テスト

既存テスト (``test_parameter_tuning_manager.py``) がカバーしていない
``tune_candidate_genes``, ``select_best_tuned_candidate`` (robustness path),
``tune_and_select_best_gene``, ``refresh_best_gene_reporting``,
``build_individual_evaluation_summary`` を検証します。
"""

from __future__ import annotations

from unittest.mock import Mock, patch

from app.services.auto_strategy.config.ga.ga_config import GAConfig
from app.services.auto_strategy.core.engine.parameter_tuning_manager import (
    ParameterTuningManager,
)
from app.services.auto_strategy.genes import StrategyGene


def _make_gene(gene_id: str = "g1") -> StrategyGene:
    gene = StrategyGene.create_default()
    gene.id = gene_id
    gene.metadata = {}
    return gene


class TestTuneCandidateGenes:
    """``tune_candidate_genes`` のテスト"""

    def test_tunes_each_candidate_and_records_rank(self) -> None:
        evaluator = Mock()
        manager = ParameterTuningManager(individual_evaluator=evaluator)
        c1 = _make_gene("c1")
        c2 = _make_gene("c2")
        tuned1 = _make_gene("tuned1")
        tuned2 = _make_gene("tuned2")

        with patch(
            "app.services.auto_strategy.optimization.StrategyParameterTuner"
        ) as MockTuner:
            MockTuner.from_ga_config.return_value.tune.side_effect = [tuned1, tuned2]
            result = manager.tune_candidate_genes([c1, c2], GAConfig())

        assert len(result) == 2
        assert result[0] is tuned1
        assert result[1] is tuned2
        # metadata に rank が記録される
        assert tuned1.metadata["tuning_candidate_rank"] == 0
        assert tuned2.metadata["tuning_candidate_rank"] == 1

    def test_falls_back_to_original_when_tune_fails(self) -> None:
        evaluator = Mock()
        manager = ParameterTuningManager(individual_evaluator=evaluator)
        c1 = _make_gene("c1")
        c2 = _make_gene("c2")
        tuned2 = _make_gene("tuned2")

        with patch(
            "app.services.auto_strategy.optimization.StrategyParameterTuner"
        ) as MockTuner:
            MockTuner.from_ga_config.return_value.tune.side_effect = [
                RuntimeError("tune fail"),
                tuned2,
            ]
            result = manager.tune_candidate_genes([c1, c2], GAConfig())

        # 1 番目は元の遺伝子がそのまま
        assert result[0] is c1
        assert result[0].metadata["tuning_candidate_rank"] == 0
        assert result[1] is tuned2
        assert result[1].metadata["tuning_candidate_rank"] == 1

    def test_setdefault_preserves_existing_rank_on_result(self) -> None:
        """結果遺伝子の metadata に既に rank がある場合、setdefault で上書きしない"""
        evaluator = Mock()
        manager = ParameterTuningManager(individual_evaluator=evaluator)
        c1 = _make_gene("c1")
        # tuner.tune が既存 rank 付きの結果を返すケース
        tuned1 = _make_gene("tuned1")
        tuned1.metadata = {"tuning_candidate_rank": 99, "custom": "value"}

        with patch(
            "app.services.auto_strategy.optimization.StrategyParameterTuner"
        ) as MockTuner:
            MockTuner.from_ga_config.return_value.tune.return_value = tuned1
            result = manager.tune_candidate_genes([c1], GAConfig())

        # setdefault なので既存 rank (99) は保持され、custom も残る
        assert result[0].metadata["tuning_candidate_rank"] == 99
        assert result[0].metadata["custom"] == "value"


class TestSelectBestTunedCandidate:
    """``select_best_tuned_candidate`` (robustness path) のテスト"""

    def test_returns_none_for_empty_candidates(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        assert manager.select_best_tuned_candidate([], GAConfig()) is None

    def test_picks_candidate_with_highest_rank_key(self) -> None:
        evaluator = Mock()
        evaluator.evaluate.return_value = (0.5,)
        # robustness report を Mock で提供
        report_high = Mock()
        report_high.metadata = {"evaluation_fidelity": "full"}
        report_low = Mock()
        report_low.metadata = {"evaluation_fidelity": "full"}
        evaluator.evaluate_robustness_report = Mock(
            side_effect=[report_high, report_low]
        )
        evaluator.get_cached_evaluation_report = Mock(side_effect=[None, None])

        manager = ParameterTuningManager(individual_evaluator=evaluator)
        c1 = _make_gene("c1")
        c2 = _make_gene("c2")

        # build_rank_key で異なる値を返すようモック
        with patch.object(
            manager, "build_individual_evaluation_summary", return_value={}
        ):
            with patch(
                "app.services.auto_strategy.core.engine.parameter_tuning_manager.build_report_rank_key_from_primary_fitness",
                side_effect=[(1.0, 0.5), (0.5, 0.3)],
            ):
                result = manager.select_best_tuned_candidate([c1, c2], GAConfig())

        assert result is not None
        assert result[0] is c1
        assert result[1] == 0.5

    def test_falls_back_to_cached_evaluation_report(self) -> None:
        """robustness report が None のとき、cached evaluation report を使う"""
        evaluator = Mock()
        evaluator.evaluate.return_value = (0.5,)
        evaluator.evaluate_robustness_report = Mock(return_value=None)
        # get_cached_evaluation_report は最初の is_evaluation_report() で False を返す
        # → 二度目で Mock を返すよう side_effect 配列
        report = Mock()
        report.metadata = {"evaluation_fidelity": "full"}
        evaluator.get_cached_evaluation_report = Mock(return_value=report)

        manager = ParameterTuningManager(individual_evaluator=evaluator)
        c1 = _make_gene("c1")

        with patch.object(
            manager, "build_individual_evaluation_summary", return_value={}
        ):
            with patch(
                "app.services.auto_strategy.core.engine.parameter_tuning_manager.build_report_rank_key_from_primary_fitness",
                return_value=(0.5,),
            ):
                with patch(
                    "app.services.auto_strategy.core.engine.parameter_tuning_manager.is_evaluation_report",
                    side_effect=[False, True],
                ):
                    result = manager.select_best_tuned_candidate([c1], GAConfig())

        assert result is not None
        assert result[0] is c1

    def test_continues_when_evaluation_fails(self) -> None:
        evaluator = Mock()
        call_count = [0]

        def mock_evaluate(ind, config, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Eval fail")
            return (0.7,)

        evaluator.evaluate.side_effect = mock_evaluate
        manager = ParameterTuningManager(individual_evaluator=evaluator)

        c1 = _make_gene("c1")
        c2 = _make_gene("c2")

        with patch.object(
            manager, "build_individual_evaluation_summary", return_value={}
        ):
            with patch(
                "app.services.auto_strategy.core.engine.parameter_tuning_manager.build_report_rank_key_from_primary_fitness",
                return_value=(0.5,),
            ):
                result = manager.select_best_tuned_candidate([c1, c2], GAConfig())

        assert result is not None
        assert result[0] is c2

    def test_continues_when_robustness_report_fails(self) -> None:
        evaluator = Mock()
        evaluator.evaluate.return_value = (0.5,)
        evaluator.evaluate_robustness_report = Mock(
            side_effect=RuntimeError("Robustness fail")
        )
        report = Mock()
        report.metadata = {"evaluation_fidelity": "full"}
        evaluator.get_cached_evaluation_report = Mock(return_value=report)

        manager = ParameterTuningManager(individual_evaluator=evaluator)
        c1 = _make_gene("c1")

        with patch.object(
            manager, "build_individual_evaluation_summary", return_value={}
        ):
            with patch(
                "app.services.auto_strategy.core.engine.parameter_tuning_manager.build_report_rank_key_from_primary_fitness",
                return_value=(0.5,),
            ):
                with patch(
                    "app.services.auto_strategy.core.engine.parameter_tuning_manager.is_evaluation_report",
                    return_value=True,
                ):
                    result = manager.select_best_tuned_candidate([c1], GAConfig())

        assert result is not None


class TestTuneAndSelectBestGene:
    """``tune_and_select_best_gene`` のテスト"""

    def test_returns_current_when_no_best_gene(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        result = manager.tune_and_select_best_gene(
            population=[],
            current_best_gene=None,
            config=GAConfig(),
            fallback_fitness=0.0,
            fallback_summary=None,
        )
        assert result == (None, 0.0, None)

    def test_uses_elite_tuning_for_multi_objective(self) -> None:
        """objectives が複数あれば tune_elite_parameters を呼ぶ"""
        manager = ParameterTuningManager(individual_evaluator=Mock())
        best = _make_gene("best")
        tuned = _make_gene("tuned")
        config = GAConfig()
        config.objectives = (Mock(), Mock())  # 2 要素

        with patch.object(
            manager, "tune_elite_parameters", return_value=tuned
        ) as mock_tune:
            with patch.object(
                manager,
                "refresh_best_gene_reporting",
                return_value=(0.5, {"s": "x"}),
            ):
                result = manager.tune_and_select_best_gene(
                    population=[],
                    current_best_gene=best,
                    config=config,
                    fallback_fitness=0.0,
                    fallback_summary=None,
                )

        assert result[0] is tuned
        mock_tune.assert_called_once_with(best, config)

    def test_returns_current_when_no_tuning_candidates(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        best = _make_gene("best")
        config = GAConfig()

        with patch.object(manager, "select_tuning_candidates", return_value=[]):
            with patch.object(
                manager,
                "refresh_best_gene_reporting",
                return_value=(0.0, None),
            ):
                result = manager.tune_and_select_best_gene(
                    population=[],
                    current_best_gene=best,
                    config=config,
                    fallback_fitness=0.0,
                    fallback_summary=None,
                )

        assert result[0] is best

    def test_uses_two_stage_selection_when_enabled(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        best = _make_gene("best")
        config = GAConfig()
        config.two_stage_selection_config.enabled = True

        winner = (best, 0.7, {"s": "x"})

        with patch.object(manager, "select_tuning_candidates", return_value=[best]):
            with patch.object(manager, "tune_candidate_genes", return_value=[best]):
                with patch.object(
                    manager, "select_best_tuned_candidate", return_value=winner
                ) as mock_sel:
                    with patch.object(
                        manager, "select_best_tuned_candidate_by_fitness"
                    ) as mock_single:
                        result = manager.tune_and_select_best_gene(
                            population=[],
                            current_best_gene=best,
                            config=config,
                            fallback_fitness=0.0,
                            fallback_summary=None,
                        )

        assert result[0] is best
        mock_sel.assert_called_once()
        mock_single.assert_not_called()

    def test_uses_single_fitness_selection_when_two_stage_disabled(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        best = _make_gene("best")
        config = GAConfig()
        config.two_stage_selection_config.enabled = False
        winner = (best, 0.7, {"s": "x"})

        with patch.object(manager, "select_tuning_candidates", return_value=[best]):
            with patch.object(manager, "tune_candidate_genes", return_value=[best]):
                with patch.object(
                    manager,
                    "select_best_tuned_candidate_by_fitness",
                    return_value=winner,
                ) as mock_single:
                    with patch.object(
                        manager, "select_best_tuned_candidate"
                    ) as mock_two_stage:
                        result = manager.tune_and_select_best_gene(
                            population=[],
                            current_best_gene=best,
                            config=config,
                            fallback_fitness=0.0,
                            fallback_summary=None,
                        )

        assert result[0] is best
        mock_single.assert_called_once()
        mock_two_stage.assert_not_called()

    def test_returns_refreshed_when_tune_winner_is_none(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        best = _make_gene("best")
        config = GAConfig()
        config.two_stage_selection_config.enabled = True

        with patch.object(manager, "select_tuning_candidates", return_value=[best]):
            with patch.object(manager, "tune_candidate_genes", return_value=[best]):
                with patch.object(
                    manager, "select_best_tuned_candidate", return_value=None
                ):
                    with patch.object(
                        manager,
                        "refresh_best_gene_reporting",
                        return_value=(0.0, None),
                    ) as mock_refresh:
                        result = manager.tune_and_select_best_gene(
                            population=[],
                            current_best_gene=best,
                            config=config,
                            fallback_fitness=0.0,
                            fallback_summary=None,
                        )

        assert result[0] is best
        mock_refresh.assert_called_once()


class TestRefreshBestGeneReporting:
    """``refresh_best_gene_reporting`` のテスト"""

    def test_returns_fallback_when_no_best_gene(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        result = manager.refresh_best_gene_reporting(
            best_gene=None,
            config=GAConfig(),
            fallback_fitness=0.5,
            fallback_summary={"old": True},
        )
        assert result == (0.5, {"old": True})

    def test_returns_normalized_fitness_on_success(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        best = _make_gene("best")
        with patch.object(
            manager,
            "evaluate_individual_with_full_fidelity",
            return_value=(0.6, 0.4),
        ):
            with patch.object(
                manager,
                "build_individual_evaluation_summary",
                return_value={"new": True},
            ) as mock_summary:
                fitness, summary = manager.refresh_best_gene_reporting(
                    best_gene=best,
                    config=GAConfig(),
                    fallback_fitness=0.0,
                    fallback_summary=None,
                )

        # tuple は normalized される
        assert fitness == (0.6, 0.4)
        assert summary == {"new": True}
        # summary 構築に best_gene と config, force_robustness を使う
        mock_summary.assert_called_once()

    def test_returns_fallback_fitness_on_exception(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        best = _make_gene("best")
        with patch.object(
            manager,
            "evaluate_individual_with_full_fidelity",
            side_effect=RuntimeError("eval fail"),
        ):
            with patch.object(
                manager,
                "build_individual_evaluation_summary",
                return_value={"x": "y"},
            ):
                fitness, summary = manager.refresh_best_gene_reporting(
                    best_gene=best,
                    config=GAConfig(),
                    fallback_fitness=(0.1, 0.2),
                    fallback_summary=None,
                )

        # 例外時は fallback_fitness がそのまま返る
        assert fitness == (0.1, 0.2)
        assert summary == {"x": "y"}

    def test_falls_back_to_fallback_summary_when_summary_none(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        best = _make_gene("best")
        with patch.object(
            manager,
            "evaluate_individual_with_full_fidelity",
            return_value=(0.5,),
        ):
            with patch.object(
                manager,
                "build_individual_evaluation_summary",
                return_value=None,
            ):
                fitness, summary = manager.refresh_best_gene_reporting(
                    best_gene=best,
                    config=GAConfig(),
                    fallback_fitness=0.0,
                    fallback_summary={"legacy": True},
                )

        assert summary == {"legacy": True}


class TestBuildIndividualEvaluationSummary:
    """``build_individual_evaluation_summary`` のテスト"""

    def test_returns_none_for_none_individual(self) -> None:
        manager = ParameterTuningManager(individual_evaluator=Mock())
        assert manager.build_individual_evaluation_summary(None, GAConfig()) is None

    def test_uses_cached_robustness_report(self) -> None:
        evaluator = Mock()
        manager = ParameterTuningManager(individual_evaluator=evaluator)
        ind = _make_gene("ind")
        report = Mock()
        report.metadata = {"evaluation_fidelity": "full"}
        evaluator.get_cached_robustness_report = Mock(return_value=report)
        evaluator.get_cached_evaluation_report = Mock(return_value=None)

        with patch(
            "app.services.auto_strategy.core.engine.parameter_tuning_manager.is_evaluation_report",
            return_value=True,
        ):
            with patch(
                "app.services.auto_strategy.core.engine.report_selection.extract_primary_fitness",
                return_value=0.5,
            ):
                with patch(
                    "app.services.auto_strategy.core.evaluation.report_persistence.build_report_summary",
                    return_value={"summary": "ok"},
                ) as mock_build:
                    result = manager.build_individual_evaluation_summary(
                        ind, GAConfig()
                    )

        assert result == {"summary": "ok"}
        mock_build.assert_called_once()

    def test_falls_back_to_evaluation_report(self) -> None:
        evaluator = Mock()
        manager = ParameterTuningManager(individual_evaluator=evaluator)
        ind = _make_gene("ind")
        report = Mock()
        report.metadata = {"evaluation_fidelity": "full"}
        evaluator.get_cached_robustness_report = Mock(return_value=None)
        evaluator.get_cached_evaluation_report = Mock(return_value=report)
        evaluator.evaluate_robustness_report = Mock(return_value=None)

        with patch(
            "app.services.auto_strategy.core.engine.parameter_tuning_manager.is_multi_fidelity_enabled",
            return_value=False,
        ):
            with patch(
                "app.services.auto_strategy.core.engine.parameter_tuning_manager.is_evaluation_report",
                return_value=True,
            ):
                with patch(
                    "app.services.auto_strategy.core.evaluation.report_persistence.build_report_summary",
                    return_value={"s": "x"},
                ):
                    result = manager.build_individual_evaluation_summary(
                        ind, GAConfig()
                    )

        assert result == {"s": "x"}

    def test_returns_none_when_no_report_found(self) -> None:
        evaluator = Mock()
        manager = ParameterTuningManager(individual_evaluator=evaluator)
        ind = _make_gene("ind")
        evaluator.get_cached_robustness_report = Mock(return_value=None)
        evaluator.evaluate_robustness_report = Mock(return_value=None)
        evaluator.get_cached_evaluation_report = Mock(return_value=None)

        with patch(
            "app.services.auto_strategy.core.engine.parameter_tuning_manager.is_multi_fidelity_enabled",
            return_value=False,
        ):
            result = manager.build_individual_evaluation_summary(ind, GAConfig())

        assert result is None

    def test_skips_coarse_fidelity_report(self) -> None:
        """report.metadata.evaluation_fidelity == 'coarse' は None にリセットされる"""
        evaluator = Mock()
        manager = ParameterTuningManager(individual_evaluator=evaluator)
        ind = _make_gene("ind")

        # 最初の get_cached_robustness_report は coarse の report を返す
        coarse_report = Mock()
        coarse_report.metadata = {"evaluation_fidelity": "coarse"}
        evaluator.get_cached_robustness_report = Mock(return_value=coarse_report)
        evaluator.evaluate_robustness_report = Mock(return_value=None)
        evaluator.get_cached_evaluation_report = Mock(return_value=None)

        with patch(
            "app.services.auto_strategy.core.engine.parameter_tuning_manager.is_multi_fidelity_enabled",
            return_value=False,
        ):
            result = manager.build_individual_evaluation_summary(ind, GAConfig())

        # coarse → None にリセットされ、以降の取得でも見つからず None
        assert result is None

    def test_returns_none_when_primary_fitness_not_finite(self) -> None:
        evaluator = Mock()
        manager = ParameterTuningManager(individual_evaluator=evaluator)
        ind = _make_gene("ind")
        report = Mock()
        report.metadata = {"evaluation_fidelity": "full"}
        evaluator.get_cached_robustness_report = Mock(return_value=report)
        evaluator.get_cached_evaluation_report = Mock(return_value=None)

        with patch(
            "app.services.auto_strategy.core.engine.parameter_tuning_manager.is_evaluation_report",
            return_value=True,
        ):
            with patch(
                "app.services.auto_strategy.core.engine.report_selection.extract_primary_fitness",
                return_value=float("inf"),
            ):
                with patch(
                    "app.services.auto_strategy.core.evaluation.report_persistence.build_report_summary",
                    return_value={"x": "y"},
                ) as mock_build:
                    result = manager.build_individual_evaluation_summary(
                        ind, GAConfig()
                    )

        # inf は isfinite=False → numeric_fitness=None
        assert result == {"x": "y"}
        call_kwargs = mock_build.call_args[1]
        assert call_kwargs["fitness_score"] is None

    def test_uses_explicit_primary_fitness(self) -> None:
        """primary_fitness 引数が extract 結果より優先される"""
        evaluator = Mock()
        manager = ParameterTuningManager(individual_evaluator=evaluator)
        ind = _make_gene("ind")
        report = Mock()
        report.metadata = {"evaluation_fidelity": "full"}
        evaluator.get_cached_robustness_report = Mock(return_value=report)
        evaluator.get_cached_evaluation_report = Mock(return_value=None)

        with patch(
            "app.services.auto_strategy.core.engine.parameter_tuning_manager.is_evaluation_report",
            return_value=True,
        ):
            with patch(
                "app.services.auto_strategy.core.evaluation.report_persistence.build_report_summary",
                return_value={"x": "y"},
            ) as mock_build:
                manager.build_individual_evaluation_summary(
                    ind,
                    GAConfig(),
                    primary_fitness=0.42,
                    selection_rank_override=3,
                    selection_score_override=(0.5, 0.3),
                )

        call_kwargs = mock_build.call_args[1]
        assert call_kwargs["fitness_score"] == 0.42
        assert call_kwargs["selection_rank"] == 3
        assert call_kwargs["selection_score"] == (0.5, 0.3)

    def test_calls_full_evaluation_in_multi_fidelity_when_no_report(self) -> None:
        """multi-fidelity 有効時、report 無しなら full eval を試みる"""
        evaluator = Mock()
        manager = ParameterTuningManager(individual_evaluator=evaluator)
        ind = _make_gene("ind")
        evaluator.get_cached_robustness_report = Mock(return_value=None)
        evaluator.evaluate_robustness_report = Mock(return_value=None)
        report = Mock()
        report.metadata = {"evaluation_fidelity": "full"}
        # 最初は None、full eval 後は report を返す
        evaluator.get_cached_evaluation_report = Mock(side_effect=[None, report])

        with patch(
            "app.services.auto_strategy.core.engine.parameter_tuning_manager.is_multi_fidelity_enabled",
            return_value=True,
        ):
            with patch(
                "app.services.auto_strategy.core.engine.parameter_tuning_manager.is_evaluation_report",
                return_value=True,
            ):
                with patch(
                    "app.services.auto_strategy.core.evaluation.report_persistence.build_report_summary",
                    return_value={"final": "x"},
                ):
                    result = manager.build_individual_evaluation_summary(
                        ind, GAConfig()
                    )

        assert result == {"final": "x"}

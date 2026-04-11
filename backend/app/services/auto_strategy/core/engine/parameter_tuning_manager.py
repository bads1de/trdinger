"""
パラメータチューニング管理モジュール

GA実行後のエリート個体のパラメータチューニングと最終選択を担当します。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, cast

from app.services.auto_strategy.genes import StrategyGene

from ..evaluation.evaluation_fidelity import is_multi_fidelity_enabled
from ..evaluation.evaluation_report import EvaluationReport
from .report_selection import build_report_rank_key_from_primary_fitness, is_evaluation_report

logger = logging.getLogger(__name__)


class ParameterTuningManager:
    """
    パラメータチューニング管理クラス

    エリート個体のパラメータをOptunaでチューニングし、
    設定に応じた基準で最終勝者を選択します。
    """

    def __init__(self, individual_evaluator: Any):
        """
        初期化

        Args:
            individual_evaluator: 個体評価器
        """
        self.individual_evaluator = individual_evaluator

    def tune_elite_parameters(
        self, best_gene: StrategyGene, config: Any
    ) -> StrategyGene:
        """
        エリート個体のパラメータをOptunaでチューニングします。

        Args:
            best_gene: 最良戦略遺伝子
            config: GA設定

        Returns:
            チューニングされた戦略遺伝子
        """
        try:
            from app.services.auto_strategy.optimization import StrategyParameterTuner

            logger.info("[Tuning] エリート個体のパラメータチューニングを開始")

            tuner = StrategyParameterTuner.from_ga_config(
                self.individual_evaluator,
                config,
            )

            tuned_gene = tuner.tune(best_gene)

            logger.info("[Done] パラメータチューニング完了")
            return tuned_gene

        except Exception as e:
            logger.warning(f"パラメータチューニング中にエラーが発生: {e}")
            # エラー時は元の遺伝子を返す
            return best_gene

    def select_tuning_candidates(
        self,
        population: List[Any],
        config: Any,
        *,
        fallback_gene: Optional[StrategyGene] = None,
    ) -> List[StrategyGene]:
        """
        チューニング対象の上位候補を抽出する。
        """
        tuning_config = getattr(config, "tuning_config", None)
        budget = getattr(tuning_config, "elite_count", 1)
        try:
            candidate_budget = max(1, int(budget))
        except (TypeError, ValueError):
            candidate_budget = 1

        # 集団をランク付け
        from .result_processor import ResultProcessor

        processor = ResultProcessor()
        ordered_population = processor.rank_population_for_persistence(population)

        candidates: List[StrategyGene] = []
        seen_keys = set()

        for individual in ordered_population:
            if not isinstance(individual, StrategyGene):
                continue
            identity = processor.get_strategy_result_key(individual)
            if identity in seen_keys:
                continue
            seen_keys.add(identity)
            candidates.append(individual)
            if len(candidates) >= candidate_budget:
                break

        if fallback_gene is not None and not candidates:
            candidates.append(fallback_gene)

        return candidates

    def tune_candidate_genes(
        self, candidates: List[StrategyGene], config: Any
    ) -> List[StrategyGene]:
        """
        候補遺伝子群を順次チューニングする。
        """
        from app.services.auto_strategy.optimization import StrategyParameterTuner

        tuner = StrategyParameterTuner.from_ga_config(
            self.individual_evaluator,
            config,
        )

        tuned_candidates: List[StrategyGene] = []
        for candidate_rank, candidate in enumerate(candidates):
            try:
                tuned_candidate = tuner.tune(candidate)
            except Exception as exc:
                logger.warning(
                    "[Tuning] 候補 %s のチューニングに失敗: %s", candidate_rank, exc
                )
                tuned_candidate = candidate
            tuned_candidate.metadata.setdefault("tuning_candidate_rank", candidate_rank)
            tuned_candidates.append(tuned_candidate)

        return tuned_candidates

    def select_best_tuned_candidate(
        self, tuned_candidates: List[StrategyGene], config: Any
    ) -> Optional[Tuple[StrategyGene, float, Optional[Dict[str, Any]]]]:
        """
        チューニング後候補を robustness で再評価し最終勝者を返す。
        """
        if not tuned_candidates:
            return None

        best_tuple: Optional[Tuple[StrategyGene, float, Optional[Dict[str, Any]]]] = (
            None
        )
        best_key = None

        for candidate_rank, candidate in enumerate(tuned_candidates):
            try:
                fitness_result = self.evaluate_individual_with_full_fidelity(
                    candidate, config
                )
            except Exception as exc:
                logger.warning(
                    "[Tuning] 候補 %s の再評価に失敗: %s",
                    candidate_rank,
                    exc,
                )
                continue

            primary_fitness = self.extract_primary_fitness_from_result(fitness_result)
            report = None
            evaluate_robustness_report = getattr(
                self.individual_evaluator,
                "evaluate_robustness_report",
                None,
            )
            if callable(evaluate_robustness_report):
                try:
                    report = evaluate_robustness_report(candidate, config)
                except Exception as exc:
                    logger.debug(
                        "[Tuning] 候補 %s の robustness 評価に失敗: %s",
                        candidate_rank,
                        exc,
                    )

            if not is_evaluation_report(report):
                get_cached_evaluation_report = getattr(
                    self.individual_evaluator,
                    "get_cached_evaluation_report",
                    None,
                )
                if callable(get_cached_evaluation_report):
                    report = get_cached_evaluation_report(candidate)

            rank_key = build_report_rank_key_from_primary_fitness(
                primary_fitness,
                cast(
                    Optional[EvaluationReport],
                    report if is_evaluation_report(report) else None,
                ),
                min_pass_rate=float(
                    getattr(
                        getattr(config, "two_stage_selection_config", None),
                        "min_pass_rate",
                        0.0,
                    )
                    or 0.0
                ),
            )
            summary = self.build_individual_evaluation_summary(
                candidate,
                config,
                force_robustness=False,
                primary_fitness=primary_fitness,
                selection_rank_override=candidate_rank,
                selection_score_override=rank_key,
            )
            candidate_result = (
                candidate,
                primary_fitness,
                summary,
            )
            if best_key is None or rank_key > best_key:
                best_key = rank_key
                best_tuple = candidate_result

        return best_tuple

    def select_best_tuned_candidate_by_fitness(
        self, tuned_candidates: List[StrategyGene], config: Any
    ) -> Optional[Tuple[StrategyGene, float, Optional[Dict[str, Any]]]]:
        """
        チューニング後候補を主 fitness だけで再選抜する。
        """
        if not tuned_candidates:
            return None

        best_tuple: Optional[Tuple[StrategyGene, float, Optional[Dict[str, Any]]]] = (
            None
        )
        best_fitness: Optional[float] = None

        for candidate in tuned_candidates:
            try:
                fitness_result = self.evaluate_individual_with_full_fidelity(
                    candidate, config
                )
            except Exception as exc:
                logger.warning("[Tuning] 候補の再評価に失敗: %s", exc)
                continue

            primary_fitness = self.extract_primary_fitness_from_result(fitness_result)
            summary = self.build_individual_evaluation_summary(
                candidate,
                config,
                force_robustness=False,
                primary_fitness=primary_fitness,
            )
            candidate_result = (
                candidate,
                primary_fitness,
                summary,
            )
            if best_fitness is None or primary_fitness > best_fitness:
                best_fitness = primary_fitness
                best_tuple = candidate_result

        return best_tuple

    def evaluate_individual_with_full_fidelity(
        self, individual: Any, config: Any
    ) -> Tuple[float, ...]:
        """
        必要に応じて full fidelity で個体を再評価する。
        """
        if is_multi_fidelity_enabled(config):
            return self.individual_evaluator.evaluate(
                individual,
                config,
                force_refresh=True,
            )
        return self.individual_evaluator.evaluate(individual, config)

    @staticmethod
    def extract_primary_fitness_from_result(result: Any) -> float:
        """
        評価結果から主 fitness を取り出す。
        """
        from .fitness_utils import extract_primary_fitness_from_result

        return extract_primary_fitness_from_result(result)

    def tune_and_select_best_gene(
        self,
        *,
        population: List[Any],
        current_best_gene: Optional[StrategyGene],
        config: Any,
        fallback_fitness: Any,
        fallback_summary: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[StrategyGene], Any, Optional[Dict[str, Any]]]:
        """
        上位候補をチューニングし、設定に応じた基準で最終勝者を選び直す。
        """
        if current_best_gene is None:
            return current_best_gene, fallback_fitness, fallback_summary

        if config.enable_multi_objective:
            tuned_gene = self.tune_elite_parameters(current_best_gene, config)
            refreshed_fitness, refreshed_summary = self.refresh_best_gene_reporting(
                best_gene=tuned_gene,
                config=config,
                fallback_fitness=fallback_fitness,
                fallback_summary=fallback_summary,
            )
            return tuned_gene, refreshed_fitness, refreshed_summary

        tuning_candidates = self.select_tuning_candidates(
            population,
            config,
            fallback_gene=current_best_gene,
        )
        if not tuning_candidates:
            refreshed_fitness, refreshed_summary = self.refresh_best_gene_reporting(
                best_gene=current_best_gene,
                config=config,
                fallback_fitness=fallback_fitness,
                fallback_summary=fallback_summary,
            )
            return current_best_gene, refreshed_fitness, refreshed_summary

        tuned_candidates = self.tune_candidate_genes(tuning_candidates, config)
        if config.two_stage_selection_config.enabled:
            tuned_winner = self.select_best_tuned_candidate(
                tuned_candidates,
                config,
            )
        else:
            tuned_winner = self.select_best_tuned_candidate_by_fitness(
                tuned_candidates,
                config,
            )
        if tuned_winner is None:
            refreshed_fitness, refreshed_summary = self.refresh_best_gene_reporting(
                best_gene=current_best_gene,
                config=config,
                fallback_fitness=fallback_fitness,
                fallback_summary=fallback_summary,
            )
            return current_best_gene, refreshed_fitness, refreshed_summary

        if config.two_stage_selection_config.enabled:
            logger.info(
                "[Tuning] %s候補をチューニングし、robustness 再選抜で最終勝者を決定しました",
                len(tuned_candidates),
            )
        else:
            logger.info(
                "[Tuning] %s候補をチューニングし、主 fitness で最終勝者を決定しました",
                len(tuned_candidates),
            )
        return tuned_winner

    def refresh_best_gene_reporting(
        self,
        *,
        best_gene: Optional[StrategyGene],
        config: Any,
        fallback_fitness: Any,
        fallback_summary: Optional[Dict[str, Any]],
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """
        チューニング後の最良遺伝子を再評価し、summary を最新化する。
        """
        if best_gene is None:
            return fallback_fitness, fallback_summary

        refreshed_fitness = fallback_fitness
        try:
            evaluated = self.evaluate_individual_with_full_fidelity(best_gene, config)
            if config.enable_multi_objective:
                refreshed_fitness = tuple(evaluated)
            elif isinstance(evaluated, (tuple, list)) and evaluated:
                refreshed_fitness = float(evaluated[0])
        except Exception as exc:
            logger.warning("最良遺伝子の再評価に失敗しました: %s", exc)

        refreshed_summary = self.build_individual_evaluation_summary(
            best_gene,
            config,
            force_robustness=bool(config.two_stage_selection_config.enabled),
            primary_fitness=self.extract_primary_fitness_from_result(
                refreshed_fitness
            ),
        )
        return refreshed_fitness, refreshed_summary or fallback_summary

    def build_individual_evaluation_summary(
        self,
        individual: Any,
        config: Any,
        *,
        force_robustness: bool = False,
        primary_fitness: Optional[float] = None,
        selection_rank_override: Optional[int] = None,
        selection_score_override: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        個体の評価 report から保存向け summary を構築する。
        """
        if individual is None:
            return None

        get_cached_robustness_report = getattr(
            self.individual_evaluator,
            "get_cached_robustness_report",
            None,
        )
        evaluate_robustness_report = getattr(
            self.individual_evaluator,
            "evaluate_robustness_report",
            None,
        )
        get_cached_evaluation_report = getattr(
            self.individual_evaluator,
            "get_cached_evaluation_report",
            None,
        )

        report: Optional[object] = None
        if callable(get_cached_robustness_report):
            report = get_cached_robustness_report(individual, config)

        if report is None and force_robustness and callable(evaluate_robustness_report):
            try:
                report = evaluate_robustness_report(individual, config)
            except Exception as exc:
                logger.debug("summary 用 robustness 評価に失敗しました: %s", exc)

        if report is None and callable(get_cached_evaluation_report):
            report = get_cached_evaluation_report(individual)

        if report is not None and is_evaluation_report(report):
            if report.metadata.get("evaluation_fidelity") == "coarse":
                report = None

        if report is None and is_multi_fidelity_enabled(config):
            try:
                self.evaluate_individual_with_full_fidelity(individual, config)
            except Exception as exc:
                logger.debug("summary 用 full 評価に失敗しました: %s", exc)
            if callable(get_cached_evaluation_report):
                report = get_cached_evaluation_report(individual)

        if not is_evaluation_report(report):
            return None

        from math import isfinite
        from .report_selection import extract_primary_fitness, get_two_stage_rank, get_two_stage_score
        from ..evaluation.report_persistence import build_report_summary

        if primary_fitness is None:
            fitness_score = extract_primary_fitness(individual)
            numeric_fitness = fitness_score if isfinite(fitness_score) else None
        else:
            numeric_fitness = (
                float(primary_fitness) if isfinite(float(primary_fitness)) else None
            )

        selection_rank = selection_rank_override
        if selection_rank is None:
            selection_rank = get_two_stage_rank(individual)

        selection_score: Any = selection_score_override
        if selection_score is None:
            selection_score = get_two_stage_score(individual)
        if not isinstance(selection_score, (tuple, list)):
            selection_score = None

        return build_report_summary(
            cast(EvaluationReport, report),
            selection_rank=selection_rank if isinstance(selection_rank, int) else None,
            selection_score=selection_score,
            fitness_score=numeric_fitness,
        )

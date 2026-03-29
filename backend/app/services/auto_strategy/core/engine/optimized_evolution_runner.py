"""
最適化された進化計算実行モジュール

進化計算の実行を担当する最適化されたEvolutionRunnerクラスを提供します。
主な最適化ポイント:
1. 交叉・突然変異のバッチ処理
2. 未評価個体のフィルタリング最適化
3. メモリ効率的な個体管理
"""

import gc
import logging
import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from deap import tools

from app.services.auto_strategy.core.fitness.fitness_sharing import FitnessSharing
from app.services.auto_strategy.core.engine.ga_utils import (
    _invalidate_individual_cache,
    _set_fitness_values,
)
from app.services.auto_strategy.core.engine.report_selection import (
    build_report_rank_key,
    extract_primary_fitness,
    get_individual_identity,
    merge_reranked_elites,
    get_two_stage_elite_count,
    get_two_stage_pool_size,
    is_evaluation_report,
    set_two_stage_metadata,
)
from app.services.auto_strategy.core.evaluation.parallel_evaluator import (
    ParallelEvaluator,
)

logger = logging.getLogger(__name__)


class EvolutionStoppedError(RuntimeError):
    """進化処理が停止要求によって中断されたことを示す例外"""


class OptimizedEvolutionRunner:
    """
    最適化された進化計算の実行を担当するクラス

    主な最適化ポイント:
    1. 交叉・突然変異のバッチ処理
    2. 未評価個体のフィルタリング最適化
    3. メモリ効率的な個体管理
    4. 統計情報の効率的な収集
    """

    def __init__(
        self,
        toolbox,
        stats,
        fitness_sharing: Optional[FitnessSharing] = None,
        population: Optional[List[Any]] = None,
        parallel_evaluator: Optional[ParallelEvaluator] = None,
        individual_evaluator: Optional[Any] = None,
    ):
        """
        初期化

        Args:
            toolbox: DEAPツールボックス
            stats: 統計情報収集オブジェクト
            fitness_sharing: 適応度共有オブジェクト（オプション）
            population: 個体集団（適応的突然変異用）
            parallel_evaluator: 並列評価器（オプション）
        """
        self.toolbox = toolbox
        self.stats = stats
        self.fitness_sharing = fitness_sharing
        self.population = population  # 適応的突然変異用
        self.parallel_evaluator = parallel_evaluator
        self.individual_evaluator = individual_evaluator

        # 互換性維持のためキャッシュ領域は残すが、
        # 確率的オペレータの結果自体は再利用しない。
        self._crossover_cache: Dict[str, Any] = {}
        self._mutation_cache: Dict[str, Any] = {}

    def run_evolution(
        self,
        population: List[Any],
        config: Any,
        halloffame: Optional[Any] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> tuple[List[Any], Any]:
        """
        進化アルゴリズムの実行（単一・多目的 統一版）

        Args:
            population: 初期個体群
            config: GA設定
            halloffame: 殿堂入り個体リスト（HallOfFame または ParetoFront）

        Returns:
            (最終個体群, 進化ログ)
        """
        if should_stop and should_stop():
            raise EvolutionStoppedError("進化処理は開始前に停止されました")

        logger.info(
            f"進化アルゴリズムを開始（世代数: {config.generations}, 目的数: {len(config.objectives)}）"
        )

        # 初期適応度評価
        population = self._evaluate_population(population)
        self._update_dynamic_objective_scalars(population, config)

        # Hall of Fame / Pareto Front 初回更新
        if halloffame is not None:
            halloffame.update(population)

        logbook = tools.Logbook()

        # 世代ループ
        for gen in range(config.generations):
            if should_stop and should_stop():
                raise EvolutionStoppedError(
                    f"進化処理は世代 {gen + 1} の開始前に停止されました"
                )

            # GC制御: 世代の変わり目でまとめて回収し、計算中の停止を防ぐ
            gc.collect()
            gc.disable()

            try:
                logger.debug("世代 %s/%s を開始", gen + 1, config.generations)

                # 適応度共有の適用（有効な場合、世代毎）
                if (
                    getattr(config, "enable_fitness_sharing", False)
                    and self.fitness_sharing
                ):
                    population = self.fitness_sharing.apply_fitness_sharing(population)

                # 選択（親個体の選択）
                offspring = list(self.toolbox.map(self.toolbox.clone, population))

                # 最適化: 交叉のバッチ処理
                offspring = self._apply_crossover_batch(offspring, config)

                # 最適化: 突然変異のバッチ処理
                offspring = self._apply_mutation_batch(offspring, config)

                # 未評価個体の評価（並列評価対応）
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                self._evaluate_invalid_individuals(invalid_ind)

                if should_stop and should_stop():
                    raise EvolutionStoppedError(
                        f"進化処理は世代 {gen + 1} の評価後に停止されました"
                    )

                # 次世代の選択 (mu+lambda)
                candidate_population = offspring + population
                population[:] = self._apply_two_stage_selection(
                    candidate_population,
                    len(population),
                    config,
                )

                if should_stop and should_stop():
                    raise EvolutionStoppedError(
                        f"進化処理は世代 {gen + 1} の選択後に停止されました"
                    )

                self._update_dynamic_objective_scalars(population, config)

                # 統計の記録
                record = self.stats.compile(population) if self.stats else {}
                logbook.record(gen=gen, **record)

                # Hall of Fame / Pareto Front の更新
                if halloffame is not None:
                    halloffame.update(population)
            finally:
                # ループ終了時にGCを有効化（エラー時も含む）
                gc.enable()

        logger.info("進化アルゴリズム完了")
        return population, logbook

    def _apply_crossover_batch(self, offspring: List[Any], config: Any) -> List[Any]:
        """
        交叉のバッチ処理（最適化版）

        最適化:
        - インデックスを使ってリストを直接更新
        """
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < config.crossover_rate:
                child1, child2 = offspring[i], offspring[i + 1]
                new_child1, new_child2 = self.toolbox.mate(child1, child2)

                offspring[i] = new_child1
                offspring[i + 1] = new_child2
                _invalidate_individual_cache(offspring[i])
                _invalidate_individual_cache(offspring[i + 1])

        return offspring

    def _apply_mutation_batch(self, offspring: List[Any], config: Any) -> List[Any]:
        """
        突然変異のバッチ処理（最適化版）

        最適化:
        - インデックスを使ってリストを直接更新
        """
        for i in range(len(offspring)):
            if random.random() < config.mutation_rate:
                mutant = offspring[i]
                result = self.toolbox.mutate(mutant)
                new_mutant = result[0]

                offspring[i] = new_mutant
                _invalidate_individual_cache(offspring[i])

        return offspring

    def _get_crossover_cache_key(self, parent1: Any, parent2: Any) -> str:
        """交叉キャッシュキーを生成"""
        try:
            p1_id = getattr(parent1, "id", "") or str(id(parent1))
            p2_id = getattr(parent2, "id", "") or str(id(parent2))
            return f"{p1_id}:{p2_id}"
        except Exception:
            return str(id(parent1)) + ":" + str(id(parent2))

    def _get_mutation_cache_key(self, individual: Any) -> str:
        """突然変異キャッシュキーを生成"""
        try:
            ind_id = getattr(individual, "id", "") or str(id(individual))
            return ind_id
        except Exception:
            return str(id(individual))

    def _evaluate_population(self, population: List[Any]) -> List[Any]:
        """
        個体群の適応度評価（並列評価対応）

        Args:
            population: 評価対象の個体群

        Returns:
            評価値（fitness.values）が設定された個体群
        """
        if self.parallel_evaluator:
            fitnesses = self.parallel_evaluator.evaluate_population(population)
            _set_fitness_values(population, fitnesses)
        else:
            fitnesses = self.toolbox.map(self.toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

        return population

    def _evaluate_invalid_individuals(self, invalid_individuals: List[Any]) -> None:
        """
        未評価個体のみを評価（最適化版）

        Args:
            invalid_individuals: 未評価個体のリスト
        """
        if not invalid_individuals:
            return

        if self.parallel_evaluator:
            fitnesses = self.parallel_evaluator.evaluate_population(invalid_individuals)
            _set_fitness_values(invalid_individuals, fitnesses)
        else:
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_individuals)
            for ind, fit in zip(invalid_individuals, fitnesses):
                ind.fitness.values = fit

    def _update_dynamic_objective_scalars(
        self, population: List[Any], config: Any
    ) -> None:
        """
        動的目的スカラーの更新

        Args:
            population: 個体群
            config: GA設定
        """
        if not getattr(config, "dynamic_objective_reweighting", False):
            config.objective_dynamic_scalars = {}
            return

        if not population:
            config.objective_dynamic_scalars = {}
            return

        scalars: dict[str, float] = {}
        for index, objective in enumerate(getattr(config, "objectives", [])):
            values: List[float] = []
            for individual in population:
                fitness = getattr(individual, "fitness", None)
                if not fitness or not getattr(fitness, "valid", False):
                    continue

                fitness_values = getattr(fitness, "values", ())
                if len(fitness_values) <= index:
                    continue

                try:
                    values.append(float(fitness_values[index]))
                except (TypeError, ValueError):
                    continue

            if not values:
                continue

            average_value = float(np.mean(values))
            if objective in {"max_drawdown", "ulcer_index", "trade_frequency_penalty"}:
                scalars[objective] = min(2.0, 1.0 + max(average_value, 0.0))
            else:
                scalars[objective] = 1.0

        config.objective_dynamic_scalars = scalars

    def _apply_two_stage_selection(
        self,
        candidate_population: List[Any],
        population_size: int,
        config: Any,
    ) -> List[Any]:
        """粗選抜後に report ベースでエリートを差し替える。"""
        selected = list(self.toolbox.select(candidate_population, population_size))
        self._clear_two_stage_metadata(selected)

        elite_count = get_two_stage_elite_count(config, population_size)
        if elite_count <= 0 or self.individual_evaluator is None:
            return selected

        rerank_pool_size = get_two_stage_pool_size(
            len(candidate_population), elite_count, config
        )
        rerank_candidates = self._select_top_candidates(
            candidate_population,
            rerank_pool_size,
        )
        reranked_elites = self._select_report_ranked_elites(
            rerank_candidates,
            elite_count,
            config,
        )
        if not reranked_elites:
            return selected

        self._mark_two_stage_elites(reranked_elites)

        return merge_reranked_elites(selected, reranked_elites, population_size)

    def _select_top_candidates(
        self,
        candidate_population: List[Any],
        limit: int,
    ) -> List[Any]:
        """単一目的 fitness の上位候補を返す。"""
        if limit <= 0:
            return []
        ranked = sorted(
            candidate_population,
            key=extract_primary_fitness,
            reverse=True,
        )
        return ranked[:limit]

    def _select_report_ranked_elites(
        self,
        candidates: List[Any],
        elite_count: int,
        config: Any,
    ) -> List[tuple[Any, tuple[float, ...]]]:
        """候補を report ベースで再ランクしてエリートを返す。"""
        ranked_candidates = []
        seen_keys = set()

        for candidate in candidates:
            candidate_key = get_individual_identity(candidate)
            if candidate_key in seen_keys:
                continue
            seen_keys.add(candidate_key)

            report = self._resolve_evaluation_report(candidate, config)
            rank_key = build_report_rank_key(
                candidate,
                report,
                getattr(config, "two_stage_min_pass_rate", 0.0),
            )
            ranked_candidates.append((rank_key, candidate))

        ranked_candidates.sort(key=lambda item: item[0], reverse=True)
        return [
            (candidate, rank_key)
            for rank_key, candidate in ranked_candidates[:elite_count]
        ]

    def _resolve_evaluation_report(
        self,
        candidate: Any,
        config: Any,
    ) -> Optional[Any]:
        """候補の report を取得し、必要なら主プロセスで再評価する。"""
        if self.individual_evaluator is None:
            return None

        report = None
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
        if callable(get_cached_robustness_report):
            report = get_cached_robustness_report(candidate, config)
            if not is_evaluation_report(report):
                report = None

        if report is None and callable(evaluate_robustness_report):
            try:
                report = evaluate_robustness_report(candidate, config)
                if not is_evaluation_report(report):
                    report = None
            except Exception as exc:
                logger.debug("二段階選抜用の robustness 評価に失敗しました: %s", exc)

        get_cached_report = getattr(
            self.individual_evaluator,
            "get_cached_evaluation_report",
            None,
        )
        if report is None and callable(get_cached_report):
            report = get_cached_report(candidate)
            if not is_evaluation_report(report):
                report = None

        if report is None:
            evaluate_individual = getattr(
                self.individual_evaluator,
                "evaluate_individual",
                None,
            )
            if callable(evaluate_individual):
                try:
                    evaluate_individual(candidate, config)
                except Exception as exc:
                    logger.debug("二段階選抜用の再評価に失敗しました: %s", exc)

            if callable(get_cached_report):
                report = get_cached_report(candidate)
                if not is_evaluation_report(report):
                    report = None

        return report

    def _clear_two_stage_metadata(self, individuals: List[Any]) -> None:
        """前世代の二段階選抜メタデータをクリアする。"""
        seen_keys = set()
        for individual in individuals:
            candidate_key = get_individual_identity(individual)
            if candidate_key in seen_keys:
                continue
            seen_keys.add(candidate_key)
            self._set_two_stage_metadata(individual, None, None)

    def _mark_two_stage_elites(
        self,
        reranked_elites: List[tuple[Any, tuple[float, ...]]],
    ) -> None:
        """二段階選抜で確定したエリートへ順位を付与する。"""
        for rank, (individual, score) in enumerate(reranked_elites):
            self._set_two_stage_metadata(individual, rank, score)

    def _set_two_stage_metadata(
        self,
        individual: Any,
        rank: Optional[int],
        score: Optional[Any],
    ) -> None:
        """個体へ二段階選抜のメタデータを付与する。"""
        try:
            set_two_stage_metadata(individual, rank, score)
        except Exception:
            logger.debug("二段階選抜メタデータの設定をスキップしました")

    def clear_caches(self):
        """キャッシュをクリア"""
        self._crossover_cache.clear()
        self._mutation_cache.clear()

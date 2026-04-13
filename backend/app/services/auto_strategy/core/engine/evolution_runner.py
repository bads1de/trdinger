"""
進化計算実行モジュール

進化計算の実行を担当するEvolutionRunnerクラスを提供します。
単一目的と多目的最適化のロジックをカプセル化し、並列評価をサポートします。
"""

import gc
import logging
import random
from typing import Any, Callable, List, Optional

import numpy as np
from deap import tools
from app.services.auto_strategy.config import objective_registry

from ..evaluation.parallel_evaluator import ParallelEvaluator
from ..evaluation.evaluation_fidelity import (
    build_coarse_ga_config,
    get_multi_fidelity_candidate_limit,
    is_multi_fidelity_enabled,
)
from ..fitness.fitness_sharing import FitnessSharing
from .report_selection import (
    build_report_rank_key,
    extract_primary_fitness,
    get_individual_identity,
    get_two_stage_elite_count,
    get_two_stage_pool_size,
    is_evaluation_report,
    merge_reranked_elites,
    set_two_stage_metadata,
)

logger = logging.getLogger(__name__)


def _invalidate_individual_cache(individual: Any) -> None:
    """個体のキャッシュを無効化する。

    交叉・突然変異後に呼ばれ、個体に紐づくキャッシュデータをクリアする。
    """
    try:
        if hasattr(individual, "_cache"):
            individual._cache = {}
    except Exception as e:
        logger.debug("個体キャッシュのクリアに失敗しました: %s", e)


def _set_fitness_values(population: List[Any], fitnesses: List[tuple[float, ...]]) -> None:
    """個体群にフィットネス値を設定する。"""
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit


class EvolutionStoppedError(RuntimeError):
    """進化処理が停止要求によって中断されたことを示す例外"""


class EvolutionRunner:
    """
    進化計算の実行を担当するクラス

    単一目的と多目的最適化のロジックをカプセル化した独立クラス。
    GAエンジンから分割されたが、現在は同一モジュール内で定義。
    並列評価をサポート。
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

    def run_evolution(
        self,
        population: List[Any],
        config: Any,
        halloffame: Optional[Any] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> tuple[List[Any], Any]:
        """
        設定された世代数分、進化計算アルゴリズムを実行します。

        このメソッドはGAのメインループを管理し、以下のライフサイクルを繰り返します：
        1. **初期評価**: 第0世代（初期集団）の適応度を評価。
        2. **世代ループ**:
           - **停止チェック**: ユーザーからの停止リクエスト（`should_stop`）を確認。
           - **GC制御**: メモリリーク防止のため、各世代の開始時にガベージコレクションを明示的に実行。
           - **適応度共有 (Fitness Sharing)**: 個体の多様性を維持するため、密集した個体の適応度を調整。
           - **親選択 (Selection)**: 適応度に基づき、次世代の親となる個体を抽出（トーナメント等）。
           - **交叉 (Crossover)**: 二つの個体を組み合わせて新しい個体（子）を生成。
           - **突然変異 (Mutation)**: 確率的に遺伝子を書き換え、局所最適解からの脱出を図る。
           - **子個体の評価**: 新しく生成された個体の適応度を計算（並列実行をサポート）。
           - **エリート保存**: ホール・オブ・フェイム（殿堂）を最新の優良個体で更新。
        3. **結果の集計**: 最終世代の集団と、全世代の統計ログを返却。

        Args:
            population (List[Any]): 進化を開始する初期個体群のリスト。
            config (Any): 世代数、交叉率、突然変異率等のハイパーパラメータを含む設定オブジェクト。
            halloffame (Optional[Any]): 最良個体を保持するDEAPの HallOfFame または ParetoFront オブジェクト。
            should_stop (Optional[Callable[[], bool]]): 外部から中断を指示するためのコールバック関数。

        Returns:
            tuple[List[Any], tools.Logbook]: (最終世代の個体群, 各世代の統計情報を含むログブック)。

        Raises:
            EvolutionStoppedError: `should_stop()` が True を返し、進化が途中で中断された場合。
            Exception: 評価中または進化計算プロセス中に発生した予期しないエラー。
        """
        if should_stop and should_stop():
            raise EvolutionStoppedError("進化処理は開始前に停止されました")

        logger.info(
            f"進化アルゴリズムを開始（世代数: {config.generations}, 目的数: {len(config.objectives)}）"
        )

        # 初期適応度評価
        population = self._evaluate_population(population, config)
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
                    config.fitness_sharing.get("enable_fitness_sharing", False)
                    and self.fitness_sharing
                ):
                    population = self.fitness_sharing.apply_fitness_sharing(population)

                # 選択（親個体の選択）
                # cloneを使用することで、交叉・変異が元の個体に影響しないようにする
                offspring = list(self.toolbox.map(self.toolbox.clone, population))

                offspring = self._apply_crossover_batch(offspring, config)
                offspring = self._apply_mutation_batch(offspring, config)

                # 未評価個体の評価（並列評価対応）
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                self._evaluate_invalid_individuals(invalid_ind, config)

                if should_stop and should_stop():
                    raise EvolutionStoppedError(
                        f"進化処理は世代 {gen + 1} の評価後に停止されました"
                    )

                # 次世代の選択 (mu+lambda)
                # toolbox.select は DEAPSetup で NSGA-II などが登録されている
                candidate_population = offspring + population
                self._promote_top_candidates_with_full_fidelity(
                    candidate_population,
                    config,
                )
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
        交叉のバッチ処理

        個体群に対して交叉操作をバッチで適用します。
        交叉率に基づいて、隣接するペアを交叉します。

        Args:
            offspring: 子個体リスト
            config: GA設定オブジェクト

        Returns:
            List[Any]: 交叉後の子個体リスト

        Note:
            交叉後の個体のキャッシュは無効化されます。
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
        突然変異のバッチ処理

        個体群に対して突然変異操作をバッチで適用します。
        突然変異率に基づいて、各個体を突然変異させます。

        Args:
            offspring: 子個体リスト
            config: GA設定オブジェクト

        Returns:
            List[Any]: 突然変異後の子個体リスト

        Note:
            突然変異後の個体のキャッシュは無効化されます。
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
        """
        交叉キャッシュキーを生成する

        親個体のIDに基づいて交叉キャッシュキーを生成します。

        Args:
            parent1: 親個体1
            parent2: 親個体2

        Returns:
            str: キャッシュキー（"id1:id2"形式）
        """
        try:
            p1_id = getattr(parent1, "id", "") or str(id(parent1))
            p2_id = getattr(parent2, "id", "") or str(id(parent2))
            return f"{p1_id}:{p2_id}"
        except Exception as e:
            logger.debug("交叉キャッシュキー生成に失敗しました: %s", e)
            return str(id(parent1)) + ":" + str(id(parent2))

    def _get_mutation_cache_key(self, individual: Any) -> str:
        """
        突然変異キャッシュキーを生成する

        個体のIDに基づいて突然変異キャッシュキーを生成します。

        Args:
            individual: 個体

        Returns:
            str: キャッシュキー（個体ID）
        """
        try:
            ind_id = getattr(individual, "id", "") or str(id(individual))
            return ind_id
        except Exception as e:
            logger.debug("突然変異キャッシュキー生成に失敗しました: %s", e)
            return str(id(individual))

    def clear_caches(self) -> None:
        """
        バッチ互換のキャッシュ領域をクリアする

        現在はこのメソッドは外部インターフェース互換のために残されています。
        """
        pass

    def _evaluate_population(
        self,
        population: List[Any],
        config: Optional[Any] = None,
    ) -> List[Any]:
        """
        個体群の適応度評価（並列評価対応）

        ツールボックスに登録された評価関数を使用して、
        集団内の全個体の適応度を計算します。並列評価器が利用可能な場合は、
        複数のプロセスで並列に評価を実行し、計算時間を短縮します。

        Args:
            population: 評価対象の個体群

        Returns:
            評価値（fitness.values）が設定された個体群
        """
        if config is not None and is_multi_fidelity_enabled(config):
            coarse_config = build_coarse_ga_config(config)
            fitnesses = self._evaluate_individuals_with_config(population, coarse_config)
            _set_fitness_values(population, fitnesses)
            self._mark_evaluation_fidelity(population, "coarse")
            self._promote_top_candidates_with_full_fidelity(population, config)
            return population

        fitnesses = self._evaluate_individuals_with_config(population, config)
        _set_fitness_values(population, fitnesses)
        self._mark_evaluation_fidelity(population, "full")

        return population

    def _evaluate_invalid_individuals(
        self,
        invalid_ind: List[Any],
        config: Optional[Any] = None,
    ) -> None:
        """
        適応度が無効な個体のみを評価（並列評価対応）

        Args:
            invalid_ind: 評価対象の無効な個体リスト
        """
        if not invalid_ind:
            return

        if config is not None and is_multi_fidelity_enabled(config):
            coarse_config = build_coarse_ga_config(config)
            fitnesses = self._evaluate_individuals_with_config(invalid_ind, coarse_config)
            _set_fitness_values(invalid_ind, fitnesses)
            self._mark_evaluation_fidelity(invalid_ind, "coarse")
            return

        fitnesses = self._evaluate_individuals_with_config(invalid_ind, config)
        _set_fitness_values(invalid_ind, fitnesses)
        self._mark_evaluation_fidelity(invalid_ind, "full")

    def _evaluate_individuals_with_config(
        self,
        individuals: List[Any],
        config: Optional[Any],
    ) -> List[tuple[float, ...]]:
        """
        設定に応じて個体列を評価する

        個体リストを評価して適応度値を返します。
        並列評価器が利用可能な場合は並列評価を行います。

        Args:
            individuals: 評価対象の個体リスト
            config: GA設定オブジェクト（オプション）

        Returns:
            List[tuple[float, ...]]: 適応度値のリスト

        Note:
            - 並列評価器が優先的に使用されます
            - 個別評価器が使用可能な場合はそれを使用
            - それ以外はツールボックスの評価関数を使用
        """
        if not individuals:
            return []

        if self.parallel_evaluator:
            return list(self.parallel_evaluator.evaluate_population(individuals))

        if self.individual_evaluator is not None and config is not None:
            return [
                self.individual_evaluator.evaluate(individual, config)
                for individual in individuals
            ]

        return list(self.toolbox.map(self.toolbox.evaluate, individuals))

    def _promote_top_candidates_with_full_fidelity(
        self,
        candidate_population: List[Any],
        config: Optional[Any],
    ) -> None:
        """
        粗評価上位だけ full fidelity で再評価する

        マルチフィデリティ評価が有効な場合、粗評価で上位の候補のみを
        完全忠実度で再評価します。

        Args:
            candidate_population: 候補個体群
            config: GA設定オブジェクト（オプション）

        Note:
            既に完全忠実度で評価された個体はスキップされます。
        """
        if (
            not candidate_population
            or not is_multi_fidelity_enabled(config)
            or self.individual_evaluator is None
        ):
            return

        limit = get_multi_fidelity_candidate_limit(len(candidate_population), config)
        if limit <= 0:
            return

        for candidate in self._select_top_candidates(candidate_population, limit):
            if getattr(candidate, "_evaluation_fidelity", None) == "full":
                continue
            fitness = self.individual_evaluator.evaluate(
                candidate,
                config,
                force_refresh=True,
            )
            candidate.fitness.values = fitness
            setattr(candidate, "_evaluation_fidelity", "full")

    def _mark_evaluation_fidelity(
        self,
        individuals: List[Any],
        fidelity: str,
    ) -> None:
        """
        個体へ現在の評価粒度を付与する

        個体に評価粒度（coarseまたはfull）のメタデータを設定します。

        Args:
            individuals: 個体リスト
            fidelity: 評価粒度（"coarse"または"full"）

        Note:
            メタデータ設定に失敗した場合はログを出力してスキップします。
        """
        for individual in individuals:
            try:
                setattr(individual, "_evaluation_fidelity", fidelity)
            except Exception as e:
                logger.debug("評価粒度メタデータの設定をスキップしました: %s", e)

    def _update_dynamic_objective_scalars(
        self, population: List[Any], config: Any
    ) -> None:
        """
        リスク回避型の重み付けのために、動的な目的正規化係数を更新します。

        集団全体の平均的なパフォーマンスに基づいて、特定の指標（ドローダウンなど）の
        ペナルティやウェイトを調整します。

        Args:
            population: 現在の集団
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
            if objective_registry.is_dynamic_scalar_objective(objective):
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
        """
        粗選抜後に report ベースでエリートを差し替える

        二段階選抜を適用します。まず通常の選択を行い、
        その後上位候補をreportベースで再ランクしてエリートを差し替えます。

        Args:
            candidate_population: 候補個体群
            population_size: 集団サイズ
            config: GA設定オブジェクト

        Returns:
            List[Any]: 選択された個体群

        Note:
            個別評価器がない場合は通常の選択のみ行います。
        """
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
        """
        単一目的 fitness の上位候補を返す

        プライマリフィットネスに基づいて上位候補を選択します。

        Args:
            candidate_population: 候補個体群
            limit: 取得する候補数

        Returns:
            List[Any]: 上位候補リスト

        Note:
            limitが0以下の場合は空リストを返します。
        """
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
        """
        候補を report ベースで再ランクしてエリートを返す

        評価レポートに基づいて候補を再ランクし、上位エリートを返します。

        Args:
            candidates: 候補リスト
            elite_count: エリート数
            config: GA設定オブジェクト

        Returns:
            List[tuple[Any, tuple[float, ...]]]: (個体, ランクキー)のタプルリスト

        Note:
            重複する候補はスキップされます。
        """
        ranked_candidates = []
        seen_keys = set()

        for candidate in candidates:
            candidate_key = get_individual_identity(candidate)
            if candidate_key in seen_keys:
                continue
            seen_keys.add(candidate_key)

            report = self._resolve_evaluation_report(candidate, config)
            two_stage_config = getattr(config, "two_stage_selection_config", None)
            rank_key = build_report_rank_key(
                candidate,
                report,
                getattr(two_stage_config, "min_pass_rate", 0.0),
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
        """
        候補の report を取得し、必要なら主プロセスで再評価する

        候補の評価レポートを取得します。キャッシュにない場合は
        必要に応じて再評価を行います。

        Args:
            candidate: 候補個体
            config: GA設定オブジェクト

        Returns:
            Optional[Any]: 評価レポート、取得失敗時はNone

        Note:
            個別評価器がない場合はNoneを返します。
        """
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
        """
        前世代の二段階選抜メタデータをクリアする

        個体群から前世代の二段階選抜メタデータをクリアします。

        Args:
            individuals: 個体リスト

        Note:
            重複する個体は一度のみ処理されます。
        """
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
        """
        二段階選抜で確定したエリートへ順位を付与する

        再ランクされたエリートに順位とスコアのメタデータを設定します。

        Args:
            reranked_elites: (個体, ランクキー)のタプルリスト
        """
        for rank, (individual, score) in enumerate(reranked_elites):
            self._set_two_stage_metadata(individual, rank, score)

    def _set_two_stage_metadata(
        self,
        individual: Any,
        rank: Optional[int],
        score: Optional[Any],
    ) -> None:
        """
        個体へ二段階選抜のメタデータを付与する

        個体に二段階選抜の順位とスコアを設定します。

        Args:
            individual: 個体
            rank: 順位（オプション）
            score: スコア（オプション）

        Note:
            メタデータ設定に失敗した場合はログを出力してスキップします。
        """
        try:
            set_two_stage_metadata(individual, rank, score)
        except Exception as e:
            logger.debug("二段階選抜メタデータの設定をスキップしました: %s", e)

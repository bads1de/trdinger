"""
進化計算実行モジュール

進化計算の実行を担当するEvolutionRunnerクラスを提供します。
目的関数数に依存しない最適化ロジックをカプセル化し、並列評価をサポートします。
"""

import gc
import logging
import math
import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ...config.ga_config import GAConfig

import numpy as np
from deap import tools

from app.services.auto_strategy.config import objective_registry
from app.services.auto_strategy.config.constants import PENALTY_FITNESS_MAGNITUDE
from app.services.auto_strategy.config.ga_config import SurvivalSelectionConfig

from ..evaluation.evaluation_fidelity import (
    build_coarse_ga_config,
    get_multi_fidelity_candidate_limit,
    is_multi_fidelity_enabled,
)
from ..evaluation.parallel_evaluator import ParallelEvaluator
from ..fitness.fitness_sharing import FitnessSharing
from ..fitness.fitness_validation import has_valid_fitness
from .ga_utils import _invalidate_individual_cache, _set_fitness_values
from .restricted_tournament import restricted_tournament_replace
from .two_stage_selection import DEFAULT_MIN_PASS_RATE, TwoStageSelection

logger = logging.getLogger(__name__)

# 定数
GC_THRESHOLD_GENERATIONS = 2000
GC_THRESHOLD_DIVISIONS = 20
DYNAMIC_SCALAR_MAX = 2.0
DEFAULT_SCALAR_VALUE = 1.0


class EvolutionStoppedError(RuntimeError):
    """進化処理が停止要求によって中断されたことを示す例外"""


class EvolutionRunner:
    """
    進化計算の実行を担当するクラス

    目的関数数に依存しない最適化ロジックをカプセル化した独立クラス。
    GAエンジンから分割された。並列評価をサポート。

    二段階選抜（report ベースのエリート再ランクと behavior 多様化）は
    ``core.engine.two_stage_selection.TwoStageSelection`` に分離済みで、
    本クラスは ``_apply_two_stage_selection`` / ``_select_top_candidates`` で委譲する。
    """

    def __init__(
        self,
        toolbox: Any,
        stats: Any,
        fitness_sharing: FitnessSharing | None = None,
        population: list[Any] | None = None,
        parallel_evaluator: ParallelEvaluator | None = None,
        individual_evaluator: Any | None = None,
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
        self._wire_fitness_sharing_provider()
        self._two_stage_selection = TwoStageSelection(
            toolbox=self.toolbox,
            fitness_sharing=self.fitness_sharing,
            individual_evaluator=self.individual_evaluator,
            behavior_profile_provider=getattr(
                self.parallel_evaluator,
                "get_cached_behavior_profile",
                None,
            )
            if self.parallel_evaluator is not None
            else None,
        )

    def _wire_fitness_sharing_provider(self) -> None:
        """fitness sharing に behavior/report 取得関数を接続する。"""
        if self.fitness_sharing is None:
            return

        set_provider = getattr(
            self.fitness_sharing,
            "set_evaluation_report_provider",
            None,
        )
        if not callable(set_provider):
            return

        get_parallel_behavior = getattr(
            self.parallel_evaluator,
            "get_cached_behavior_profile",
            None,
        )
        get_cached_report = getattr(
            self.individual_evaluator,
            "get_cached_evaluation_report",
            None,
        )

        def resolve_behavior_or_report(individual: Any) -> Any:
            if getattr(individual, "_evaluation_fidelity", None) == "full" and callable(
                get_cached_report
            ):
                report = get_cached_report(individual)
                if report is not None:
                    return report
            if callable(get_parallel_behavior):
                behavior_profile = get_parallel_behavior(individual)
                if behavior_profile:
                    return behavior_profile
            if callable(get_cached_report):
                return get_cached_report(individual)
            return None

        set_provider(resolve_behavior_or_report)

    def run_evolution(
        self,
        population: list[Any],
        config: "GAConfig",
        halloffame: Any | None = None,
        should_stop: Callable[[], bool] | None = None,
        progress_callback: Callable[[int, int, float | None], None] | None = None,
    ) -> tuple[list[Any], object]:
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
           - **進捗通知**: `progress_callback` で現在の世代・総世代・最高フィットネスを通知。
        3. **結果の集計**: 最終世代の集団と、全世代の統計ログを返却。

        Args:
            population (List[Any]): 進化を開始する初期個体群のリスト。
            config (Any): 世代数、交叉率、突然変異率等のハイパーパラメータを含む設定オブジェクト。
            halloffame (Optional[Any]): 最良個体を保持するDEAPの
                HallOfFame または ParetoFront オブジェクト。
            should_stop (Optional[Callable[[], bool]]): 外部から中断を指示するためのコールバック関数。
            progress_callback (
                Optional[Callable[[int, int, Optional[float]], None]]
            ):
                各世代終了時に呼び出される進捗通知コールバック。
                引数: (current_generation, total_generations, best_fitness)。

        Returns:
            tuple[List[Any], tools.Logbook]: (最終世代の個体群, 各世代の統計情報を含むログブック)。

        Raises:
            EvolutionStoppedError: `should_stop()` が True を返し、進化が途中で中断された場合。
            Exception: 評価中または進化計算プロセス中に発生した予期しないエラー。
        """
        if should_stop and should_stop():
            raise EvolutionStoppedError("進化処理は開始前に停止されました")

        logger.info(
            f"進化アルゴリズムを開始"
            f"（世代数: {config.generations}, "
            f"目的数: {len(config.objectives)}）"
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

            # GC制御: 世代の変わり目でまとめて回収し、評価中の不要なGCを抑制
            gc.collect()
            # GCを完全に無効化するのではなく、閾値を上げて頻度を下げる
            # デフォルト: (700, 10, 10) → 評価中は(2000, 20, 20)に変更
            original_threshold = gc.get_threshold()
            gc.set_threshold(
                GC_THRESHOLD_GENERATIONS,
                GC_THRESHOLD_DIVISIONS,
                GC_THRESHOLD_DIVISIONS,
            )

            try:
                logger.debug("世代 %s/%s を開始", gen + 1, config.generations)

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
                population[:] = self._apply_shared_selection(
                    candidate_population,
                    len(population),
                    config,
                    offspring=offspring,
                    parents=population,
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

                # 進捗通知
                if progress_callback:
                    best_fitness = record.get("max")
                    if isinstance(best_fitness, (list, tuple, np.ndarray)):
                        best_fitness = float(best_fitness[0])
                    else:
                        best_fitness = (
                            float(best_fitness) if best_fitness is not None else None
                        )
                    # ±inf / NaN は DB 保存や API 応答で問題になるため None に落とす
                    if best_fitness is not None and not math.isfinite(best_fitness):
                        best_fitness = None
                    progress_callback(gen, config.generations, best_fitness)
            finally:
                # ループ終了時に元のGC閾値を復元（エラー時も含む）
                gc.set_threshold(*original_threshold)

        logger.info("進化アルゴリズム完了")
        return population, logbook

    def _apply_crossover_batch(
        self, offspring: list[Any], config: "GAConfig"
    ) -> list[Any]:
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

    def _apply_mutation_batch(
        self, offspring: list[Any], config: "GAConfig"
    ) -> list[Any]:
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

    def _evaluate_population(
        self,
        population: list[Any],
        config: Optional["GAConfig"] = None,
    ) -> list[Any]:
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
            # 並列評価時はワーカー初期化時点で coarse config が適用済みのため、
            # 世代ごとの deepcopy を回避する。
            coarse_config = (
                config
                if self.parallel_evaluator is not None
                else build_coarse_ga_config(config)
            )
            fitnesses = self._evaluate_individuals_with_config(
                population, coarse_config
            )
            _set_fitness_values(population, fitnesses)
            self._mark_evaluation_fidelity(population, "coarse")
            self._promote_top_candidates_with_full_fidelity(population, config)
            return population

        fitnesses = self._evaluate_individuals_with_config(population, config)
        _set_fitness_values(population, fitnesses)
        self._mark_evaluation_fidelity(population, "full")

        return population

    def _build_default_fitness(self, config: Optional["GAConfig"]) -> tuple[float, ...]:
        """評価失敗個体に与えるデフォルトフィットネス（方向を考慮したペナルティ値）。

        0.0 を返すと最小化目的（max_drawdown 等）で最良スコアになり、
        失敗個体が選択を勝ち抜けてしまうため、目的ごとのペナルティ値を返す。
        """
        objectives: list[str] = []
        if config is not None:
            try:
                objectives = list(config.objectives or [])
            except Exception:
                objectives = []
        if not objectives:
            return (-PENALTY_FITNESS_MAGNITUDE,)
        return objective_registry.build_penalty_values(objectives)

    def _evaluate_invalid_individuals(
        self,
        invalid_ind: list[Any],
        config: Optional["GAConfig"] = None,
    ) -> None:
        """
        適応度が無効な個体のみを評価（並列評価対応）

        Args:
            invalid_ind: 評価対象の無効な個体リスト
        """
        if not invalid_ind:
            return

        if config is not None and is_multi_fidelity_enabled(config):
            # 並列評価時はワーカー初期化時点で coarse config が適用済みのため、
            # 世代ごとの deepcopy を回避する。
            coarse_config = (
                config
                if self.parallel_evaluator is not None
                else build_coarse_ga_config(config)
            )
            fitnesses = self._evaluate_individuals_with_config(
                invalid_ind, coarse_config
            )
            _set_fitness_values(invalid_ind, fitnesses)
            self._mark_evaluation_fidelity(invalid_ind, "coarse")
            return

        fitnesses = self._evaluate_individuals_with_config(invalid_ind, config)
        _set_fitness_values(invalid_ind, fitnesses)
        self._mark_evaluation_fidelity(invalid_ind, "full")

    def _evaluate_individuals_with_config(
        self,
        individuals: list[Any],
        config: Optional["GAConfig"],
    ) -> list[tuple[float, ...]]:
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
            default_fitness = self._build_default_fitness(config)
            # 世代ごとに更新される動的スカラーはワーカーの持つ設定が古いため、
            # 評価要求ごとに最新値を伝播する（並列ワーカー伝播）
            dynamic_scalars = None
            if config is not None:
                scalars = getattr(config, "objective_dynamic_scalars", None)
                if scalars:
                    dynamic_scalars = dict(scalars)
            return list(
                self.parallel_evaluator.evaluate_population(
                    individuals,
                    default_fitness=default_fitness,
                    dynamic_scalars=dynamic_scalars,
                )
            )

        if self.individual_evaluator is not None and config is not None:
            fitness_values = [
                self.individual_evaluator.evaluate(individual, config)
                for individual in individuals
            ]
            return fitness_values

        return list(self.toolbox.map(self.toolbox.evaluate, individuals))

    def _promote_top_candidates_with_full_fidelity(
        self,
        candidate_population: list[Any],
        config: Optional["GAConfig"],
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
            candidate._evaluation_fidelity = "full"

    def _mark_evaluation_fidelity(
        self,
        individuals: list[Any],
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
                individual._evaluation_fidelity = fidelity
            except Exception as e:
                logger.debug("評価粒度メタデータの設定をスキップしました: %s", e)

    def _update_dynamic_objective_scalars(
        self, population: list[Any], config: "GAConfig"
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
            values: list[float] = []
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

            # ±inf（制約違反ペナルティ）の fitness は平均計算の対象外とする。
            # そのまま np.mean に渡すと NaN / RuntimeWarning を引き起こす。
            finite_values = [v for v in values if math.isfinite(v)]
            if not finite_values:
                continue

            average_value = float(np.mean(finite_values))
            if objective_registry.is_dynamic_scalar_objective(objective):
                scalars[objective] = min(
                    DYNAMIC_SCALAR_MAX,
                    DEFAULT_SCALAR_VALUE + max(average_value, DEFAULT_MIN_PASS_RATE),
                )
            else:
                scalars[objective] = DEFAULT_SCALAR_VALUE

        config.objective_dynamic_scalars = scalars

    def _apply_shared_selection(
        self,
        candidate_population: list[Any],
        population_size: int,
        config: "GAConfig",
        offspring: list[Any] | None = None,
        parents: list[Any] | None = None,
    ) -> list[Any]:
        """生存選択の前段（RTR）と適応度共有を経て次世代を選ぶ。

        制限トーナメント置換 (RTR) が有効な場合、まず各子個体を構造的に
        最も近い既存個体と局所競争させ、ニッチごとに絞った生存プールを
        組む。これにより多数の類似した子が適応度差だけで多様な親を
        一斉に置き換える経路を断つ。

        共有は親集団だけ・子孫だけに適用すると、減衰した個体と生の適応度の
        個体が同じ選択プールで競争してしまう。そこで選択直前にプール全体へ
        対称的に適用し、選択完了後は適応度を元の値へ復元する。これにより
        統計・Hall of Fame・次世代の共有計算が常に生の適応度を参照し、
        生存個体の適応度が世代をまたいで複利減衰することもなくなる。
        """
        candidate_population = self._apply_restricted_tournament(
            candidate_population,
            config,
            offspring=offspring,
            parents=parents,
        )

        if not (config.fitness_sharing.enable_fitness_sharing and self.fitness_sharing):
            return self._apply_two_stage_selection(
                candidate_population,
                population_size,
                config,
            )

        original_values: dict[int, tuple[float, ...]] = {}
        for individual in candidate_population:
            if has_valid_fitness(individual):
                original_values[id(individual)] = individual.fitness.values

        self.fitness_sharing.apply_fitness_sharing(
            candidate_population, objectives=getattr(config, "objectives", None)
        )
        try:
            return self._apply_two_stage_selection(
                candidate_population,
                population_size,
                config,
            )
        finally:
            for individual in candidate_population:
                values = original_values.get(id(individual))
                if values is not None and has_valid_fitness(individual):
                    individual.fitness.values = values

    def _apply_restricted_tournament(
        self,
        candidate_population: list[Any],
        config: "GAConfig",
        offspring: list[Any] | None,
        parents: list[Any] | None,
    ) -> list[Any]:
        """制限トーナメント置換で候補プールを構造ニッチごとに絞る。

        各子個体は生存プールからサンプリングした中で指標構成距離が最小の
        個体と比較され、ニッチ代表を支配した場合のみ置換・追加される。
        失敗した場合はそのままの候補プールで選択を続行する。
        """
        survival_config = getattr(config, "survival_selection_config", None)
        if not isinstance(survival_config, SurvivalSelectionConfig):
            return candidate_population
        if not survival_config.enable_restricted_tournament:
            return candidate_population
        if not parents or not offspring:
            return candidate_population

        try:
            survivors = restricted_tournament_replace(
                parents,
                offspring,
                crowding_factor=survival_config.restricted_tournament_crowding_factor,
            )
        except Exception as exc:
            logger.warning(
                "制限トーナメント置換に失敗したため通常の候補プールで選択します: %s",
                exc,
            )
            return candidate_population

        logger.debug(
            "RTR: parents=%d offspring=%d survivors=%d",
            len(parents),
            len(offspring),
            len(survivors),
        )
        return survivors

    def _apply_two_stage_selection(
        self,
        candidate_population: list[Any],
        population_size: int,
        config: "GAConfig",
    ) -> list[Any]:
        """二段階選抜を適用する（TwoStageSelection へ委譲）。"""
        return self._two_stage_selection.apply_two_stage_selection(
            candidate_population,
            population_size,
            config,
        )

    def _select_top_candidates(
        self,
        candidate_population: list[Any],
        limit: int,
    ) -> list[Any]:
        """主 fitness の上位候補を返す（TwoStageSelection へ委譲）。"""
        return self._two_stage_selection.select_top_candidates(
            candidate_population,
            limit,
        )

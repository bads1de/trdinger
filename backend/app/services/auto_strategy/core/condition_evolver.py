"""
ConditionEvolver - 条件進化最適化システム

32指標に対応した進化的最適化システムを提供します。
YAML設定ファイルから指標情報を読み込み、DEAPベースのGAエンジンで
条件の最適化を行います。

パフォーマンス最適化機能:
- 並列適応度評価（ParallelFitnessEvaluator）
- 適応度キャッシュ（FitnessCache）
- 早期打ち切り（EarlyStopping）
"""

import logging
import os
import random
import copy
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from app.services.backtest.backtest_service import BacktestService
from app.services.indicators.manifests.registry import manifest_to_yaml_dict
from app.utils.error_handler import safe_operation
from ..models import Condition, ConditionGroup

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """進化アルゴリズム設定"""

    population_size: int = 20
    generations: int = 10
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    tournament_size: int = 3
    objectives: List[str] = None  # ["total_return", "sharpe_ratio", "max_drawdown"]

    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ["total_return"]


class EarlyStopping:
    """早期打ち切り機能"""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_fitness = float("-inf")
        self.should_stop = False

    def update(self, fitness: float) -> bool:
        if fitness > self.best_fitness + self.min_delta:
            self.best_fitness = fitness
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return self.should_stop

    def reset(self):
        self.counter = 0
        self.best_fitness = float("-inf")
        self.should_stop = False


class FitnessCache:
    """適応度キャッシュ"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _make_key(self, condition: Union[Condition, ConditionGroup]) -> str:
        """条件からキャッシュキーを生成"""
        if isinstance(condition, ConditionGroup):
            # 再帰的にキーを生成
            sub_keys = [self._make_key(c) for c in condition.conditions]
            return f"GROUP|{condition.operator}|{'&'.join(sub_keys)}"
        else:
            # Conditionの場合
            # direction属性がある場合はそれも含める（動的属性）
            direction = getattr(condition, "direction", "unknown")
            return f"{condition.left_operand}|{condition.operator}|{condition.right_operand}|{direction}"

    def get(self, condition: Union[Condition, ConditionGroup]) -> Optional[float]:
        key = self._make_key(condition)
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, condition: Union[Condition, ConditionGroup], fitness: float):
        key = self._make_key(condition)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
        self._cache[key] = fitness

    def __len__(self) -> int:
        """キャッシュのサイズを返す"""
        return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
        }


class ParallelFitnessEvaluator:
    """並列適応度評価器"""

    def __init__(
        self,
        backtest_service: BacktestService,
        yaml_indicator_utils: "YamlIndicatorUtils",
        max_workers: Optional[int] = None,
        cache: Optional[FitnessCache] = None,
    ):
        self.backtest_service = backtest_service
        self.yaml_indicator_utils = yaml_indicator_utils
        self.max_workers = max_workers or min(os.cpu_count() or 4, 8)
        self.cache = cache

    def _evaluate_single(
        self,
        condition: Union[Condition, ConditionGroup],
        backtest_config: Dict[str, Any],
    ) -> float:
        if self.cache:
            cached = self.cache.get(condition)
            if cached is not None:
                return cached

        try:
            strategy_config = create_simple_strategy(condition)
            test_config = backtest_config.copy()
            test_config.update(strategy_config["parameters"])
            result = self.backtest_service.run_backtest(test_config)

            metrics = result.get("performance_metrics", {})
            total_return = metrics.get("total_return", 0.0)
            sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
            max_drawdown = metrics.get("max_drawdown", 1.0)
            total_trades = metrics.get("total_trades", 0)

            if total_trades < 10:
                fitness = 0.1
            else:
                fitness = (
                    0.4 * max(0, total_return)
                    + 0.3 * max(0, sharpe_ratio)
                    + 0.2 * max(0, (1 - max_drawdown))
                    + 0.1 * min(1, total_trades / 100)
                )
            fitness = max(0.0, fitness)

            if self.cache:
                self.cache.set(condition, fitness)
            return fitness
        except Exception as e:
            logger.error(f"適応度評価エラー: {e}")
            return 0.0

    def evaluate_population(
        self,
        population: List[Union[Condition, ConditionGroup]],
        backtest_config: Dict[str, Any],
    ) -> List[float]:
        fitness_values = [0.0] * len(population)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._evaluate_single, condition, backtest_config): i
                for i, condition in enumerate(population)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    fitness_values[index] = future.result()
                except Exception as e:
                    logger.error(f"並列評価エラー (index={index}): {e}")
                    fitness_values[index] = 0.0
        return fitness_values


class YamlIndicatorUtils:
    """YAML設定ファイルから指標情報を管理するユーティリティクラス"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config = self._load_yaml_config()
        self._validate_config()

    def _load_yaml_config(self) -> Dict[str, Any]:
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as file:
                    return yaml.safe_load(file)
            except Exception as exc:
                logger.warning("YAMLロード失敗: %s", exc)
        return manifest_to_yaml_dict()

    def _validate_config(self):
        if not self.config or "indicators" not in self.config:
            # 必須ではないが警告
            pass

    def get_available_indicators(self) -> List[str]:
        return list(self.config.get("indicators", {}).keys())

    def get_indicator_info(self, indicator_name: str) -> Dict[str, Any]:
        if indicator_name not in self.config.get("indicators", {}):
            raise ValueError(f"Unknown indicator: {indicator_name}")
        return self.config["indicators"][indicator_name].copy()


def create_simple_strategy(
    condition: Union[Condition, ConditionGroup],
) -> Dict[str, Any]:
    """条件からシンプルな戦略設定を作成"""

    # ネストされた構造を辞書形式に変換するヘルパー
    def _cond_to_dict(cond):
        if isinstance(cond, ConditionGroup):
            return {
                "type": "group",
                "operator": cond.operator,
                "conditions": [_cond_to_dict(c) for c in cond.conditions],
            }
        else:
            # Condition
            direction = getattr(cond, "direction", "long")
            return {
                "type": "single",
                "condition": {
                    "indicator": cond.left_operand,
                    "operator": cond.operator,
                    "threshold": cond.right_operand,
                    "direction": direction,
                },
            }

    cond_dict = _cond_to_dict(condition)

    # ルートが単一Conditionの場合のラッパー
    if cond_dict["type"] == "single":
        # 古いcreate_simple_strategyとの互換性
        entry_config = cond_dict
    else:
        entry_config = cond_dict

    return {
        "name": f"Strategy_{id(condition)}",
        "conditions": {"entry": entry_config},
        "parameters": {"symbol": "BTC/USDT:USDT", "timeframe": "1h"},
    }


class ConditionEvolver:
    """条件進化最適化のメインクラス"""

    def __init__(
        self,
        backtest_service: BacktestService,
        yaml_indicator_utils: YamlIndicatorUtils,
        enable_parallel: bool = False,
        max_workers: Optional[int] = None,
        enable_cache: bool = False,
        cache_size: int = 1000,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.001,
    ):
        self.backtest_service = backtest_service
        self.yaml_indicator_utils = yaml_indicator_utils
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers

        self.cache = FitnessCache(max_size=cache_size) if enable_cache else None
        self.early_stopping = (
            EarlyStopping(
                patience=early_stopping_patience, min_delta=early_stopping_min_delta
            )
            if early_stopping_patience
            else None
        )
        self.parallel_evaluator = (
            ParallelFitnessEvaluator(
                backtest_service, yaml_indicator_utils, max_workers, self.cache
            )
            if enable_parallel
            else None
        )
        logger.info(f"ConditionEvolver 初期化完了")

    def _create_individual(self) -> Condition:
        """単一のCondition個体を生成（動的direction属性付き）"""
        available_indicators = self.yaml_indicator_utils.get_available_indicators()
        if not available_indicators:
            raise ValueError("利用可能な指標がありません")

        indicator_name = random.choice(available_indicators)
        indicator_info = self.yaml_indicator_utils.get_indicator_info(indicator_name)

        direction = random.choice(["long", "short"])
        operators = [">", "<", ">=", "<=", "==", "!="]
        operator = random.choice(operators)
        threshold = self._generate_threshold(indicator_info)

        cond = Condition(
            left_operand=indicator_name, operator=operator, right_operand=threshold
        )
        # 動的にdirection属性を追加（進化コンテキスト用）
        cond.direction = direction
        return cond

    def _generate_threshold(self, indicator_info: Dict[str, Any]) -> float:
        scale_type = indicator_info.get("scale_type", "oscillator_0_100")
        thresholds = indicator_info.get("thresholds", {})

        if scale_type == "oscillator_0_100":
            base_range = (0, 100)
        elif scale_type == "oscillator_plus_minus_100":
            base_range = (-100, 100)
        elif scale_type == "momentum_zero_centered":
            base_range = (-1, 1)
        else:
            base_range = (0, 100)

        if thresholds and "normal" in thresholds:
            normal_thresholds = thresholds["normal"]
            if isinstance(normal_thresholds, dict):
                # 簡易的にどちらかを使用
                if "long_gt" in normal_thresholds:
                    return float(normal_thresholds["long_gt"])

        min_val, max_val = base_range
        return random.uniform(min_val, max_val)

    def generate_initial_population(self, population_size: int = 20) -> List[Condition]:
        return [self._create_individual() for _ in range(population_size)]

    @safe_operation(context="ConditionEvolver.evaluate_fitness", default_return=0.0)
    def evaluate_fitness(
        self,
        condition: Union[Condition, ConditionGroup],
        backtest_config: Dict[str, Any],
    ) -> float:
        if self.cache:
            cached = self.cache.get(condition)
            if cached is not None:
                return cached

        try:
            # 並列評価器と同じロジックを使用（再利用）
            evaluator = ParallelFitnessEvaluator(
                self.backtest_service, self.yaml_indicator_utils, cache=self.cache
            )
            return evaluator._evaluate_single(condition, backtest_config)

        except Exception as e:
            logger.error(f"適応度評価エラー: {e}")
            return 0.0

    def tournament_selection(
        self,
        population: List[Union[Condition, ConditionGroup]],
        fitness_values: List[float],
        k: int = 3,
    ) -> List[Union[Condition, ConditionGroup]]:
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(list(zip(population, fitness_values)), k)
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def crossover(
        self,
        parent1: Union[Condition, ConditionGroup],
        parent2: Union[Condition, ConditionGroup],
    ) -> Tuple[Union[Condition, ConditionGroup], Union[Condition, ConditionGroup]]:
        """多態性対応の交叉"""

        # 1. 両方がConditionGroupの場合
        if isinstance(parent1, ConditionGroup) and isinstance(parent2, ConditionGroup):
            # 構造が同じ（要素数が同じ）なら、子要素同士を交叉
            if len(parent1.conditions) == len(parent2.conditions):
                new_conds1 = []
                new_conds2 = []
                for c1, c2 in zip(parent1.conditions, parent2.conditions):
                    nc1, nc2 = self.crossover(c1, c2)
                    new_conds1.append(nc1)
                    new_conds2.append(nc2)

                child1 = ConditionGroup(
                    operator=parent1.operator, conditions=new_conds1
                )
                child2 = ConditionGroup(
                    operator=parent2.operator, conditions=new_conds2
                )
                return child1, child2
            else:
                # 構造が違う場合は交叉しない（コピーを返す）
                # またはサブツリー交換などが考えられるが、ここではシンプルに
                return copy.deepcopy(parent1), copy.deepcopy(parent2)

        # 2. 両方がConditionの場合
        elif isinstance(parent1, Condition) and isinstance(parent2, Condition):
            crossover_point = random.choice(["operator", "threshold"])

            # direction属性の維持（あれば）
            d1 = getattr(parent1, "direction", "long")
            d2 = getattr(parent2, "direction", "long")

            if crossover_point == "operator":
                child1 = Condition(
                    parent1.left_operand, parent2.operator, parent1.right_operand
                )
                child2 = Condition(
                    parent2.left_operand, parent1.operator, parent2.right_operand
                )
            else:
                # 閾値平均
                try:
                    avg = (
                        float(parent1.right_operand) + float(parent2.right_operand)
                    ) / 2
                except:
                    avg = parent1.right_operand  # 数値でない場合はそのまま

                child1 = Condition(parent1.left_operand, parent1.operator, avg)
                child2 = Condition(parent2.left_operand, parent2.operator, avg)

            child1.direction = d1
            child2.direction = d2
            return child1, child2

        # 3. 型が異なる場合
        else:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

    def mutate(
        self, condition: Union[Condition, ConditionGroup]
    ) -> Union[Condition, ConditionGroup]:
        """多態性対応の突然変異"""

        if isinstance(condition, ConditionGroup):
            # グループの場合: 子要素のどれかを変異させる
            mutated_group = copy.deepcopy(condition)
            if mutated_group.conditions:
                idx = random.randrange(len(mutated_group.conditions))
                mutated_group.conditions[idx] = self.mutate(
                    mutated_group.conditions[idx]
                )
            return mutated_group

        elif isinstance(condition, Condition):
            # Conditionの場合: 既存ロジック
            mutated = copy.deepcopy(condition)
            mutation_type = random.choice(["operator", "threshold", "indicator"])

            if mutation_type == "operator":
                operators = [">", "<", ">=", "<=", "==", "!="]
                if mutated.operator in operators:
                    operators.remove(mutated.operator)
                mutated.operator = random.choice(operators)

            elif mutation_type == "threshold":
                try:
                    val = float(mutated.right_operand)
                    variation = random.uniform(-0.1, 0.1)
                    mutated.right_operand = max(0, val * (1 + variation))
                except:
                    pass

            elif mutation_type == "indicator":
                # 指標変更
                available = self.yaml_indicator_utils.get_available_indicators()
                if mutated.left_operand in available:
                    available.remove(mutated.left_operand)
                if available:
                    new_ind = random.choice(available)
                    mutated.left_operand = new_ind
                    info = self.yaml_indicator_utils.get_indicator_info(new_ind)
                    mutated.right_operand = self._generate_threshold(info)

            return mutated

        return condition

    def run_evolution(
        self,
        backtest_config: Dict[str, Any],
        population_size: int = 20,
        generations: int = 10,
    ) -> Dict[str, Any]:
        """進化アルゴリズムを実行"""
        try:
            logger.info(f"進化開始: 個体数={population_size}, 世代数={generations}")

            if self.early_stopping:
                self.early_stopping.reset()

            population = self.generate_initial_population(population_size)
            evolution_history = []
            early_stopped = False
            generations_completed = 0

            for generation in range(generations):
                if self.enable_parallel and self.parallel_evaluator:
                    fitness_values = self.parallel_evaluator.evaluate_population(
                        population, backtest_config
                    )
                else:
                    fitness_values = [
                        self.evaluate_fitness(ind, backtest_config)
                        for ind in population
                    ]

                best_fitness = max(fitness_values)
                avg_fitness = sum(fitness_values) / len(fitness_values)
                evolution_history.append(
                    {
                        "generation": generation + 1,
                        "best_fitness": best_fitness,
                        "avg_fitness": avg_fitness,
                    }
                )

                generations_completed = generation + 1

                if self.early_stopping:
                    if self.early_stopping.update(best_fitness):
                        early_stopped = True
                        break

                elite = [max(zip(population, fitness_values), key=lambda x: x[1])[0]]
                selected = self.tournament_selection(population, fitness_values, k=3)

                offspring = []
                for i in range(0, len(selected) - 1, 2):
                    if random.random() < 0.8:
                        child1, child2 = self.crossover(selected[i], selected[i + 1])
                        offspring.extend([child1, child2])
                    else:
                        offspring.extend([selected[i], selected[i + 1]])

                offspring = [
                    self.mutate(ind) if random.random() < 0.2 else ind
                    for ind in offspring
                ]
                population = (elite + offspring)[:population_size]

            if self.enable_parallel and self.parallel_evaluator:
                fitness_values = self.parallel_evaluator.evaluate_population(
                    population, backtest_config
                )
            else:
                fitness_values = [
                    self.evaluate_fitness(ind, backtest_config) for ind in population
                ]

            best_idx = fitness_values.index(max(fitness_values))
            best_condition = population[best_idx]
            best_fitness = max(fitness_values)

            result = {
                "best_condition": best_condition,
                "best_fitness": best_fitness,
                "final_population": population,
                "evolution_history": evolution_history,
                "generations_completed": generations_completed,
                "parallel_enabled": self.enable_parallel,
                "early_stopped": early_stopped,
            }

            if self.cache:
                result["cache_stats"] = self.cache.get_stats()

            return result

        except Exception as e:
            logger.error(f"進化実行エラー: {e}")
            raise

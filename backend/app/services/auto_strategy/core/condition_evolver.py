"""
ConditionEvolver - 条件進化最適化システム

32指標に対応した進化的最適化システムを提供します。
YAML設定ファイルから指標情報を読み込み、DEAPベースのGAエンジンで
条件の最適化を行います。
"""

import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from deap import base, creator, tools

from app.services.backtest.backtest_service import BacktestService
from app.services.indicators.manifest import manifest_to_yaml_dict
from app.utils.error_handler import safe_operation

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


class YamlIndicatorUtils:
    """
    YAML設定ファイルから指標情報を管理するユーティリティクラス

    32指標の設定を読み込み、指標タイプ別の分類や閾値範囲の取得を行います。
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初期化

        Args:
            config_path: YAML設定ファイルのパス
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = self._load_yaml_config()
        self._validate_config()

    def _load_yaml_config(self) -> Dict[str, Any]:
        """YAML設定ファイルを読み込み"""
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as file:
                    return yaml.safe_load(file)
            except Exception as exc:
                logger.warning(
                    "YAML設定ファイルの読み込みに失敗したためメタデータ定義を使用します: %s",
                    exc,
                )

        return manifest_to_yaml_dict()

    def _validate_config(self):
        """設定ファイルの検証"""
        if not self.config:
            raise ValueError("設定ファイルが空です")

        if "indicators" not in self.config:
            raise ValueError("indicatorsセクションが見つかりません")

        required_keys = ["conditions", "scale_type", "type"]
        for indicator_name, indicator_config in self.config["indicators"].items():
            for key in required_keys:
                if key not in indicator_config:
                    logger.warning(
                        f"指標 '{indicator_name}' に '{key}' キーが見つかりません"
                    )

    def get_available_indicators(self) -> List[str]:
        """利用可能な指標リストを取得"""
        return list(self.config["indicators"].keys())

    def get_indicator_info(self, indicator_name: str) -> Dict[str, Any]:
        """
        指定された指標の詳細情報を取得

        Args:
            indicator_name: 指標名

        Returns:
            指標の設定情報

        Raises:
            ValueError: 指定された指標が存在しない場合
        """
        if indicator_name not in self.config["indicators"]:
            available = self.get_available_indicators()
            raise ValueError(
                f"指標 '{indicator_name}' が見つかりません。利用可能な指標: {available}"
            )

        return self.config["indicators"][indicator_name].copy()

    def get_indicator_types(self) -> Dict[str, List[str]]:
        """
        指標タイプ別の分類を取得

        Returns:
            指標タイプ（momentum, volatility, volume, trend）別の指標リスト
        """
        type_mapping = {
            "momentum": [],
            "volatility": [],
            "volume": [],
            "trend": [],
            "price_transform": [],
        }

        for indicator_name, config in self.config["indicators"].items():
            indicator_type = config.get("type", "unknown")
            if indicator_type in type_mapping:
                type_mapping[indicator_type].append(indicator_name)

        return type_mapping

    def get_threshold_ranges(self) -> Dict[str, Dict[str, Any]]:
        """
        閾値範囲を取得

        Returns:
            スケールタイプ別の閾値範囲設定
        """
        scale_types = self.config.get("scale_types", {})
        default_thresholds = self.config.get("default_thresholds", {})

        # デフォルト値で補完
        result = {}
        for scale_type, config in scale_types.items():
            result[scale_type] = {
                "range": config.get("range"),
                "defaults": default_thresholds.get(scale_type, {}),
            }

        return result

    def get_condition_format(self, indicator_name: str) -> Dict[str, str]:
        """
        指定された指標の条件形式を取得

        Args:
            indicator_name: 指標名

        Returns:
            long/short条件のフォーマット文字列
        """
        indicator_info = self.get_indicator_info(indicator_name)
        return indicator_info.get("conditions", {})

    def get_threshold_configs(self, indicator_name: str) -> Dict[str, Any]:
        """
        指定された指標の閾値設定を取得

        Args:
            indicator_name: 指標名

        Returns:
            閾値設定（aggressive, normal, conservative）
        """
        indicator_info = self.get_indicator_info(indicator_name)
        return indicator_info.get("thresholds", {})


@dataclass
class Condition:
    """
    最適化対象の条件を表現するデータクラス

    Attributes:
        indicator_name: 指標名
        operator: 演算子（>, <, >=, <=, ==, !=）
        threshold: 閾値
        direction: 方向（long, short）
    """

    indicator_name: str
    operator: str
    threshold: float
    direction: str

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Condition":
        """辞書形式からConditionを作成"""
        return cls(
            indicator_name=data["indicator_name"],
            operator=data["operator"],
            threshold=data["threshold"],
            direction=data["direction"],
        )

    def __eq__(self, other) -> bool:
        """等価性の判定"""
        if not isinstance(other, Condition):
            return False
        return (
            self.indicator_name == other.indicator_name
            and self.operator == other.operator
            and self.threshold == other.threshold
            and self.direction == other.direction
        )

    def __str__(self) -> str:
        """文字列表現"""
        return (
            f"{self.indicator_name} {self.operator} {self.threshold} ({self.direction})"
        )


def create_simple_strategy(condition: Condition) -> Dict[str, Any]:
    """
    単一の条件からシンプルな戦略設定を作成

    Args:
        condition: 条件オブジェクト

    Returns:
        戦略設定辞書
    """
    return {
        "name": f"Simple_{condition.indicator_name}_{condition.direction}",
        "conditions": {
            "entry": {
                "type": "single",
                "condition": {
                    "indicator": condition.indicator_name,
                    "operator": condition.operator,
                    "threshold": condition.threshold,
                    "direction": condition.direction,
                },
            }
        },
        "parameters": {"symbol": "BTC/USDT:USDT", "timeframe": "1h"},
    }


class ConditionEvolver:
    """
    条件進化最適化のメインクラス

    DEAPベースのGAエンジンを使用して、32指標全てに対応した
    条件の進化的最適化を行います。
    """

    def __init__(
        self,
        backtest_service: BacktestService,
        yaml_indicator_utils: YamlIndicatorUtils,
    ):
        """
        初期化

        Args:
            backtest_service: バックテストサービス
            yaml_indicator_utils: YAML指標ユーティリティ
        """
        self.backtest_service = backtest_service
        self.yaml_indicator_utils = yaml_indicator_utils

        # DEAP設定
        self._setup_deap()

        logger.info("ConditionEvolver 初期化完了")

    def _setup_deap(self):
        """DEAP環境のセットアップ"""
        # フィットネス関数（最大化）
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # 個体クラス
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # ツールボックス
        self.toolbox = base.Toolbox()

        # 個体生成関数
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # 交叉・突然変異関数
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", self.tournament_selection, k=3)

    def _create_individual(self) -> Condition:
        """
        個体（Condition）を生成

        Returns:
            ランダムに生成されたCondition
        """
        available_indicators = self.yaml_indicator_utils.get_available_indicators()
        if not available_indicators:
            raise ValueError("利用可能な指標がありません")

        # ランダムな指標を選択
        indicator_name = random.choice(available_indicators)
        indicator_info = self.yaml_indicator_utils.get_indicator_info(indicator_name)

        # 方向を決定
        direction = random.choice(["long", "short"])

        # 演算子を決定
        operators = [">", "<", ">=", "<=", "==", "!="]
        operator = random.choice(operators)

        # 閾値を決定
        threshold = self._generate_threshold(indicator_info)

        return Condition(
            indicator_name=indicator_name,
            operator=operator,
            threshold=threshold,
            direction=direction,
        )

    def _generate_threshold(self, indicator_info: Dict[str, Any]) -> float:
        """指標情報に基づいて閾値を生成"""
        scale_type = indicator_info.get("scale_type", "oscillator_0_100")
        thresholds = indicator_info.get("thresholds", {})

        # デフォルトの範囲設定
        if scale_type == "oscillator_0_100":
            base_range = (0, 100)
        elif scale_type == "oscillator_plus_minus_100":
            base_range = (-100, 100)
        elif scale_type == "momentum_zero_centered":
            base_range = (-1, 1)
        else:
            base_range = (0, 100)  # デフォルト

        # 閾値設定がある場合はそれを使用
        if thresholds and "normal" in thresholds:
            normal_thresholds = thresholds["normal"]
            if isinstance(normal_thresholds, dict):
                # 方向に応じた閾値を取得
                if "long_gt" in normal_thresholds:
                    return float(normal_thresholds["long_gt"])
                elif "short_lt" in normal_thresholds:
                    return float(normal_thresholds["short_lt"])

        # 範囲内のランダム値を生成
        min_val, max_val = base_range
        return random.uniform(min_val, max_val)

    def generate_initial_population(self, population_size: int = 20) -> List[Condition]:
        """
        初期個体群を生成

        Args:
            population_size: 個体群サイズ

        Returns:
            Conditionのリスト
        """
        return [self._create_individual() for _ in range(population_size)]

    @safe_operation(context="ConditionEvolver.evaluate_fitness", default_return=0.0)
    def evaluate_fitness(
        self, condition: Condition, backtest_config: Dict[str, Any]
    ) -> float:
        """
        個体の適応度を評価

        Args:
            condition: 評価する条件
            backtest_config: バックテスト設定

        Returns:
            適応度値（高いほど良い）
        """
        try:
            # 条件から戦略設定を作成
            strategy_config = create_simple_strategy(condition)

            # バックテスト設定を構築
            test_config = backtest_config.copy()
            test_config.update(strategy_config["parameters"])

            # バックテスト実行
            result = self.backtest_service.run_backtest(test_config)

            # パフォーマンスメトリクスから適応度を計算
            metrics = result.get("performance_metrics", {})
            total_return = metrics.get("total_return", 0.0)
            sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
            max_drawdown = metrics.get("max_drawdown", 1.0)
            total_trades = metrics.get("total_trades", 0)

            # 取引回数が少ない場合はペナルティ
            if total_trades < 10:
                return 0.1

            # 多目的適応度を単一値に統合（重み付け）
            fitness = (
                0.4 * max(0, total_return)
                + 0.3 * max(0, sharpe_ratio)
                + 0.2 * max(0, (1 - max_drawdown))
                + 0.1 * min(1, total_trades / 100)  # 取引回数ボーナス
            )

            logger.debug(f"Condition {condition} fitness: {fitness:.4f}")
            return max(0.0, fitness)

        except Exception as e:
            logger.error(f"適応度評価エラー: {e}")
            return 0.0

    def tournament_selection(
        self, population: List[Condition], fitness_values: List[float], k: int = 3
    ) -> List[Condition]:
        """
        トーナメント選択

        Args:
            population: 個体群
            fitness_values: 適応度値リスト
            k: トーナメントサイズ

        Returns:
            選択された個体群
        """
        selected = []
        for _ in range(len(population)):
            # トーナメント参加者を選択
            tournament = random.sample(list(zip(population, fitness_values)), k)
            # 最も適応度の高い個体を選択
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def crossover(
        self, parent1: Condition, parent2: Condition
    ) -> Tuple[Condition, Condition]:
        """
        一点交叉

        Args:
            parent1: 親個体1
            parent2: 親個体2

        Returns:
            子個体2つ
        """
        # ランダムな交叉点を決定
        crossover_point = random.choice(["operator", "threshold"])

        if crossover_point == "operator":
            # 演算子を交叉
            child1 = Condition(
                indicator_name=parent1.indicator_name,
                operator=parent2.operator,
                threshold=parent1.threshold,
                direction=parent1.direction,
            )
            child2 = Condition(
                indicator_name=parent2.indicator_name,
                operator=parent1.operator,
                threshold=parent2.threshold,
                direction=parent2.direction,
            )
        else:
            # 閾値を交叉（平均値を使用）
            new_threshold = (parent1.threshold + parent2.threshold) / 2
            child1 = Condition(
                indicator_name=parent1.indicator_name,
                operator=parent1.operator,
                threshold=new_threshold,
                direction=parent1.direction,
            )
            child2 = Condition(
                indicator_name=parent2.indicator_name,
                operator=parent2.operator,
                threshold=new_threshold,
                direction=parent2.direction,
            )

        return child1, child2

    def mutate(self, condition: Condition) -> Condition:
        """
        突然変異

        Args:
            condition: 変異対象の条件

        Returns:
            変異後の条件
        """
        mutated_condition = Condition(
            indicator_name=condition.indicator_name,
            operator=condition.operator,
            threshold=condition.threshold,
            direction=condition.direction,
        )

        # 突然変異の種類を決定
        mutation_type = random.choice(["operator", "threshold", "indicator"])

        if mutation_type == "operator":
            # 演算子を突然変異
            operators = [">", "<", ">=", "<=", "==", "!="]
            current_op = mutated_condition.operator
            operators.remove(current_op)  # 現在の演算子を除外
            mutated_condition.operator = random.choice(operators)

        elif mutation_type == "threshold":
            # 閾値を突然変異（10%程度の変動）
            variation = random.uniform(-0.1, 0.1)
            mutated_condition.threshold *= 1 + variation
            mutated_condition.threshold = max(
                0, mutated_condition.threshold
            )  # 負値防止

        elif mutation_type == "indicator":
            # 指標を突然変異
            available_indicators = self.yaml_indicator_utils.get_available_indicators()
            available_indicators.remove(mutated_condition.indicator_name)
            if available_indicators:
                mutated_condition.indicator_name = random.choice(available_indicators)

                # 新しい指標の閾値設定を取得
                indicator_info = self.yaml_indicator_utils.get_indicator_info(
                    mutated_condition.indicator_name
                )
                mutated_condition.threshold = self._generate_threshold(indicator_info)

        return mutated_condition

    def run_evolution(
        self,
        backtest_config: Dict[str, Any],
        population_size: int = 20,
        generations: int = 10,
    ) -> Dict[str, Any]:
        """
        進化アルゴリズムを実行

        Args:
            population_size: 個体群サイズ
            generations: 世代数
            backtest_config: バックテスト設定

        Returns:
            進化結果
        """
        try:
            logger.info(f"進化開始: 個体数={population_size}, 世代数={generations}")

            # 初期個体群生成
            population = self.generate_initial_population(population_size)

            # 進化履歴
            evolution_history = []

            for generation in range(generations):
                logger.info(f"世代 {generation + 1}/{generations}")

                # 適応度評価
                fitness_values = [
                    self.evaluate_fitness(individual, backtest_config)
                    for individual in population
                ]

                # 進化履歴を記録
                best_fitness = max(fitness_values)
                avg_fitness = sum(fitness_values) / len(fitness_values)
                evolution_history.append(
                    {
                        "generation": generation + 1,
                        "best_fitness": best_fitness,
                        "avg_fitness": avg_fitness,
                    }
                )

                logger.info(
                    f"世代 {generation + 1}: 最高適応度={best_fitness:.4f}, 平均適応度={avg_fitness:.4f}"
                )

                # エリート選択（最良個体を保存）
                elite = [max(zip(population, fitness_values), key=lambda x: x[1])[0]]

                # 選択
                selected = self.tournament_selection(population, fitness_values, k=3)

                # 交叉・突然変異による次世代生成
                offspring = []
                for i in range(0, len(selected) - 1, 2):
                    if random.random() < 0.8:  # 交叉確率80%
                        child1, child2 = self.crossover(selected[i], selected[i + 1])
                        offspring.extend([child1, child2])
                    else:
                        offspring.extend([selected[i], selected[i + 1]])

                # 突然変異
                offspring = [
                    self.mutate(ind) if random.random() < 0.2 else ind
                    for ind in offspring
                ]

                # エリートを追加
                offspring.extend(elite)

                # 次世代を現在の個体群とする
                population = offspring[:population_size]

            # 最終結果
            fitness_values = [
                self.evaluate_fitness(individual, backtest_config)
                for individual in population
            ]

            best_idx = fitness_values.index(max(fitness_values))
            best_condition = population[best_idx]
            best_fitness = max(fitness_values)

            result = {
                "best_condition": best_condition,
                "best_fitness": best_fitness,
                "final_population": population,
                "evolution_history": evolution_history,
                "generations_completed": generations,
            }

            logger.info(
                f"進化完了: 最高適応度={best_fitness:.4f}, 最高条件={best_condition}"
            )
            return result

        except Exception as e:
            logger.error(f"進化実行エラー: {e}")
            raise

    def create_strategy_from_condition(self, condition: Condition) -> Dict[str, Any]:
        """
        条件から戦略設定を作成

        Args:
            condition: 条件オブジェクト

        Returns:
            戦略設定辞書
        """
        return create_simple_strategy(condition)

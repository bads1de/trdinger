"""
DEAP設定器

DEAPライブラリの環境セットアップを担当するモジュール。
フィットネスクラス、個体クラス、ツールボックスの設定を行います。
"""

import random
import multiprocessing
import logging
from typing import Optional, Callable
import numpy as np

from deap import base, creator, tools

from ..models.strategy_gene import encode_gene_to_list
from ..models.ga_config import GAConfig
from ..generators.random_gene_generator import RandomGeneGenerator

logger = logging.getLogger(__name__)


class DEAPConfigurator:
    """
    DEAP設定器

    DEAPライブラリの環境セットアップを担当します。
    """

    def __init__(self, gene_generator: RandomGeneGenerator):
        """
        初期化

        Args:
            gene_generator: ランダム遺伝子生成器
        """
        self.gene_generator = gene_generator
        self.toolbox: Optional[base.Toolbox] = None
        self.stats: Optional[tools.Statistics] = None
        self.logbook: Optional[tools.Logbook] = None

    def setup_deap_environment(
        self, config: GAConfig, evaluation_function: Callable
    ) -> base.Toolbox:
        """
        DEAP環境のセットアップ

        Args:
            config: GA設定
            evaluation_function: 評価関数

        Returns:
            設定済みのツールボックス
        """
        # フィットネスクラスの定義（最大化問題）
        self._setup_fitness_class()

        # 個体クラスの定義
        self._setup_individual_class()

        # ツールボックスの初期化
        self.toolbox = base.Toolbox()
        assert self.toolbox is not None, "Toolbox should be initialized at this point."

        # 個体生成関数の登録
        self._register_individual_creation(config)

        # 評価関数の登録（configを含むラッパー）
        def evaluate_with_config(individual):
            return evaluation_function(individual, config)

        self.toolbox.register("evaluate", evaluate_with_config)

        # 進化演算子の登録
        self._register_evolution_operators()

        # 並列処理の設定
        self._setup_parallel_processing(config)

        # 統計情報の設定
        self._setup_statistics()

        logger.info("DEAP環境のセットアップ完了")
        return self.toolbox

    def _setup_fitness_class(self):
        """フィットネスクラスの定義"""
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    def _setup_individual_class(self):
        """個体クラスの定義"""
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)  # type: ignore

    def _register_individual_creation(self, config: GAConfig):
        """個体生成関数の登録"""

        def create_individual():
            """新しいランダム遺伝子生成器を使用して個体を生成"""
            try:
                # ランダム遺伝子生成器で戦略遺伝子を生成
                gene = self.gene_generator.generate_random_gene()

                # 戦略遺伝子を数値リストにエンコード
                individual = encode_gene_to_list(gene)

                return creator.Individual(individual)  # type: ignore
            except Exception as e:
                logger.warning(f"新しい遺伝子生成に失敗、フォールバックを使用: {e}")
                # フォールバック: 従来の方法
                return self._create_fallback_individual(config)

        self.toolbox.register("individual", create_individual)
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual  # type: ignore
        )

    def _create_fallback_individual(self, config: GAConfig):
        """フォールバック個体生成"""
        individual = []

        # 指標部分（最低1個の指標を保証）
        for i in range(config.max_indicators):
            if i == 0:
                # 最初の指標は必ず有効にする
                indicator_id = random.uniform(0.1, 0.9)  # 0を避ける
            else:
                # 他の指標は50%の確率で有効
                indicator_id = random.uniform(0.0, 1.0)

            param_val = random.uniform(0.0, 1.0)
            individual.extend([indicator_id, param_val])

        # 条件部分
        for _ in range(6):  # エントリー3 + エグジット3
            individual.append(random.uniform(0.0, 1.0))

        return creator.Individual(individual)  # type: ignore

    def _register_evolution_operators(self):
        """進化演算子の登録"""
        # 交叉関数: 2つの個体から子孫を生成する関数
        self.toolbox.register("mate", tools.cxTwoPoint)

        # 突然変異関数: 個体の遺伝子にランダムな変更を加える関数
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)

        # 選択関数: 次世代の個体群を選択する関数
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _setup_parallel_processing(self, config: GAConfig):
        """並列処理の設定"""
        if config.parallel_processes:
            pool = multiprocessing.Pool(config.parallel_processes)
            self.toolbox.register("map", pool.map)

    def _setup_statistics(self):
        """統計情報の設定"""
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        # ログブックの初期化
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "std", "min", "avg", "max"  # type: ignore

    def get_statistics(self) -> Optional[tools.Statistics]:
        """統計情報を取得"""
        return self.stats

    def get_logbook(self) -> Optional[tools.Logbook]:
        """ログブックを取得"""
        return self.logbook

    def apply_constraints_decorator(self, func):
        """
        制約条件を適用するデコレータ

        Args:
            func: デコレートする関数

        Returns:
            デコレートされた関数
        """

        def wrapper(*args, **kwargs):
            # 元の関数 (mate または mutate) を実行し、結果を取得
            result = func(*args, **kwargs)
            # ここに具体的な制約条件のロジックを実装可能
            # 現状はデコレータとして機能するが、具体的な制約ロジックは未実装
            return result

        return wrapper

    def decorate_operators_with_constraints(self):
        """演算子に制約条件を適用"""
        if self.toolbox:
            self.toolbox.decorate("mate", self.apply_constraints_decorator)
            self.toolbox.decorate("mutate", self.apply_constraints_decorator)

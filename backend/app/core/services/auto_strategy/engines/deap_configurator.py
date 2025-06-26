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
        if not self.toolbox:
            return

        def create_individual():
            """
            新しいランダム遺伝子生成器を使用して個体を生成します。
            生成に失敗した場合は、フォールバックとして固定の構造を持つ個体を生成します。
            """
            try:
                # ランダム遺伝子生成器で戦略遺伝子を生成
                gene = self.gene_generator.generate_random_gene()

                # 戦略遺伝子を数値リストにエンコード
                individual = encode_gene_to_list(gene)

                logger.debug(
                    f"正常に遺伝子生成: 指標数={len(gene.indicators)}, 遺伝子長={len(individual)}"
                )
                return creator.Individual(individual)  # type: ignore
            except Exception as e:
                logger.warning(
                    f"新しい遺伝子生成に失敗、フォールバックを使用: {e}", exc_info=True
                )
                # フォールバック: 従来の方法（固定の構造を持つ個体を生成）
                fallback_individual = self._create_fallback_individual(config)

                # フォールバック個体にメタデータを追加
                if hasattr(fallback_individual, "metadata"):
                    fallback_individual.metadata = {
                        "source": "fallback_individual",
                        "error": str(e),
                    }

                return fallback_individual

        self.toolbox.register("individual", create_individual)
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual  # type: ignore
        )

    def _create_fallback_individual(self, config: GAConfig):
        """フォールバック個体生成（多様性を確保）"""
        individual = []

        # 指標部分（より多様な指標を生成）
        for i in range(config.max_indicators):
            if i == 0:
                # 最初の指標は必ず有効にし、より広い範囲から選択
                indicator_id = random.uniform(0.1, 0.95)  # より広い範囲
            elif i < 3:
                # 最初の3つは高確率で有効
                indicator_id = (
                    random.uniform(0.05, 0.9) if random.random() < 0.8 else 0.0
                )
            else:
                # 残りは30%の確率で有効
                indicator_id = (
                    random.uniform(0.1, 0.9) if random.random() < 0.3 else 0.0
                )

            # パラメータ値もより多様に
            param_val = random.uniform(0.1, 0.9)
            individual.extend([indicator_id, param_val])

        # 条件部分（より多様な条件を生成）
        for i in range(6):  # エントリー3 + エグジット3
            if i < 3:  # エントリー条件
                individual.append(random.uniform(0.2, 0.8))
            else:  # エグジット条件
                individual.append(random.uniform(0.1, 0.9))

        logger.info(
            f"フォールバック個体生成: 指標数={config.max_indicators}, 遺伝子長={len(individual)}"
        )
        return creator.Individual(individual)  # type: ignore

    def _register_evolution_operators(self):
        """進化演算子の登録"""
        if not self.toolbox:
            return
        # 交叉関数: 2つの個体から子孫を生成する関数 (ここでは2点交叉を使用)
        self.toolbox.register("mate", tools.cxTwoPoint)

        # 突然変異関数: 個体の遺伝子にランダムな変更を加える関数 (ここではガウス変異を使用)
        # mu: 平均 (変異の中心), sigma: 標準偏差 (変異の幅), indpb: 各遺伝子が変異する確率
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)

        # 選択関数: 次世代の個体群を選択する関数 (ここではトーナメント選択を使用)
        # tournsize: トーナメントサイズ (選択される個体数)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _setup_parallel_processing(self, config: GAConfig):
        """並列処理の設定"""
        if not self.toolbox:
            return
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

"""
DEAP設定

DEAPライブラリの設定とツールボックスの初期化を担当します。
"""

import logging
from typing import Optional

from deap import base, creator, tools

from ..models.ga_config import GAConfig

logger = logging.getLogger(__name__)


class DEAPSetup:
    """
    DEAP設定クラス

    DEAPライブラリの設定とツールボックスの初期化を担当します。
    """

    def __init__(self):
        """初期化"""
        self.toolbox: Optional[base.Toolbox] = None
        self.Individual = None

    def setup_deap(
        self,
        config: GAConfig,
        create_individual_func,
        evaluate_func,
        crossover_func,
        mutate_func,
    ):
        """
        DEAP環境のセットアップ（多目的最適化専用）

        Args:
            config: GA設定
            create_individual_func: 個体生成関数
            evaluate_func: 評価関数
            crossover_func: 交叉関数
            mutate_func: 突然変異関数
        """
        # 多目的最適化用フィットネスクラスの定義
        fitness_class_name = "FitnessMulti"
        weights = tuple(config.objective_weights)
        logger.info(f"多目的最適化モード: 目的={config.objectives}, 重み={weights}")

        # 既存のフィットネスクラスを削除（再定義のため）
        if hasattr(creator, fitness_class_name):
            delattr(creator, fitness_class_name)

        # フィットネスクラスを作成
        creator.create(fitness_class_name, base.Fitness, weights=weights)
        fitness_class = getattr(creator, fitness_class_name)

        # 個体クラスの定義
        if hasattr(creator, "Individual"):
            delattr(creator, "Individual")
        creator.create("Individual", list, fitness=fitness_class)  # type: ignore
        self.Individual = creator.Individual  # type: ignore # 生成したクラスをインスタンス変数に格納

        # ツールボックスの初期化
        self.toolbox = base.Toolbox()

        # 個体生成関数の登録
        self.toolbox.register("individual", create_individual_func)
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual  # type: ignore
        )

        # 評価関数の登録
        self.toolbox.register("evaluate", evaluate_func, config=config)

        # 進化演算子の登録（戦略遺伝子レベル）
        self.toolbox.register("mate", crossover_func)

        # 突然変異の登録（DEAP互換の返り値 (ind,) を保証するラッパー）
        def _mutate_wrapper(individual):
            res = mutate_func(individual, mutation_rate=config.mutation_rate)
            if isinstance(res, tuple):
                return res
            return (res,)

        self.toolbox.register("mutate", _mutate_wrapper)
        # 多目的最適化用選択アルゴリズムの登録（NSGA-II）
        self.toolbox.register("select", tools.selNSGA2)
        logger.info("NSGA-II選択アルゴリズムを登録")

        logger.info("DEAP環境のセットアップ完了")

    def get_toolbox(self) -> Optional[base.Toolbox]:
        """ツールボックスを取得"""
        return self.toolbox

    def get_individual_class(self):
        """個体クラスを取得"""
        return self.Individual

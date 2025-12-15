"""
DEAP設定

DEAPライブラリの設定とツールボックスの初期化を担当します。
"""

import logging
from typing import Optional

from deap import base, creator, tools

from ..config import GAConfig

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

        from ..genes import StrategyGene

        # StrategyGeneを継承し、fitness属性を持つクラスを作成
        creator.create("Individual", StrategyGene, fitness=fitness_class)  # type: ignore
        self.Individual = creator.Individual  # type: ignore # 生成したクラスをインスタンス変数に格納

        # ツールボックスの初期化
        self.toolbox = base.Toolbox()

        # 個体生成関数の登録
        self.toolbox.register("individual", create_individual_func)
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual,  # type: ignore
        )

        # 評価関数の登録
        self.toolbox.register("evaluate", evaluate_func, config=config)

        # 進化演算子の登録（戦略遺伝子レベル）
        self.toolbox.register("mate", crossover_func, config=config)

        # 突然変異の登録（DEAP互換の返り値 (ind,) を保証するラッパー）
        def _mutate_wrapper(individual):
            res = mutate_func(individual, mutation_rate=config.mutation_rate)
            if isinstance(res, tuple):
                return res
            return (res,)

        self.toolbox.register("mutate", _mutate_wrapper)

        # 選択アルゴリズムの登録（目的数に応じて切り替え）
        if config.enable_multi_objective:
            self.toolbox.register("select", tools.selNSGA2)
            logger.info("多目的最適化モード: NSGA-II選択アルゴリズムを登録")
        else:
            # 単一目的の場合はトーナメント選択（デフォルトサイズ3）
            tourn_size = getattr(config, "tournament_size", 3)
            self.toolbox.register("select", tools.selTournament, tournsize=tourn_size)
            logger.info(
                f"単一目的最適化モード: トーナメント選択アルゴリズム(size={tourn_size})を登録"
            )

        logger.info("DEAP環境のセットアップ完了")

    def get_toolbox(self) -> Optional[base.Toolbox]:
        """ツールボックスを取得"""
        return self.toolbox

    def get_individual_class(self):
        """個体クラスを取得"""
        return self.Individual






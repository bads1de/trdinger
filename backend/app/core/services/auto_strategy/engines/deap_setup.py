"""
DEAP設定

DEAPライブラリの設定とツールボックスの初期化を担当します。
"""

import logging
from typing import Optional
from deap import base, creator, tools

from ..models.ga_config import GAConfig
from ..operators import genetic_operators

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
        DEAP環境のセットアップ（単一目的・多目的最適化対応）

        Args:
            config: GA設定
            create_individual_func: 個体生成関数
            evaluate_func: 評価関数
            crossover_func: 交叉関数
            mutate_func: 突然変異関数
        """
        # フィットネスクラスの定義（単一目的・多目的対応）
        if config.enable_multi_objective:
            # 多目的最適化用フィットネスクラス
            fitness_class_name = "FitnessMulti"
            weights = tuple(config.objective_weights)
            logger.info(f"多目的最適化モード: 目的={config.objectives}, 重み={weights}")
        else:
            # 単一目的最適化用フィットネスクラス（後方互換性）
            fitness_class_name = "FitnessMax"
            weights = (1.0,)
            logger.info("単一目的最適化モード")

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

        # 突然変異の登録（ショートバイアス対応）
        if config.enable_short_bias_mutation:
            # ショートバイアス付き突然変異
            self.toolbox.register(
                "mutate",
                genetic_operators.mutate_with_short_bias,
                mutation_rate=config.mutation_rate,
                short_bias_rate=config.short_bias_rate,
            )
        else:
            # 通常の突然変異
            self.toolbox.register(
                "mutate", mutate_func, mutation_rate=config.mutation_rate
            )
        # 選択アルゴリズムの登録（単一目的・多目的対応）
        if config.enable_multi_objective:
            # 多目的最適化用選択（NSGA-II）
            self.toolbox.register("select", tools.selNSGA2)
            logger.info("NSGA-II選択アルゴリズムを登録")
        else:
            # 単一目的最適化用選択（トーナメント選択）
            self.toolbox.register("select", tools.selTournament, tournsize=3)
            logger.info("トーナメント選択アルゴリズムを登録")

        logger.info("DEAP環境のセットアップ完了")

    def get_toolbox(self) -> Optional[base.Toolbox]:
        """ツールボックスを取得"""
        return self.toolbox

    def get_individual_class(self):
        """個体クラスを取得"""
        return self.Individual

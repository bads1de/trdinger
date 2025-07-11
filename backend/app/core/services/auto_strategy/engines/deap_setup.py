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

    def setup_deap(self, config: GAConfig, create_individual_func, evaluate_func, crossover_func, mutate_func):
        """
        DEAP環境のセットアップ

        Args:
            config: GA設定
            create_individual_func: 個体生成関数
            evaluate_func: 評価関数
            crossover_func: 交叉関数
            mutate_func: 突然変異関数
        """
        # フィットネスクラスの定義（最大化問題）
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # 個体クラスの定義
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)  # type: ignore
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
                evolution_operators.mutate_with_short_bias,
                mutation_rate=config.mutation_rate,
                short_bias_rate=config.short_bias_rate
            )
        else:
            # 通常の突然変異
            self.toolbox.register(
                "mutate", mutate_func, mutation_rate=config.mutation_rate
            )
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        logger.info("DEAP環境のセットアップ完了")

    def get_toolbox(self) -> Optional[base.Toolbox]:
        """ツールボックスを取得"""
        return self.toolbox

    def get_individual_class(self):
        """個体クラスを取得"""
        return self.Individual

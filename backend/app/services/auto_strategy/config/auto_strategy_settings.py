"""
Auto Strategy 環境変数設定

遺伝的アルゴリズムによる自動戦略生成のデフォルトパラメータを
環境変数から読み込みます。
unified_config.py 経由でアプリケーション全体に提供されます。
"""

from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AutoStrategyConfig(BaseSettings):
    """自動戦略生成設定。

    遺伝的アルゴリズムによる自動戦略生成の各種パラメータを設定します。
    """

    # 遺伝的アルゴリズム基本設定
    population_size: int = Field(default=50, description="個体数")
    generations: int = Field(default=20, description="世代数")
    tournament_size: int = Field(default=3, description="トーナメントサイズ")
    crossover_rate: float = Field(default=0.8, description="交叉率")
    mutation_rate: float = Field(default=0.1, description="突然変異率")
    elite_size: int = Field(default=5, description="エリート保存数")

    # 戦略生成制約
    max_indicators: int = Field(default=5, description="最大指標数")
    min_indicators: int = Field(default=2, description="最小指標数")
    max_conditions: int = Field(default=5, description="最大条件数")
    min_conditions: int = Field(default=2, description="最小条件数")

    # 多目的最適化設定
    enable_multi_objective: bool = Field(
        default=False, description="多目的最適化を有効にするか"
    )
    objectives: List[str] = Field(
        default=["total_return"], description="最適化対象の指標"
    )
    objective_weights: List[float] = Field(default=[1.0], description="各指標の重み")

    # フィットネス共有設定
    enable_fitness_sharing: bool = Field(
        default=False, description="フィットネス共有を有効にするか"
    )
    fitness_sharing_radius: float = Field(default=0.1, description="共有半径")
    sharing_alpha: float = Field(default=1.0, description="共有アルファ")

    # フォールバック設定（GA実行時のデフォルト値）
    fallback_symbol: str = Field(
        default="BTC/USDT:USDT", description="フォールバックシンボル"
    )
    fallback_timeframe: str = Field(default="1d", description="フォールバック時間足")
    fallback_start_date: str = Field(
        default="2024-01-01", description="フォールバック開始日"
    )
    fallback_end_date: str = Field(
        default="2024-04-09", description="フォールバック終了日"
    )
    fallback_initial_capital: float = Field(
        default=100000.0, description="フォールバック初期資金"
    )
    fallback_commission_rate: float = Field(
        default=0.001, description="フォールバック手数料率"
    )

    # 戦略API設定
    default_strategies_limit: int = Field(
        default=20, description="戦略取得デフォルト件数"
    )
    max_strategies_limit: int = Field(default=100, description="戦略取得最大件数")

    model_config = SettingsConfigDict(env_prefix="AUTO_STRATEGY_", extra="ignore")

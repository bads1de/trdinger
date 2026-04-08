"""
Auto Strategy 環境変数設定

AutoStrategyConfig は環境変数経由でアプリケーション全体の設定を提供する pydantic モデルです。
unified_config.py 経由で使用され、主に API 層やサービス初期化時に参照されます。

注意: GAConfig (ga.py) は GA 実行時のランタイム設定用 dataclass です。
両者は役割が異なりますが、基本的なGAパラメータのデフォルト値は
ga_constants.GA_DEFAULT_CONFIG を共有しています。
"""

from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import (
    AUTO_STRATEGY_CONFIG_DEFAULTS,
    GA_DEFAULT_CONFIG,
)


class AutoStrategyConfig(BaseSettings):
    """自動戦略生成設定（環境変数ベース）。

    遺伝的アルゴリズムによる自動戦略生成の各種パラメータを設定します。
    環境変数 `AUTO_STRATEGY_*` から値を読み取ります。
    """

    # 遺伝的アルゴリズム基本設定（デフォルト値は GA_DEFAULT_CONFIG と同期）
    population_size: int = Field(
        default=int(GA_DEFAULT_CONFIG["population_size"]), description="個体数"
    )
    generations: int = Field(
        default=int(GA_DEFAULT_CONFIG["generations"]), description="世代数"
    )
    tournament_size: int = Field(
        default=int(AUTO_STRATEGY_CONFIG_DEFAULTS["tournament_size"]),
        description="トーナメントサイズ",
    )
    crossover_rate: float = Field(
        default=GA_DEFAULT_CONFIG["crossover_rate"], description="交叉率"
    )
    mutation_rate: float = Field(
        default=GA_DEFAULT_CONFIG["mutation_rate"], description="突然変異率"
    )
    elite_size: int = Field(
        default=int(GA_DEFAULT_CONFIG["elite_size"]), description="エリート保存数"
    )

    # 戦略生成制約
    max_indicators: int = Field(
        default=int(GA_DEFAULT_CONFIG["max_indicators"]), description="最大指標数"
    )
    min_indicators: int = Field(
        default=int(AUTO_STRATEGY_CONFIG_DEFAULTS["min_indicators"]),
        description="最小指標数",
    )
    max_conditions: int = Field(
        default=int(AUTO_STRATEGY_CONFIG_DEFAULTS["max_conditions"]),
        description="最大条件数",
    )
    min_conditions: int = Field(
        default=int(AUTO_STRATEGY_CONFIG_DEFAULTS["min_conditions"]),
        description="最小条件数",
    )

    # 多目的最適化設定
    enable_multi_objective: bool = Field(
        default=AUTO_STRATEGY_CONFIG_DEFAULTS["enable_multi_objective"],
        description="多目的最適化を有効にするか",
    )
    objectives: List[str] = Field(
        default=AUTO_STRATEGY_CONFIG_DEFAULTS["objectives"],
        description="最適化対象の指標",
    )
    objective_weights: List[float] = Field(
        default=AUTO_STRATEGY_CONFIG_DEFAULTS["objective_weights"],
        description="各指標の重み",
    )

    # フィットネス共有設定
    enable_fitness_sharing: bool = Field(
        default=AUTO_STRATEGY_CONFIG_DEFAULTS["enable_fitness_sharing"],
        description="フィットネス共有を有効にするか",
    )
    fitness_sharing_radius: float = Field(
        default=AUTO_STRATEGY_CONFIG_DEFAULTS["fitness_sharing_radius"],
        description="共有半径",
    )
    sharing_alpha: float = Field(
        default=AUTO_STRATEGY_CONFIG_DEFAULTS["sharing_alpha"],
        description="共有アルファ",
    )

    # フォールバック設定（GA実行時のデフォルト値）
    fallback_symbol: str = Field(
        default=AUTO_STRATEGY_CONFIG_DEFAULTS["fallback_symbol"],
        description="フォールバックシンボル",
    )
    fallback_timeframe: str = Field(
        default=AUTO_STRATEGY_CONFIG_DEFAULTS["fallback_timeframe"],
        description="フォールバック時間足",
    )
    fallback_start_date: str = Field(
        default=AUTO_STRATEGY_CONFIG_DEFAULTS["fallback_start_date"],
        description="フォールバック開始日",
    )
    fallback_end_date: str = Field(
        default=AUTO_STRATEGY_CONFIG_DEFAULTS["fallback_end_date"],
        description="フォールバック終了日",
    )
    fallback_initial_capital: float = Field(
        default=AUTO_STRATEGY_CONFIG_DEFAULTS["fallback_initial_capital"],
        description="フォールバック初期資金",
    )
    fallback_commission_rate: float = Field(
        default=AUTO_STRATEGY_CONFIG_DEFAULTS["fallback_commission_rate"],
        description="フォールバック手数料率",
    )

    # 戦略API設定
    default_strategies_limit: int = Field(
        default=int(AUTO_STRATEGY_CONFIG_DEFAULTS["default_strategies_limit"]),
        description="戦略取得デフォルト件数",
    )
    max_strategies_limit: int = Field(
        default=int(AUTO_STRATEGY_CONFIG_DEFAULTS["max_strategies_limit"]),
        description="戦略取得最大件数",
    )

    model_config = SettingsConfigDict(env_prefix="AUTO_STRATEGY_", extra="ignore")

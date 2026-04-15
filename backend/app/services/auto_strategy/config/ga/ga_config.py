"""
GA実行時設定クラス

GAConfig クラスを提供します。
GAConfig は GA エンジンのランタイム設定用 dataclass です。
戦略一覧 API の既定値は constants.common_constants.DEFAULT_STRATEGIES_LIMIT
と MAX_STRATEGIES_LIMIT を参照してください。
GA の基本パラメータは constants.ga_constants.GA_DEFAULT_CONFIG を参照します。
"""

import copy
import logging
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Dict, List, Optional, cast

from app.utils.serialization import dataclass_to_dict

from ..indicator_universe import normalize_indicator_universe_mode
from ..constants import (
    DEFAULT_FITNESS_CONSTRAINTS,
    DEFAULT_FITNESS_WEIGHTS,
    DEFAULT_GA_OBJECTIVE_WEIGHTS,
    DEFAULT_GA_OBJECTIVES,
    GA_FALLBACK_END_DATE,
    GA_FALLBACK_START_DATE,
    GA_DEFAULT_CONFIG,
    GA_DEFAULT_FITNESS_SHARING,
    GA_PARAMETER_RANGES,
    GA_THRESHOLD_RANGES,
)
from .nested_configs import (
    EvaluationConfig,
    HybridConfig,
    MutationConfig,
    RobustnessConfig,
    TuningConfig,
    TwoStageSelectionConfig,
)


logger = logging.getLogger(__name__)


def _get_default_values_from_fields(cls: type[Any]) -> Dict[str, Any]:
    """dataclass フィールド定義からデフォルト値辞書を組み立てる。"""
    defaults: Dict[str, Any] = {}
    for field_info in fields(cls):
        if field_info.default is not MISSING:
            defaults[field_info.name] = field_info.default
        if field_info.default_factory is not MISSING:
            try:
                if callable(field_info.default_factory):
                    defaults[field_info.name] = field_info.default_factory()
                else:
                    defaults[field_info.name] = field_info.default_factory
            except Exception as exc:
                logger.warning(
                    "デフォルト値生成失敗: %s, %s",
                    field_info.name,
                    exc,
                )
                defaults[field_info.name] = None
    return defaults





@dataclass
class GAConfig:
    """
    実行時GA設定クラス

    GA実行時の canonical 設定を管理する。
    """



    # 基本GA設定
    population_size: int = int(GA_DEFAULT_CONFIG["population_size"])
    generations: int = int(GA_DEFAULT_CONFIG["generations"])
    crossover_rate: float = GA_DEFAULT_CONFIG["crossover_rate"]
    mutation_rate: float = GA_DEFAULT_CONFIG["mutation_rate"]
    elite_size: int = int(GA_DEFAULT_CONFIG["elite_size"])
    max_indicators: int = int(GA_DEFAULT_CONFIG["max_indicators"])

    # 戦略生成制約
    min_indicators: int = 1
    min_conditions: int = 1
    max_conditions: int = 3

    # ペナルティ設定
    zero_trades_penalty: float = GA_DEFAULT_CONFIG["zero_trades_penalty"]
    constraint_violation_penalty: float = GA_DEFAULT_CONFIG[
        "constraint_violation_penalty"
    ]

    # 交叉・突然変異拡張設定
    # サブ設定: mutation_config

    # パラメータ範囲
    parameter_ranges: Dict[str, List] = field(
        default_factory=lambda: cast(
            Dict[str, List], copy.deepcopy(GA_PARAMETER_RANGES)
        )
    )
    threshold_ranges: Dict[str, List[float]] = field(
        default_factory=lambda: cast(
            Dict[str, List[float]], copy.deepcopy(GA_THRESHOLD_RANGES)
        )
    )

    # フィットネス設定
    fitness_weights: Dict[str, float] = field(
        default_factory=lambda: DEFAULT_FITNESS_WEIGHTS.copy()
    )
    fitness_constraints: Dict[str, Any] = field(
        default_factory=lambda: DEFAULT_FITNESS_CONSTRAINTS.copy()
    )

    # フィットネス共有設定
    fitness_sharing: Dict[str, Any] = field(
        default_factory=lambda: GA_DEFAULT_FITNESS_SHARING.copy()
    )

    # 多目的最適化設定
    enable_multi_objective: bool = False
    objectives: List[str] = field(default_factory=lambda: DEFAULT_GA_OBJECTIVES.copy())
    objective_weights: List[float] = field(
        default_factory=lambda: DEFAULT_GA_OBJECTIVE_WEIGHTS.copy()
    )
    dynamic_objective_reweighting: bool = True
    objective_dynamic_scalars: Dict[str, float] = field(default_factory=dict)

    # 評価設定拡張（単一目的最適化用）
    primary_metric: str = "sharpe_ratio"

    # 実行時設定
    parallel_processes: Optional[int] = None
    random_state: Optional[int] = None

    log_level: str = "ERROR"
    save_intermediate_results: bool = True

    # フォールバック設定
    fallback_start_date: str = GA_FALLBACK_START_DATE
    fallback_end_date: str = GA_FALLBACK_END_DATE

    # PurgedKFold設定（過学習対策）
    enable_purged_kfold: bool = False  # PurgedKFoldを有効にするフラグ
    purged_kfold_splits: int = 5  # 分割数
    purged_kfold_embargo: float = 0.01  # エンバーゴ率

    # 遺伝子生成設定拡張
    price_data_weight: int = 3
    volume_data_weight: int = 1
    oi_fr_data_weight: int = 1
    numeric_threshold_probability: float = 0.8
    min_compatibility_score: float = 0.8
    strict_compatibility_score: float = 0.9
    indicator_universe_mode: str = "curated"





    # マルチタイムフレーム（MTF）設定
    enable_multi_timeframe: bool = False
    available_timeframes: Optional[List[str]] = (
        None  # 利用可能なタイムフレーム（Noneの場合はデフォルト）
    )
    mtf_indicator_probability: float = 0.3  # MTF指標が生成される確率（0.0-1.0）

    # パラメータ範囲プリセット設定
    # 探索範囲のプリセット名（例: "short_term", "mid_term", "long_term"）
    # None の場合はデフォルト範囲を使用
    parameter_range_preset: Optional[str] = None

    # シード戦略設定（ハイブリッド初期化）
    # 実戦的な戦略テンプレートを初期集団に注入し、探索効率を向上させる
    use_seed_strategies: bool = True  # シード戦略を使用するか
    seed_injection_rate: float = 0.1  # 初期集団のうちシードで置き換える割合（0.0-1.0）

    # パラメータチューニング設定（GA×Optunaハイブリッド）
    # サブ設定: tuning_config

    # 二段階選抜設定
    # サブ設定: two_stage_selection_config

    # 二段階選抜用 robustness 設定
    # サブ設定: robustness_config



    # サブ設定（ネスト辞書からの復元用）
    mutation_config: MutationConfig = field(default_factory=MutationConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    hybrid_config: HybridConfig = field(default_factory=HybridConfig)
    tuning_config: TuningConfig = field(default_factory=TuningConfig)
    two_stage_selection_config: TwoStageSelectionConfig = field(
        default_factory=TwoStageSelectionConfig
    )
    robustness_config: RobustnessConfig = field(default_factory=RobustnessConfig)

    def __init__(self, **data: Any) -> None:
        """canonical フィールドのみを受け付ける手動初期化器。"""
        defaults = self._from_dict_defaults()
        unknown_keys = sorted(key for key in data if key not in defaults)
        if unknown_keys:
            raise ValueError(f"未対応の設定キーがあります: {', '.join(unknown_keys)}")

        defaults.update(copy.deepcopy(data))
        object.__setattr__(self, "_provided_keys", set(data.keys()))

        for key, value in defaults.items():
            object.__setattr__(self, key, value)

        self.__post_init__()

    def __post_init__(self) -> None:
        """初期化後に呼ばれる整合性同期フック。"""
        self._sync_runtime_fields()

    def __setattr__(self, name: str, value: Any) -> None:
        """既知フィールドのみ代入を許可する。"""
        if name.startswith("_") or name in type(self).__dataclass_fields__:
            object.__setattr__(self, name, value)
            return

        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def _sync_runtime_fields(self) -> None:
        """派生フィールドを同期する。"""
        object.__setattr__(
            self,
            "indicator_universe_mode",
            normalize_indicator_universe_mode(self.indicator_universe_mode),
        )

        if not isinstance(self.fitness_sharing, dict):
            object.__setattr__(
                self,
                "fitness_sharing",
                copy.deepcopy(GA_DEFAULT_FITNESS_SHARING),
            )
        else:
            merged_fitness_sharing = copy.deepcopy(GA_DEFAULT_FITNESS_SHARING)
            merged_fitness_sharing.update(self.fitness_sharing)
            object.__setattr__(self, "fitness_sharing", merged_fitness_sharing)

        if isinstance(self.mutation_config, dict):
            object.__setattr__(
                self,
                "mutation_config",
                MutationConfig.from_dict(self.mutation_config),
            )
        if isinstance(self.evaluation_config, dict):
            object.__setattr__(
                self,
                "evaluation_config",
                EvaluationConfig.from_dict(self.evaluation_config),
            )
        if isinstance(self.hybrid_config, dict):
            object.__setattr__(
                self, "hybrid_config", HybridConfig.from_dict(self.hybrid_config)
            )
        if isinstance(self.tuning_config, dict):
            object.__setattr__(
                self, "tuning_config", TuningConfig.from_dict(self.tuning_config)
            )
        if isinstance(self.two_stage_selection_config, dict):
            object.__setattr__(
                self,
                "two_stage_selection_config",
                TwoStageSelectionConfig.from_dict(self.two_stage_selection_config),
            )
        if isinstance(self.robustness_config, dict):
            object.__setattr__(
                self,
                "robustness_config",
                RobustnessConfig.from_dict(self.robustness_config),
            )

        mutation_config = self.mutation_config
        provided_keys = getattr(self, "_provided_keys", set())
        if "mutation_config" in provided_keys:
            object.__setattr__(self, "mutation_rate", mutation_config.rate)
        else:
            mutation_config.rate = self.mutation_rate

    def to_dict(self) -> Dict[str, Any]:
        """
        設定オブジェクトを辞書形式に変換

        DBへの保存やJSONシリアライズのために使用されます。

        Returns:
            設定値を含む辞書
        """
        return cast(Dict[str, Any], dataclass_to_dict(self))

    @classmethod
    def _from_dict_defaults(cls) -> Dict[str, Any]:
        """from_dict 用のデフォルト値を生成する。"""
        return cast(Dict[str, Any], copy.deepcopy(_get_default_values_from_fields(cls)))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GAConfig":
        """辞書形式から GAConfig インスタンスを生成する。"""
        return cls(**copy.deepcopy(data))

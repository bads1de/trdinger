"""
GA実行時設定クラス

GAConfig クラスと、GAConfig にぶら下がるネスト設定 dataclass 群を提供します。
GAConfig は GA エンジンのランタイム設定用 dataclass です。
戦略一覧 API の既定値は constants.DEFAULT_STRATEGIES_LIMIT
と MAX_STRATEGIES_LIMIT を参照してください。
GA の基本パラメータは constants.GA_DEFAULT_CONFIG を参照します。
"""

import copy
import logging
from collections.abc import Iterable, Mapping
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, cast

from app.utils.serialization import dataclass_to_dict

from .constants import (
    DEFAULT_FITNESS_CONSTRAINTS,
    DEFAULT_FITNESS_WEIGHTS,
    DEFAULT_GA_OBJECTIVE_WEIGHTS,
    DEFAULT_GA_OBJECTIVES,
    DEFAULT_NON_PRICE_INDICATOR_PROBABILITY,
    GA_DEFAULT_CONFIG,
    GA_DEFAULT_FITNESS_SHARING,
    GA_FALLBACK_END_DATE,
    GA_FALLBACK_START_DATE,
    GA_MUTATION_SETTINGS,
    GA_PARAMETER_RANGES,
    OPERATORS,
)
from .evaluation_plan import EvaluationPlan
from .indicator_universe import normalize_indicator_universe_mode

logger = logging.getLogger(__name__)

DEFAULT_EARLY_TERMINATION_VALUES = {
    "enabled": True,
    "max_drawdown": 0.15,
    "min_trades": 30,
    "min_trade_check_progress": 0.33,
    "trade_pace_tolerance": 0.5,
    "min_expectancy": -0.05,
    "expectancy_min_trades": 10,
    "expectancy_progress": 0.1,
}


def _read_value(source: object, key: str, default: object = None) -> object:
    """dict / オブジェクトのどちらからでも値を取得する。"""
    if isinstance(source, Mapping):
        if key in source:
            return source[key]
        return default

    try:
        values = vars(source)
    except TypeError:
        values = None

    if values is not None:
        if key in values:
            return values[key]
        return default

    return getattr(source, key, default)


def _filter_known_fields(
    cls: type, data: Mapping[str, Any], config_name: str
) -> dict[str, Any]:
    """既知フィールドのみを残し、未知キーは警告する。"""
    known = {field_info.name for field_info in fields(cls)}
    unknown = sorted(key for key in data if key not in known)
    if unknown:
        logger.warning(
            "%s の未対応キーを無視しました: %s",
            config_name,
            ", ".join(unknown),
        )
    return {key: value for key, value in data.items() if key in known}


class NestedConfigMixin:
    """ネスト設定 dataclass の共通変換処理。"""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Any:
        filtered = _filter_known_fields(cls, dict(data), cls.__name__)
        for field_info in fields(cast(type[Any], cls)):
            value = filtered.get(field_info.name)
            if not isinstance(value, Mapping):
                continue
            default_factory = getattr(field_info, "default_factory", MISSING)
            if default_factory is MISSING:
                sample = field_info.default
            else:
                try:
                    sample = default_factory()
                except Exception:
                    continue
            converter = getattr(type(sample), "from_source", None)
            if converter is not None:
                filtered[field_info.name] = converter(value)
        return cls(**filtered)


@dataclass(frozen=True)
class EarlyTerminationSettings(NestedConfigMixin):
    """早期終了の正規化済み設定。"""

    enabled: bool = True
    max_drawdown: float | None = 0.15
    min_trades: int | None = 30
    min_trade_check_progress: float = 0.33
    trade_pace_tolerance: float = 0.5
    min_expectancy: float | None = -0.05
    expectancy_min_trades: int = 10
    expectancy_progress: float = 0.1

    @classmethod
    def from_source(cls, source: object) -> "EarlyTerminationSettings":
        """dict / オブジェクトのどちらからでも設定を生成する。"""
        if isinstance(source, cls):
            return source
        if isinstance(source, Mapping):
            return cast("EarlyTerminationSettings", cls.from_dict(source))

        filtered: dict[str, Any] = {
            field_name: _read_value(source, field_name, default)
            for field_name, default in DEFAULT_EARLY_TERMINATION_VALUES.items()
        }
        return cls(**filtered)


@dataclass
class MutationConfig(NestedConfigMixin):
    """突然変異関連設定。"""

    crossover_field_selection_probability: float = float(
        cast(float, GA_MUTATION_SETTINGS["crossover_field_selection_probability"])
    )
    indicator_param_range: list[float] = field(
        default_factory=lambda: list(
            cast(
                Iterable[float], GA_MUTATION_SETTINGS["indicator_param_mutation_range"]
            )
        )
    )
    risk_param_range: list[float] = field(
        default_factory=lambda: list(
            cast(Iterable[float], GA_MUTATION_SETTINGS["risk_param_mutation_range"])
        )
    )
    indicator_add_delete_probability: float = float(
        cast(float, GA_MUTATION_SETTINGS["indicator_add_delete_probability"])
    )
    indicator_add_vs_delete_probability: float = float(
        cast(float, GA_MUTATION_SETTINGS["indicator_add_vs_delete_probability"])
    )
    condition_change_multiplier: float = float(
        cast(float, GA_MUTATION_SETTINGS["condition_change_probability_multiplier"])
    )
    condition_selection_probability: float = float(
        cast(float, GA_MUTATION_SETTINGS["condition_selection_probability"])
    )
    condition_operator_switch_probability: float = float(
        cast(float, GA_MUTATION_SETTINGS["condition_operator_switch_probability"])
    )
    tpsl_gene_creation_multiplier: float = float(
        cast(float, GA_MUTATION_SETTINGS["tpsl_gene_creation_probability_multiplier"])
    )
    position_sizing_gene_creation_multiplier: float = float(
        cast(
            float,
            GA_MUTATION_SETTINGS[
                "position_sizing_gene_creation_probability_multiplier"
            ],
        )
    )
    entry_gene_creation_multiplier: float = float(
        cast(float, GA_MUTATION_SETTINGS["entry_gene_creation_probability_multiplier"])
    )
    exit_gene_creation_multiplier: float = float(
        cast(float, GA_MUTATION_SETTINGS["exit_gene_creation_probability_multiplier"])
    )
    adaptive_variance_threshold: float = float(
        cast(float, GA_MUTATION_SETTINGS["adaptive_mutation_variance_threshold"])
    )
    adaptive_decrease_multiplier: float = float(
        cast(float, GA_MUTATION_SETTINGS["adaptive_mutation_rate_decrease_multiplier"])
    )
    adaptive_increase_multiplier: float = float(
        cast(float, GA_MUTATION_SETTINGS["adaptive_mutation_rate_increase_multiplier"])
    )
    valid_condition_operators: list[str] = field(
        default_factory=lambda: OPERATORS.copy()
    )


@dataclass
class EvaluationConfig(NestedConfigMixin):
    """評価・検証関連設定。"""

    enable_parallel: bool = True
    max_workers: int | None = None  # Noneの場合は自動で物理コア-2, 最大8に制限
    timeout: float = 300.0
    enable_multi_fidelity_evaluation: bool = False
    multi_fidelity_window_ratio: float = 0.3
    multi_fidelity_oos_ratio: float = 0.2
    multi_fidelity_candidate_ratio: float = 0.25
    multi_fidelity_min_candidates: int = 3
    early_termination_settings: EarlyTerminationSettings = field(
        default_factory=EarlyTerminationSettings
    )
    oos_split_ratio: float = 0.25
    oos_fitness_weight: float = 0.5
    enable_walk_forward: bool = False
    wfa_n_folds: int = 3
    wfa_train_ratio: float = 0.5
    wfa_anchored: bool = False
    # GA選抜用の WFA を IS 窓内で実行するフラグ (デフォルト無効)。
    # 有効時は execute_ga_report が IS 窓を fold 分割し robust 集約で選抜する。
    enable_walk_forward_for_ga: bool = False
    # 明示的な評価モード（"single" | "walk_forward" | "auto"）。
    # "auto"（既定）の場合は enable_walk_forward フラグから自動判定する（後方互換）。
    evaluation_mode: str = "auto"


@dataclass
class TwoStageSelectionConfig(NestedConfigMixin):
    """二段階選抜関連設定。"""

    enabled: bool = True
    elite_count: int = 3
    candidate_pool_size: int = 5
    min_pass_rate: float = 0.5


@dataclass
class RobustnessConfig(NestedConfigMixin):
    """候補戦略の robustness gate 設定。"""

    enabled: bool = False
    min_pass_rate: float = 1.0
    validation_symbols: list[str] | None = None
    regime_windows: list[dict] = field(default_factory=list)
    stress_slippage: list[float] = field(default_factory=list)
    stress_commission_multipliers: list[float] = field(default_factory=list)
    aggregate_method: str = "robust"


@dataclass
class IterativeImprovementConfig(NestedConfigMixin):
    """反復改善ループ設定。

    自動検証に合格した過去の生成戦略を、次のGA実験のシード戦略として
    自動注入し、戦略開発を反復的に改善させるための設定。
    """

    # 反復改善ループを有効化するか
    enabled: bool = False
    # 注入する過去戦略の最大数（上位N件をシードとして使用）
    max_seed_strategies: int = 5
    # シードとして再利用する戦略の最低フィットネス（None の場合はチェックしない）
    min_fitness: float | None = None
    # 自動検証に合格した戦略のみをシードにするか（False の場合は全戦略から選択）
    validation_passed_only: bool = True


@dataclass
class ValidationConfig(NestedConfigMixin):
    """自動検証パイプライン関連設定。

    GA実行後に Walk-Forward Analysis で最終候補戦略を検証し、
    合格（Pass）/不合格（Fail）を判定するための設定。
    合格した戦略だけが DB へ保存される。
    """

    # 自動検証パイプラインを有効化するか
    # 過学習対策のためデフォルト有効（合格戦略のみDB保存）
    enabled: bool = True

    # 合格判定基準
    # WFA のフォールド合格率（0.0-1.0）。この値以上の戦略のみ合格。
    min_pass_rate: float = 0.5
    # 集約プライマリフィットネスの下限（None の場合はチェックしない）
    min_primary_fitness: float | None = None
    # 全フォールドの最小取引回数（None の場合はチェックしない）
    min_trades: int | None = None
    # 全フォールドの最大ドローダウン上限（0.0-1.0、None の場合はチェックしない）
    max_drawdown: float | None = None

    # 過学習対策ゲート（PBO / DSR）
    # PBO (Probability of Backtest Overfitting) 風ゲート:
    # 総リターンが負のフォールドの比率が pbo_threshold を超える戦略は不合格。
    # 例: 0.5 = 過半数のフォールドで利益を出せていない戦略は過学習として弾く。
    enable_pbo_gate: bool = True
    pbo_threshold: float = 0.5
    # Deflated Sharpe Ratio ゲート（多重検定補正）。
    # GA の探索試行数（population_size x generations）でシャープレシオを
    # 割り引いた有意性が min_dsr 未満の戦略は不合格。
    # 厳格な統計検定のためデフォルト無効（有効時は合格水準が大きく上がる）。
    enable_dsr_gate: bool = False
    min_dsr: float = 0.95
    # DSR の有効試行数（None の場合は population_size * generations を自動計算）
    dsr_effective_trials: int | None = None
    # 帰無分布での年率シャープレシオの標準偏差（通常 1.0 を仮定）
    dsr_sigma_sharpe: float = 1.0

    # WFA 設定（検証用。評価設定を上書きする）
    # フォールド数と train 比率は、テスト窓が取引回数確保に十分な長さに
    # なるよう調整済み。2024H1・4h足の場合、テスト窓は約30日
    # （=(1 - 0.5) * 181日 / 3）となり、レンジ相場でも正リターン達成の
    # 可能性が十分に評価できる。min_pass_rate=0.5 は 3 フォールド中 2 合格
    # を要求する。
    wfa_n_folds: int = 3
    wfa_train_ratio: float = 0.5
    wfa_anchored: bool = False

    # 候補戦略も検証するか（False の場合は最良戦略のみ検証）
    validate_candidates: bool = True
    # 候補検証の対象数（上位 N 件）
    max_candidates: int = 5


@dataclass
class FitnessSharingConfig(NestedConfigMixin):
    """フィットネス共有関連設定。"""

    enable_fitness_sharing: bool = cast(
        bool, GA_DEFAULT_FITNESS_SHARING["enable_fitness_sharing"]
    )
    sharing_radius: float = GA_DEFAULT_FITNESS_SHARING["sharing_radius"]
    sharing_alpha: float = GA_DEFAULT_FITNESS_SHARING["sharing_alpha"]
    sampling_threshold: int = cast(
        int, GA_DEFAULT_FITNESS_SHARING["sampling_threshold"]
    )
    sampling_ratio: float = GA_DEFAULT_FITNESS_SHARING["sampling_ratio"]


@dataclass
class SurvivalSelectionConfig(NestedConfigMixin):
    """生存選択の多様性維持設定。"""

    # 制限トーナメント置換 (RTR) を次世代選択の前段で適用するか。
    # 有効時、各子個体は指標構成距離で最も近い既存個体との局所競争を経て
    # NSGA-II 選抜プールに入る（類似子がニッチ外の多様な親を置き換えられない）。
    enable_restricted_tournament: bool = True
    # 局所競争相手のサンプル数（crowding factor）
    restricted_tournament_crowding_factor: int = 4


def _get_default_values_from_fields(cls: type[Any]) -> dict[str, Any]:
    """dataclass フィールド定義からデフォルト値辞書を組み立てる。"""
    defaults: dict[str, Any] = {}
    for field_info in fields(cls):
        if field_info.default is not MISSING:
            defaults[field_info.name] = field_info.default
        if field_info.default_factory is not MISSING:
            try:
                defaults[field_info.name] = field_info.default_factory()
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

    # 非価格指標（OI/FR/LSR由来）の最低数保証
    # 1 以上にすると各戦略が必ず指定数以上の非価格指標を含むよう生成・進化する。
    min_non_price_indicators: int = 1

    # 非価格指標の選択確率（公平な自由探索用）
    # トレンド70%優先バイアスで非価格指標が埋もれるのを防ぐ。
    # 0.0 で無効化（従来挙動）、デフォルト 0.3 で非価格指標にも
    # 専用の選択機会を与える。
    non_price_indicator_probability: float = DEFAULT_NON_PRICE_INDICATOR_PROBABILITY

    # ペナルティ設定
    zero_trades_penalty: float = GA_DEFAULT_CONFIG["zero_trades_penalty"]
    constraint_violation_penalty: float = GA_DEFAULT_CONFIG[
        "constraint_violation_penalty"
    ]
    max_enabled_filters: int = int(GA_DEFAULT_CONFIG.get("max_enabled_filters", 3))

    # 交叉・突然変異拡張設定
    # サブ設定: mutation_config

    # パラメータ範囲
    parameter_ranges: dict[str, list] = field(
        default_factory=lambda: cast(
            dict[str, list], copy.deepcopy(GA_PARAMETER_RANGES)
        )
    )

    # フィットネス設定
    fitness_weights: dict[str, float] = field(
        default_factory=lambda: DEFAULT_FITNESS_WEIGHTS.copy()
    )
    fitness_constraints: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_FITNESS_CONSTRAINTS.copy()
    )

    # フィットネス共有設定
    fitness_sharing: FitnessSharingConfig = field(default_factory=FitnessSharingConfig)

    # 生存選択の多様性維持設定
    survival_selection_config: SurvivalSelectionConfig = field(
        default_factory=SurvivalSelectionConfig
    )

    # 目的関数設定
    objectives: list[str] = field(default_factory=lambda: DEFAULT_GA_OBJECTIVES.copy())
    objective_weights: list[float] = field(
        default_factory=lambda: DEFAULT_GA_OBJECTIVE_WEIGHTS.copy()
    )
    dynamic_objective_reweighting: bool = True
    objective_dynamic_scalars: dict[str, float] = field(default_factory=dict)

    # 目的関数プリセット: エッジ検証や長期窓で識別力を保つための切替
    # "weighted_score" (デフォルト) / "excess_return" (B&H超過を直接最適化)
    objective_preset: str | None = None

    # 実行時設定
    random_state: int | None = None

    log_level: str = "ERROR"

    # フォールバック設定
    fallback_start_date: str = GA_FALLBACK_START_DATE
    fallback_end_date: str = GA_FALLBACK_END_DATE

    # 遺伝子生成設定拡張
    indicator_universe_mode: str = "curated"

    # マルチタイムフレーム（MTF）設定
    enable_multi_timeframe: bool = False
    available_timeframes: list[str] | None = (
        None  # 利用可能なタイムフレーム（Noneの場合はデフォルト）
    )
    mtf_indicator_probability: float = 0.3  # MTF指標が生成される確率（0.0-1.0）

    # パラメータ範囲プリセット設定
    # 探索範囲のプリセット名（例: "short_term", "mid_term", "long_term"）
    # None の場合はデフォルト範囲を使用
    parameter_range_preset: str | None = None

    # シード戦略設定（ハイブリッド初期化）
    # 実戦的な戦略テンプレートを初期集団に注入し、探索効率を向上させる
    use_seed_strategies: bool = True  # シード戦略を使用するか
    seed_injection_rate: float = 0.20  # 初期集団のうちシードで置き換える割合（0.0-1.0）

    # 二段階選抜設定
    # サブ設定: two_stage_selection_config

    # 二段階選抜用 robustness 設定
    # サブ設定: robustness_config

    # 自動検証パイプライン設定（GA後のWFA検証 + 合格判定）
    # サブ設定: validation_config

    # 反復改善ループ設定（合格した過去戦略を次のシードとして再利用）
    # サブ設定: iterative_improvement_config

    # サブ設定（ネスト辞書からの復元用）
    mutation_config: MutationConfig = field(default_factory=MutationConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    two_stage_selection_config: TwoStageSelectionConfig = field(
        default_factory=TwoStageSelectionConfig
    )
    robustness_config: RobustnessConfig = field(default_factory=RobustnessConfig)
    iterative_improvement_config: IterativeImprovementConfig = field(
        default_factory=IterativeImprovementConfig
    )
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    evaluation_plan: EvaluationPlan | dict[str, Any] | None = None

    def __init__(self, **data: dict[str, Any]) -> None:
        """canonical フィールドのみを受け付ける手動初期化器。"""
        defaults = self._from_dict_defaults()
        unknown_keys = sorted(key for key in data if key not in defaults)
        if unknown_keys:
            raise ValueError(f"未対応の設定キーがあります: {', '.join(unknown_keys)}")

        defaults.update(copy.deepcopy(data))

        for key, value in defaults.items():
            object.__setattr__(self, key, value)

        self.__post_init__()

    def __post_init__(self) -> None:
        """初期化後に呼ばれる整合性同期フック。"""
        self._sync_runtime_fields()

    def __setattr__(self, name: str, value: Any) -> None:
        """既知フィールドのみ代入を許可する。"""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        if name not in type(self).__dataclass_fields__:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )

        if name == "mutation_config" and isinstance(value, dict):
            value = MutationConfig.from_dict(value)
        if name == "fitness_sharing" and isinstance(value, dict):
            value = FitnessSharingConfig.from_dict(value)
        if name == "survival_selection_config" and isinstance(value, dict):
            value = SurvivalSelectionConfig.from_dict(value)

        object.__setattr__(self, name, value)

    def _sync_runtime_fields(self) -> None:
        """派生フィールドを同期する。"""
        object.__setattr__(
            self,
            "indicator_universe_mode",
            normalize_indicator_universe_mode(self.indicator_universe_mode),
        )

        fitness_sharing: Any = self.fitness_sharing
        if not isinstance(fitness_sharing, FitnessSharingConfig):
            object.__setattr__(
                self,
                "fitness_sharing",
                FitnessSharingConfig.from_dict(fitness_sharing),
            )

        survival_selection: Any = self.survival_selection_config
        if isinstance(survival_selection, dict):
            object.__setattr__(
                self,
                "survival_selection_config",
                SurvivalSelectionConfig.from_dict(survival_selection),
            )

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
        if isinstance(self.iterative_improvement_config, dict):
            object.__setattr__(
                self,
                "iterative_improvement_config",
                IterativeImprovementConfig.from_dict(self.iterative_improvement_config),
            )
        if isinstance(self.validation_config, dict):
            object.__setattr__(
                self,
                "validation_config",
                ValidationConfig.from_dict(self.validation_config),
            )
        if isinstance(self.evaluation_plan, dict):
            object.__setattr__(
                self,
                "evaluation_plan",
                EvaluationPlan.from_dict(self.evaluation_plan),
            )

        preset = getattr(self, "objective_preset", None)
        if isinstance(preset, str) and preset.strip():
            normalized = preset.strip()
            if normalized == "excess_return":
                object.__setattr__(self, "objectives", ["excess_return"])
                object.__setattr__(self, "objective_weights", [1.0])
            elif normalized == "weighted_score":
                object.__setattr__(self, "objectives", list(DEFAULT_GA_OBJECTIVES))
                object.__setattr__(
                    self, "objective_weights", list(DEFAULT_GA_OBJECTIVE_WEIGHTS)
                )

    def to_dict(self) -> dict[str, Any]:
        """
        設定オブジェクトを辞書形式に変換

        DBへの保存やJSONシリアライズのために使用されます。

        Returns:
            設定値を含む辞書
        """
        return cast(dict[str, Any], dataclass_to_dict(self))

    @classmethod
    def _from_dict_defaults(cls) -> dict[str, Any]:
        """from_dict 用のデフォルト値を生成する。"""
        return copy.deepcopy(_get_default_values_from_fields(cls))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GAConfig":
        """辞書形式から GAConfig インスタンスを生成する。"""
        return cls(**copy.deepcopy(data))


__all__ = [
    "EarlyTerminationSettings",
    "MutationConfig",
    "EvaluationConfig",
    "TwoStageSelectionConfig",
    "RobustnessConfig",
    "IterativeImprovementConfig",
    "ValidationConfig",
    "FitnessSharingConfig",
    "SurvivalSelectionConfig",
    "GAConfig",
]

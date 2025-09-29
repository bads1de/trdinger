"""
GA実行時設定クラス

GAConfigクラスとGAProgressクラスを提供します。
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, cast

from .auto_strategy import AutoStrategyConfig
from .base import BaseConfig
from .ga import (
    DEFAULT_FITNESS_CONSTRAINTS,
    DEFAULT_FITNESS_WEIGHTS,
    DEFAULT_GA_OBJECTIVE_WEIGHTS,
    DEFAULT_GA_OBJECTIVES,
    GA_DEFAULT_CONFIG,
    GA_DEFAULT_FITNESS_SHARING,
    GA_PARAMETER_RANGES,
    GA_THRESHOLD_RANGES,
)
from .tpsl import (
    GA_DEFAULT_TPSL_METHOD_CONSTRAINTS,
    GA_TPSL_ATR_MULTIPLIER_RANGE,
    GA_TPSL_RR_RANGE,
    GA_TPSL_SL_RANGE,
    GA_TPSL_TP_RANGE,
)

logger = logging.getLogger(__name__)


@dataclass
class GAConfig(BaseConfig):
    """
    実行時GA設定クラス

    GA実行時のフラット設定を管理する。
    重複を避けるため、GASettingsから動的なプロパティアクセスを使用。

    Args:
        auto_strategy_config: AutoStrategyConfigインスタンス（オプション）
                               指定された場合、このGAConfigはAutoStrategyConfig.gaから設定を継承します。
    """

    # 参照設定（AutoStrategyConfig統合用）
    auto_strategy_config: Optional[AutoStrategyConfig] = None

    # 基本GA設定
    population_size: int = GA_DEFAULT_CONFIG["population_size"]
    generations: int = GA_DEFAULT_CONFIG["generations"]
    crossover_rate: float = GA_DEFAULT_CONFIG["crossover_rate"]
    mutation_rate: float = GA_DEFAULT_CONFIG["mutation_rate"]
    elite_size: int = GA_DEFAULT_CONFIG["elite_size"]
    max_indicators: int = GA_DEFAULT_CONFIG["max_indicators"]

    # 戦略生成制約
    min_indicators: int = 1
    min_conditions: int = 1
    max_conditions: int = 3

    # パラメータ範囲
    parameter_ranges: Dict[str, List] = field(
        default_factory=lambda: GA_PARAMETER_RANGES.copy()
    )
    threshold_ranges: Dict[str, List[float]] = field(
        default_factory=lambda: GA_THRESHOLD_RANGES.copy()
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
    enable_fitness_sharing: bool = GA_DEFAULT_FITNESS_SHARING["enable_fitness_sharing"]
    sharing_radius: float = GA_DEFAULT_FITNESS_SHARING["sharing_radius"]
    sharing_alpha: float = GA_DEFAULT_FITNESS_SHARING["sharing_alpha"]

    # 多目的最適化設定
    enable_multi_objective: bool = False
    objectives: List[str] = field(default_factory=lambda: DEFAULT_GA_OBJECTIVES.copy())
    objective_weights: List[float] = field(
        default_factory=lambda: DEFAULT_GA_OBJECTIVE_WEIGHTS.copy()
    )

    # 評価設定拡張（単一目的最適化用）
    primary_metric: str = "sharpe_ratio"

    # 実行時設定
    parallel_processes: Optional[int] = None
    random_state: Optional[int] = None
    log_level: str = "ERROR"
    save_intermediate_results: bool = True
    progress_callback: Optional[Callable[["GAProgress"], None]] = None

    # フォールバック設定
    fallback_start_date: str = "2024-01-01"
    fallback_end_date: str = "2024-04-09"

    # レジーム適応設定
    regime_adaptation_enabled: bool = False

    # 指標設定拡張
    allowed_indicators: List[str] = field(default_factory=list)

    # 遺伝子生成設定拡張
    price_data_weight: int = 3
    volume_data_weight: int = 1
    oi_fr_data_weight: int = 1
    numeric_threshold_probability: float = 0.8
    min_compatibility_score: float = 0.8
    strict_compatibility_score: float = 0.9

    # TPSL関連設定拡張
    tpsl_method_constraints: Optional[List[str]] = None
    tpsl_sl_range: Optional[List[float]] = None
    tpsl_tp_range: Optional[List[float]] = None
    tpsl_rr_range: Optional[List[float]] = None
    tpsl_atr_multiplier_range: Optional[List[float]] = None

    def get_ga_settings(self):
        """関連するGASettingsを取得"""
        if self.auto_strategy_config:
            return self.auto_strategy_config.ga
        return None  # 変更: GASettings() を None に

    def __post_init__(self) -> None:
        """Post-initialization validation"""
        # Validate integer fields
        if not isinstance(self.population_size, int) or self.population_size <= 0:
            raise ValueError("population_size は正の整数である必要があります")
        if not isinstance(self.generations, int) or self.generations <= 0:
            raise ValueError("generations は正の整数である必要があります")
        if not isinstance(self.elite_size, int) or self.elite_size < 0:
            raise ValueError("elite_size は負でない整数である必要があります")
        if not isinstance(self.max_indicators, int) or self.max_indicators <= 0:
            raise ValueError("max_indicators は正の整数である必要があります")

        # Validate float fields
        if not isinstance(self.crossover_rate, (int, float)) or not (
            0 <= self.crossover_rate <= 1
        ):
            raise ValueError("crossover_rate は0から1の範囲の実数である必要があります")
        if not isinstance(self.mutation_rate, (int, float)) or not (
            0 <= self.mutation_rate <= 1
        ):
            raise ValueError("mutation_rate は0から1の範囲の実数である必要があります")

        # Convert int to float if necessary
        if isinstance(self.crossover_rate, int):
            self.crossover_rate = float(self.crossover_rate)
        if isinstance(self.mutation_rate, int):
            self.mutation_rate = float(self.mutation_rate)

    def validate(self) -> tuple[bool, List[str]]:
        """
        設定の妥当性を検証

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # 進化設定の検証
        try:
            if self.population_size <= 0:
                errors.append("個体数は正の整数である必要があります")
            elif self.population_size > 1000:
                errors.append(
                    "個体数は1000以下である必要があります（パフォーマンス上の制約）"
                )
        except TypeError:
            errors.append("個体数は数値である必要があります")

        try:
            if self.generations <= 0:
                errors.append("世代数は正の整数である必要があります")
            elif self.generations > 500:
                errors.append(
                    "世代数は500以下である必要があります（パフォーマンス上の制約）"
                )
        except TypeError:
            errors.append("世代数は数値である必要があります")

        try:
            if not 0 <= self.crossover_rate <= 1:
                errors.append("交叉率は0-1の範囲である必要があります")
        except (TypeError, ValueError):
            errors.append("交叉率は数値である必要があります")

        try:
            if not 0 <= self.mutation_rate <= 1:
                errors.append("突然変異率は0-1の範囲である必要があります")
        except (TypeError, ValueError):
            errors.append("突然変異率は数値である必要があります")

        try:
            if self.elite_size < 0 or self.elite_size >= self.population_size:
                errors.append("エリート保存数は0以上、個体数未満である必要があります")
        except (TypeError, ValueError):
            errors.append("elite_size と population_size は数値である必要があります")

        # 評価設定の検証
        if abs(sum(self.fitness_weights.values()) - 1.0) > 0.01:
            errors.append("フィットネス重みの合計は1.0である必要があります")

        required_metrics = {"total_return", "sharpe_ratio", "max_drawdown", "win_rate"}
        missing_metrics = required_metrics - set(self.fitness_weights.keys())
        if missing_metrics:
            errors.append(f"必要なメトリクスが不足しています: {missing_metrics}")

        if self.primary_metric not in self.fitness_weights:
            errors.append(
                f"プライマリメトリクス '{self.primary_metric}' がフィットネス重みに含まれていません"
            )

        # 指標設定の検証
        try:
            if self.max_indicators <= 0:
                errors.append("最大指標数は正の整数である必要があります")
            elif self.max_indicators > 10:
                errors.append(
                    "最大指標数は10以下である必要があります（パフォーマンス上の制約）"
                )
        except TypeError:
            errors.append("最大指標数は数値である必要があります")

        if self.allowed_indicators:
            try:
                from app.services.indicators import TechnicalIndicatorService

                valid_indicators = set(
                    TechnicalIndicatorService().get_supported_indicators().keys()
                )
                invalid_indicators = set(self.allowed_indicators) - valid_indicators
                if invalid_indicators:
                    errors.append(f"無効な指標が含まれています: {invalid_indicators}")
            except Exception:
                # インポートできない場合は検証スキップ
                logger.warning("指標検証がスキップされました")

        # パラメータ設定の検証
        for param_name, range_values in self.parameter_ranges.items():
            if not isinstance(range_values, list) or len(range_values) != 2:
                errors.append(
                    f"パラメータ '{param_name}' の範囲は [min, max] の形式である必要があります"
                )
            elif range_values[0] >= range_values[1]:
                errors.append(
                    f"パラメータ '{param_name}' の最小値は最大値より小さい必要があります"
                )

        # 実行設定の検証
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            errors.append(
                f"無効なログレベル: {self.log_level}. 有効な値: {valid_log_levels}"
            )

        if self.parallel_processes is not None:
            if self.parallel_processes <= 0:
                errors.append("並列プロセス数は正の整数である必要があります")
            elif self.parallel_processes > 32:
                errors.append("並列プロセス数は32以下である必要があります")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "elite_size": self.elite_size,
            "fitness_weights": self.fitness_weights,
            "primary_metric": self.primary_metric,
            "max_indicators": self.max_indicators,
            # 多目的最適化設定
            "enable_multi_objective": self.enable_multi_objective,
            "objectives": self.objectives,
            "objective_weights": self.objective_weights,
            "allowed_indicators": self.allowed_indicators,
            "parameter_ranges": self.parameter_ranges,
            "threshold_ranges": self.threshold_ranges,
            "fitness_constraints": self.fitness_constraints,
            "min_indicators": self.min_indicators,
            "min_conditions": self.min_conditions,
            "max_conditions": self.max_conditions,
            "parallel_processes": self.parallel_processes,
            "random_state": self.random_state,
            "log_level": self.log_level,
            "save_intermediate_results": self.save_intermediate_results,
            # フォールバック設定
            "fallback_start_date": self.fallback_start_date,
            "fallback_end_date": self.fallback_end_date,
            # フィットネス共有設定
            "enable_fitness_sharing": self.enable_fitness_sharing,
            "sharing_radius": self.sharing_radius,
            "sharing_alpha": self.sharing_alpha,
            # TPSL設定
            "tpsl_method_constraints": self.tpsl_method_constraints,
            "tpsl_sl_range": self.tpsl_sl_range,
            "tpsl_tp_range": self.tpsl_tp_range,
            "tpsl_rr_range": self.tpsl_rr_range,
            "tpsl_atr_multiplier_range": self.tpsl_atr_multiplier_range,
            # レジーム適応設定
            "regime_adaptation_enabled": self.regime_adaptation_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GAConfig":
        """辞書から復元（BaseConfig統一化バージョン）"""
        # GA特有のデータ前処理
        processed_data = cls._preprocess_ga_dict(data)

        # BaseConfigの標準from_dict処理を使用
        return cast(GAConfig, super().from_dict(processed_data))

    @classmethod
    def _preprocess_ga_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """GAConfig特有のデータの前処理"""
        # allowed_indicatorsが空の場合はデフォルトの指標リストを使用
        if not data.get("allowed_indicators"):
            try:
                from app.services.indicators import TechnicalIndicatorService

                indicator_service = TechnicalIndicatorService()
                data["allowed_indicators"] = list(
                    indicator_service.get_supported_indicators().keys()
                )
            except Exception:
                # インポートできない場合はデフォルトを使用
                logger.warning("指標サービスの取得が失敗しました")
                data["allowed_indicators"] = []

        # fitness_weightsが指定されていない場合はデフォルト値を使用
        if not data.get("fitness_weights"):  # 空の辞書やNoneの場合
            data["fitness_weights"] = DEFAULT_FITNESS_WEIGHTS

        # 他のデフォルト値も設定（既存のget()ロジックを維持）
        defaults = {
            "population_size": GA_DEFAULT_CONFIG["population_size"],
            "generations": GA_DEFAULT_CONFIG["generations"],
            "crossover_rate": GA_DEFAULT_CONFIG["crossover_rate"],
            "mutation_rate": GA_DEFAULT_CONFIG["mutation_rate"],
            "elite_size": GA_DEFAULT_CONFIG.get("elite_size", 10),
            "primary_metric": "sharpe_ratio",
            "fitness_constraints": DEFAULT_FITNESS_CONSTRAINTS,
            "max_indicators": GA_DEFAULT_CONFIG["max_indicators"],
            "parameter_ranges": GA_PARAMETER_RANGES,
            "threshold_ranges": GA_THRESHOLD_RANGES,
            "min_indicators": 1,
            "min_conditions": 1,
            "max_conditions": 3,
            "log_level": "ERROR",
            "save_intermediate_results": True,
            # フィットネス共有設定
            "enable_fitness_sharing": GA_DEFAULT_FITNESS_SHARING[
                "enable_fitness_sharing"
            ],
            "sharing_radius": GA_DEFAULT_FITNESS_SHARING["sharing_radius"],
            "sharing_alpha": GA_DEFAULT_FITNESS_SHARING["sharing_alpha"],
            # 多目的最適化設定
            "enable_multi_objective": False,
            "objectives": DEFAULT_GA_OBJECTIVES,
            "objective_weights": DEFAULT_GA_OBJECTIVE_WEIGHTS,
            # 実行設定
            "parallel_processes": None,
            "random_state": None,
            # フォールバック設定
            "fallback_start_date": data.get("fallback_start_date", "2024-01-01"),
            "fallback_end_date": data.get("fallback_end_date", "2024-04-09"),
            # TPSL設定デフォルト
            "tpsl_method_constraints": data.get(
                "tpsl_method_constraints", GA_DEFAULT_TPSL_METHOD_CONSTRAINTS
            ),
            "tpsl_sl_range": data.get("tpsl_sl_range", GA_TPSL_SL_RANGE),
            "tpsl_tp_range": data.get("tpsl_tp_range", GA_TPSL_TP_RANGE),
            "tpsl_rr_range": data.get("tpsl_rr_range", GA_TPSL_RR_RANGE),
            "tpsl_atr_multiplier_range": data.get(
                "tpsl_atr_multiplier_range", GA_TPSL_ATR_MULTIPLIER_RANGE
            ),
            # レジーム適応設定
            "regime_adaptation_enabled": data.get("regime_adaptation_enabled", False),
        }

        # デフォルト値をマージ
        for key, default_value in defaults.items():
            if data.get(key) is None:  # 明示的にNoneが設定されている場合のみ
                data[key] = default_value

        return data

    def apply_auto_strategy_config(self, config: AutoStrategyConfig) -> None:
        """
        AutoStrategyConfigから設定を適用

        Args:
            config: AutoStrategyConfigインスタンス
        """
        # GA設定をAutoStrategyConfigから継承
        ga_config = config.ga
        self.auto_strategy_config = config

        # 基本GAパラメータ
        self.population_size = ga_config.population_size
        self.generations = ga_config.generations
        self.crossover_rate = ga_config.crossover_rate
        self.mutation_rate = ga_config.mutation_rate
        self.elite_size = ga_config.elite_size
        self.max_indicators = ga_config.max_indicators
        self.min_indicators = ga_config.min_indicators
        self.min_conditions = ga_config.min_conditions
        self.max_conditions = ga_config.max_conditions

        # 評価関連設定
        self.fitness_weights = ga_config.fitness_weights.copy()
        self.fitness_constraints = ga_config.fitness_constraints.copy()
        self.enable_multi_objective = ga_config.enable_multi_objective
        self.objectives = ga_config.ga_objectives.copy()
        self.objective_weights = ga_config.ga_objective_weights.copy()

        # フィットネス共有設定
        self.enable_fitness_sharing = ga_config.fitness_sharing[
            "enable_fitness_sharing"
        ]
        self.sharing_radius = ga_config.fitness_sharing["sharing_radius"]
        self.sharing_alpha = ga_config.fitness_sharing["sharing_alpha"]

        # パラメータ範囲
        self.parameter_ranges = ga_config.parameter_ranges.copy()
        self.threshold_ranges = ga_config.threshold_ranges.copy()

        # TPSL設定を適用
        if self.tpsl_method_constraints is None:
            from ..constants import (
                GA_DEFAULT_TPSL_METHOD_CONSTRAINTS,
                GA_TPSL_ATR_MULTIPLIER_RANGE,
                GA_TPSL_RR_RANGE,
                GA_TPSL_SL_RANGE,
                GA_TPSL_TP_RANGE,
            )

            self.tpsl_method_constraints = GA_DEFAULT_TPSL_METHOD_CONSTRAINTS.copy()

        if self.tpsl_sl_range is None:
            self.tpsl_sl_range = GA_TPSL_SL_RANGE.copy()

        if self.tpsl_tp_range is None:
            self.tpsl_tp_range = GA_TPSL_TP_RANGE.copy()

        if self.tpsl_rr_range is None:
            self.tpsl_rr_range = GA_TPSL_RR_RANGE.copy()

        if self.tpsl_atr_multiplier_range is None:
            self.tpsl_atr_multiplier_range = GA_TPSL_ATR_MULTIPLIER_RANGE.copy()

        # フォールバック設定
        self.fallback_start_date = getattr(
            ga_config, "fallback_start_date", "2024-01-01"
        )
        self.fallback_end_date = getattr(ga_config, "fallback_end_date", "2024-04-09")

        # レジーム適応設定（デフォルトはFalse、後続ステップで設定可能にする）
        self.regime_adaptation_enabled = getattr(
            ga_config, "regime_adaptation_enabled", False
        )

        # 許可指標リスト
        if not self.allowed_indicators:
            try:
                from app.services.indicators import TechnicalIndicatorService

                indicator_service = TechnicalIndicatorService()
                self.allowed_indicators = list(
                    indicator_service.get_supported_indicators().keys()
                )
            except Exception:
                # Fallback: 設定から取得
                self.allowed_indicators = config.indicators.valid_indicator_types[:]

    @classmethod
    def from_auto_strategy_config(cls, config: AutoStrategyConfig) -> "GAConfig":
        """
        AutoStrategyConfigからGAConfigを作成

        Args:
            config: AutoStrategyConfigインスタンス

        Returns:
            GAConfigインスタンス
        """
        instance = cls()
        instance.apply_auto_strategy_config(config)
        return instance

    def get_default_values(self) -> Dict[str, Any]:
        """BaseConfig用のデフォルト値を取得（自動生成を利用）"""
        # フィールドから自動生成したデフォルト値を取得
        defaults = self.get_default_values_from_fields()
        # GASettingsの設定も取得して統合
        ga_settings = self.get_ga_settings()
        if ga_settings:
            ga_defaults = ga_settings.get_default_values()
            # 統合（GAConfig特有のフィールドを追加）
            integrated_defaults = {
                **defaults,
                **ga_defaults,
                "primary_metric": self.primary_metric,
            }
        else:
            integrated_defaults = {
                **defaults,
                "primary_metric": self.primary_metric,
            }
        return integrated_defaults

    # BaseConfigのメソッドをオーバーライド（既存機能を保持）
    def to_json(self) -> str:
        """JSON文字列に変換（BaseConfigの機能を活用）"""
        return super().to_json()

    @classmethod
    def from_json(cls, json_str: str) -> "GAConfig":
        """JSON文字列から復元（BaseConfigの機能を活用）"""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"JSON復元エラー: {e}", exc_info=True)
            raise ValueError(f"JSON からの復元に失敗しました: {e}")


@dataclass
class GAProgress:
    """
    GA実行進捗情報

    リアルタイム進捗表示用のデータ構造
    """

    experiment_id: str
    current_generation: int
    total_generations: int
    best_fitness: float
    average_fitness: float
    execution_time: float
    estimated_remaining_time: float
    status: str = "running"  # "running", "completed", "error"
    best_strategy_preview: Optional[Dict[str, Any]] = None

    @property
    def progress_percentage(self) -> float:
        """進捗率（0-100）"""
        if self.total_generations == 0:
            return 0.0
        return (self.current_generation / self.total_generations) * 100

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "experiment_id": self.experiment_id,
            "current_generation": self.current_generation,
            "total_generations": self.total_generations,
            "best_fitness": self.best_fitness,
            "average_fitness": self.average_fitness,
            "execution_time": self.execution_time,
            "estimated_remaining_time": self.estimated_remaining_time,
            "progress_percentage": self.progress_percentage,
            "status": self.status,
            "best_strategy_preview": self.best_strategy_preview,
        }

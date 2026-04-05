"""
GA設定サブクラス

GAConfig の設定項目を意味的にグループ化するサブデータクラス群。
GAConfig のフラットフィールドと併存し、ネスト辞書からの復元にも対応する。
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


def _filter_known_fields(cls, data: dict, config_name: str) -> dict:
    """既知フィールドのみを残し、未知キーは警告する。"""
    known = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = sorted(key for key in data.keys() if key not in known)
    if unknown:
        logger.warning(
            "%s の未対応キーを無視しました: %s",
            config_name,
            ", ".join(unknown),
        )
    return {key: value for key, value in data.items() if key in known}


@dataclass
class MutationConfig:
    """突然変異関連設定。"""

    rate: float = 0.1
    crossover_field_selection_probability: float = 0.5
    indicator_param_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
    risk_param_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
    indicator_add_delete_probability: float = 0.3
    indicator_add_vs_delete_probability: float = 0.5
    condition_change_multiplier: float = 1.0
    condition_selection_probability: float = 0.5
    condition_operator_switch_probability: float = 0.2
    tpsl_gene_creation_multiplier: float = 0.2
    position_sizing_gene_creation_multiplier: float = 0.2
    adaptive_variance_threshold: float = 0.001
    adaptive_decrease_multiplier: float = 0.8
    adaptive_increase_multiplier: float = 1.2
    valid_condition_operators: List[str] = field(
        default_factory=lambda: [
            ">",
            "<",
            ">=",
            "<=",
            "==",
            "!=",
            "CROSS_UP",
            "CROSS_DOWN",
        ]
    )

    @classmethod
    def from_dict(cls, data: dict) -> "MutationConfig":
        """
        辞書からMutationConfigインスタンスを生成

        Args:
            data: 設定値を含む辞書。未知のキーは無視されます。

        Returns:
            初期化されたMutationConfigインスタンス
        """
        filtered = _filter_known_fields(cls, data, "MutationConfig")
        return cls(**filtered)


@dataclass
class EvaluationConfig:
    """評価・検証関連設定。"""

    enable_parallel: bool = True
    max_workers: Optional[int] = None
    timeout: float = 300.0
    oos_split_ratio: float = 0.0
    oos_fitness_weight: float = 0.5
    enable_walk_forward: bool = False
    wfa_n_folds: int = 5
    wfa_train_ratio: float = 0.7
    wfa_anchored: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationConfig":
        """
        辞書からEvaluationConfigインスタンスを生成

        Args:
            data: 設定値を含む辞書。未知のキーは無視されます。

        Returns:
            初期化されたEvaluationConfigインスタンス
        """
        filtered = _filter_known_fields(cls, data, "EvaluationConfig")
        return cls(**filtered)


@dataclass
class HybridConfig:
    """ハイブリッドGA+ML関連設定。"""

    mode: bool = False
    model_type: str = "lightgbm"
    model_types: Optional[List[str]] = None
    volatility_gate_enabled: bool = False
    volatility_model_path: Optional[str] = None
    ml_filter_enabled: bool = False
    ml_model_path: Optional[str] = None
    preprocess_features: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "HybridConfig":
        """
        辞書からHybridConfigインスタンスを生成

        Args:
            data: 設定値を含む辞書。未知のキーは無視されます。

        Returns:
            初期化されたHybridConfigインスタンス
        """
        filtered = _filter_known_fields(cls, data, "HybridConfig")
        return cls(**filtered)


@dataclass
class TuningConfig:
    """パラメータチューニング（Optuna）関連設定。"""

    enabled: bool = True
    n_trials: int = 30
    elite_count: int = 3
    use_wfa: bool = True
    include_indicators: bool = True
    include_tpsl: bool = True
    include_thresholds: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "TuningConfig":
        """
        辞書からTuningConfigインスタンスを生成

        Args:
            data: 設定値を含む辞書。未知のキーは無視されます。

        Returns:
            初期化されたTuningConfigインスタンス
        """
        filtered = _filter_known_fields(cls, data, "TuningConfig")
        return cls(**filtered)


@dataclass
class TwoStageSelectionConfig:
    """二段階選抜関連設定。"""

    enabled: bool = True
    elite_count: int = 3
    candidate_pool_size: int = 5
    min_pass_rate: float = 0.5

    @classmethod
    def from_dict(cls, data: dict) -> "TwoStageSelectionConfig":
        """
        辞書からTwoStageSelectionConfigインスタンスを生成

        Args:
            data: 設定値を含む辞書。未知のキーは無視されます。

        Returns:
            初期化されたTwoStageSelectionConfigインスタンス
        """
        filtered = _filter_known_fields(cls, data, "TwoStageSelectionConfig")
        return cls(**filtered)


@dataclass
class RobustnessConfig:
    """robustness 評価関連設定。"""

    validation_symbols: Optional[List[str]] = None
    regime_windows: List[dict] = field(default_factory=list)
    stress_slippage: List[float] = field(default_factory=list)
    stress_commission_multipliers: List[float] = field(default_factory=list)
    aggregate_method: str = "robust"

    @classmethod
    def from_dict(cls, data: dict) -> "RobustnessConfig":
        """
        辞書からRobustnessConfigインスタンスを生成

        Args:
            data: 設定値を含む辞書。未知のキーは無視されます。

        Returns:
            初期化されたRobustnessConfigインスタンス
        """
        filtered = _filter_known_fields(cls, data, "RobustnessConfig")
        return cls(**filtered)

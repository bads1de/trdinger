"""
遺伝的アルゴリズム設定モデル

GA実行時の各種パラメータを単一のクラスで管理します。
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

from app.core.services.indicators import TechnicalIndicatorService


@dataclass
class GAConfig:
    """
    遺伝的アルゴリズム設定

    GA実行時の全パラメータをフラットな構造で管理します。
    """

    # 進化アルゴリズム設定
    population_size: int = 10
    generations: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_size: int = 2

    # 評価設定
    fitness_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }
    )
    primary_metric: str = "sharpe_ratio"
    ga_objective: str = "Sharpe Ratio"  # フロントエンドとの互換性のため
    fitness_constraints: Dict[str, float] = field(
        default_factory=lambda: {
            "min_trades": 10,
            "max_drawdown_limit": 0.3,
            "min_sharpe_ratio": 1.0,
        }
    )

    # 指標設定
    max_indicators: int = 3
    allowed_indicators: List[str] = field(
        default_factory=lambda: list(
            TechnicalIndicatorService().get_supported_indicators().keys()
        )
    )

    # パラメータ範囲設定
    parameter_ranges: Dict[str, List[float]] = field(
        default_factory=lambda: {
            # 基本パラメータ
            "period": [5, 200],
            "fast_period": [5, 20],
            "slow_period": [20, 50],
            "signal_period": [5, 15],
            # 特殊パラメータ
            "std_dev": [1.5, 2.5],
            "k_period": [10, 20],
            "d_period": [3, 7],
            "slowing": [1, 5],
            # 閾値パラメータ
            "overbought": [70, 90],
            "oversold": [10, 30],
        }
    )
    threshold_ranges: Dict[str, List[float] | Dict[str, List[float]]] = field(
        default_factory=lambda: {
            "oscillator_0_100": [20, 80],
            "oscillator_plus_minus_100": [-100, 100],
            "momentum_zero_centered": [-0.5, 0.5],
            "funding_rate": [0.0001, 0.0005, 0.001, -0.0001, -0.0005, -0.001],
            "open_interest": [1000000, 5000000, 10000000, 50000000],
            "price_ratio": [0.95, 1.05],
        }
    )

    # 遺伝子生成設定
    min_indicators: int = 1
    min_conditions: int = 1
    max_conditions: int = 3
    price_data_weight: int = 3
    volume_data_weight: int = 1
    oi_fr_data_weight: int = 1
    numeric_threshold_probability: float = 0.8
    min_compatibility_score: float = 0.8
    strict_compatibility_score: float = 0.9
    stop_loss_range: List[float] = field(default_factory=lambda: [0.02, 0.05])
    take_profit_range: List[float] = field(default_factory=lambda: [0.01, 0.15])
    position_size_range: List[float] = field(default_factory=lambda: [0.1, 0.5])

    # TP/SL GA最適化範囲設定（ユーザー設定ではなくGA制約）
    tpsl_method_constraints: List[str] = field(
        default_factory=lambda: [
            "fixed_percentage",
            "risk_reward_ratio",
            "volatility_based",
            "statistical",
            "adaptive",
        ]
    )  # GA最適化で使用可能なTP/SLメソッド
    tpsl_sl_range: List[float] = field(
        default_factory=lambda: [0.01, 0.08]
    )  # SL範囲（1%-8%）
    tpsl_tp_range: List[float] = field(
        default_factory=lambda: [0.02, 0.20]
    )  # TP範囲（2%-20%）
    tpsl_rr_range: List[float] = field(
        default_factory=lambda: [1.2, 4.0]
    )  # リスクリワード比範囲
    tpsl_atr_multiplier_range: List[float] = field(
        default_factory=lambda: [1.0, 4.0]
    )  # ATR倍率範囲

    # 実行設定
    parallel_processes: Optional[int] = None
    random_state: Optional[int] = None
    log_level: str = "WARNING"
    save_intermediate_results: bool = True
    # enable_detailed_logging: bool = True
    progress_callback: Optional[Callable[["GAProgress"], None]] = None

    def validate(self) -> tuple[bool, List[str]]:
        """
        設定の妥当性を検証

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # 進化設定の検証
        if self.population_size <= 0:
            errors.append("個体数は正の整数である必要があります")
        elif self.population_size > 1000:
            errors.append(
                "個体数は1000以下である必要があります（パフォーマンス上の制約）"
            )

        if self.generations <= 0:
            errors.append("世代数は正の整数である必要があります")
        elif self.generations > 500:
            errors.append(
                "世代数は500以下である必要があります（パフォーマンス上の制約）"
            )

        if not 0 <= self.crossover_rate <= 1:
            errors.append("交叉率は0-1の範囲である必要があります")

        if not 0 <= self.mutation_rate <= 1:
            errors.append("突然変異率は0-1の範囲である必要があります")

        if self.elite_size < 0 or self.elite_size >= self.population_size:
            errors.append("エリート保存数は0以上、個体数未満である必要があります")

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
        if self.max_indicators <= 0:
            errors.append("最大指標数は正の整数である必要があります")
        elif self.max_indicators > 10:
            errors.append(
                "最大指標数は10以下である必要があります（パフォーマンス上の制約）"
            )

        if not self.allowed_indicators:
            errors.append("許可された指標リストが空です")
        else:
            valid_indicators = set(
                TechnicalIndicatorService().get_supported_indicators().keys()
            )
            invalid_indicators = set(self.allowed_indicators) - valid_indicators
            if invalid_indicators:
                errors.append(f"無効な指標が含まれています: {invalid_indicators}")

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
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GAConfig":
        """辞書から復元"""
        # allowed_indicatorsが空の場合はデフォルトの指標リストを使用
        indicator_service = TechnicalIndicatorService()
        allowed_indicators = data.get("allowed_indicators") or list(
            indicator_service.get_supported_indicators().keys()
        )

        # デフォルト値を使用してフラットな構造で復元
        return cls(
            population_size=data.get("population_size", 100),
            generations=data.get("generations", 50),
            crossover_rate=data.get("crossover_rate", 0.8),
            mutation_rate=data.get("mutation_rate", 0.1),
            elite_size=data.get("elite_size", 10),
            fitness_weights=data.get("fitness_weights", {}),
            primary_metric=data.get("primary_metric", "sharpe_ratio"),
            fitness_constraints=data.get("fitness_constraints", {}),
            max_indicators=data.get("max_indicators", 5),
            allowed_indicators=allowed_indicators,
            parameter_ranges=data.get("parameter_ranges", {}),
            threshold_ranges=data.get("threshold_ranges", {}),
            min_indicators=data.get("min_indicators", 1),
            min_conditions=data.get("min_conditions", 1),
            max_conditions=data.get("max_conditions", 3),
            parallel_processes=data.get("parallel_processes"),
            random_state=data.get("random_state"),
            log_level=data.get("log_level", "INFO"),
            save_intermediate_results=data.get("save_intermediate_results", True),
            # enable_detailed_logging=data.get("enable_detailed_logging", True),
        )

    def to_json(self) -> str:
        """JSON文字列に変換"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "GAConfig":
        """JSON文字列から復元"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def create_default(cls) -> "GAConfig":
        """デフォルト設定を作成"""
        return cls()

    @classmethod
    def create_fast(cls) -> "GAConfig":
        """高速実行用設定を作成（オートストラテジー用デフォルト）"""
        return cls(
            population_size=10,
            generations=5,
            elite_size=2,
            max_indicators=3,
        )

    @classmethod
    def create_thorough(cls) -> "GAConfig":
        """徹底的な探索用設定を作成"""
        return cls(
            population_size=200,
            generations=100,
            crossover_rate=0.85,
            mutation_rate=0.05,
            elite_size=20,
            max_indicators=5,
            log_level="INFO",
            save_intermediate_results=True,
        )


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

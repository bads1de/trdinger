"""
遺伝的アルゴリズム設定モデル

GA実行時の各種パラメータを管理します。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json

from ...indicators.constants import ALL_INDICATORS


@dataclass
class GAConfig:
    """
    遺伝的アルゴリズム設定

    GA実行時の全パラメータを管理します。
    """

    # 基本GA設定
    population_size: int = 100  # 個体数
    generations: int = 50  # 世代数
    crossover_rate: float = 0.8  # 交叉率
    mutation_rate: float = 0.1  # 突然変異率
    elite_size: int = 10  # エリート保存数

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

    # 制約条件
    max_indicators: int = 5  # 最大指標数
    allowed_indicators: List[str] = field(default_factory=lambda: ALL_INDICATORS.copy())

    # パラメータ範囲
    parameter_ranges: Dict[str, List[float]] = field(
        default_factory=lambda: {
            # 移動平均系
            "SMA_period": [5, 200],
            "EMA_period": [5, 200],
            "WMA_period": [5, 200],
            "HMA_period": [9, 50],
            "KAMA_period": [10, 50],
            "TEMA_period": [10, 50],
            "DEMA_period": [10, 50],
            "T3_period": [5, 30],
            "T3_vfactor": [0.5, 0.9],
            "MAMA_period": [20, 40],
            "MAMA_fastlimit": [0.4, 0.6],
            "MAMA_slowlimit": [0.02, 0.08],
            "ZLEMA_period": [10, 50],
            "TRIMA_period": [14, 50],
            # モメンタム系
            "RSI_period": [10, 30],
            "RSI_overbought": [70, 90],
            "RSI_oversold": [10, 30],
            "STOCH_k_period": [10, 20],
            "STOCH_d_period": [3, 7],
            "STOCHRSI_period": [14, 21],
            "STOCHRSI_fastk_period": [3, 5],
            "STOCHRSI_fastd_period": [3, 5],
            "STOCHF_period": [5, 14],
            "STOCHF_fastd_period": [3, 5],
            "CCI_period": [10, 25],
            "WILLR_period": [10, 20],
            "MOMENTUM_period": [5, 20],
            "MOM_period": [5, 20],
            "ROC_period": [5, 25],
            "ROCP_period": [10, 20],
            "ROCR_period": [10, 20],
            "ADX_period": [10, 25],
            "AROON_period": [10, 25],
            "MFI_period": [10, 25],
            "CMO_period": [14, 28],
            "TRIX_period": [14, 30],
            "ULTOSC_period": [7, 28],
            "PLUS_DI_period": [14, 30],
            "MINUS_DI_period": [14, 30],
            # ボラティリティ系
            "MACD_fast": [5, 20],
            "MACD_slow": [20, 50],
            "MACD_signal": [5, 15],
            "BB_period": [15, 25],
            "BB_std_dev": [1.5, 2.5],
            "KELTNER_period": [14, 20],
            "KELTNER_multiplier": [1.5, 2.5],
            "ATR_period": [10, 25],
            "NATR_period": [10, 25],
            "TRANGE_period": [10, 25],
            "STDDEV_period": [10, 30],
            "DONCHIAN_period": [14, 30],
            # 出来高系
            "VWMA_period": [10, 30],
            "VWAP_period": [14, 30],
            "PVT_period": [1, 1],
            "EMV_period": [14, 30],
            "ADOSC_fast": [3, 7],
            "ADOSC_slow": [8, 15],
            # 価格変換系
            "MIDPOINT_period": [14, 30],
            "MIDPRICE_period": [14, 30],
            # その他
            "BOP_period": [1, 1],
            "APO_period": [12, 26],
            "APO_slow_period": [26, 50],
            "PPO_period": [12, 26],
            "PPO_slow_period": [26, 50],
            "AROONOSC_period": [14, 25],
            "DX_period": [14, 21],
            "ADXR_period": [14, 21],
        }
    )

    # 制約条件
    fitness_constraints: Dict[str, float] = field(
        default_factory=lambda: {
            "min_trades": 10,
            "max_drawdown_limit": 0.3,
            "min_sharpe_ratio": 0.5,
        }
    )

    # 実行設定
    parallel_processes: Optional[int] = None  # None = CPU数自動検出
    random_state: Optional[int] = None  # 再現性のためのシード

    # 進捗・ログ設定
    progress_callback: Optional[callable] = None
    log_level: str = "INFO"
    save_intermediate_results: bool = True

    def validate(self) -> tuple[bool, List[str]]:
        """
        設定の妥当性を検証

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # 基本パラメータの範囲チェック
        if self.population_size <= 0:
            errors.append("個体数は正の整数である必要があります")
        if self.generations <= 0:
            errors.append("世代数は正の整数である必要があります")
        if not 0 <= self.crossover_rate <= 1:
            errors.append("交叉率は0-1の範囲である必要があります")
        if not 0 <= self.mutation_rate <= 1:
            errors.append("突然変異率は0-1の範囲である必要があります")
        if self.elite_size < 0 or self.elite_size >= self.population_size:
            errors.append("エリート保存数は0以上、個体数未満である必要があります")

        # フィットネス重みの検証
        if abs(sum(self.fitness_weights.values()) - 1.0) > 0.01:
            errors.append("フィットネス重みの合計は1.0である必要があります")

        # 指標制約の検証
        if self.max_indicators <= 0:
            errors.append("最大指標数は正の整数である必要があります")
        if not self.allowed_indicators:
            errors.append("使用可能指標が設定されていません")

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
            "fitness_constraints": self.fitness_constraints,
            "parallel_processes": self.parallel_processes,
            "random_state": self.random_state,
            "log_level": self.log_level,
            "save_intermediate_results": self.save_intermediate_results,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GAConfig":
        """辞書から復元"""
        return cls(
            population_size=data.get("population_size", 100),
            generations=data.get("generations", 50),
            crossover_rate=data.get("crossover_rate", 0.8),
            mutation_rate=data.get("mutation_rate", 0.1),
            elite_size=data.get("elite_size", 10),
            fitness_weights=data.get("fitness_weights", {}),
            primary_metric=data.get("primary_metric", "sharpe_ratio"),
            max_indicators=data.get("max_indicators", 5),
            allowed_indicators=data.get("allowed_indicators", []),
            parameter_ranges=data.get("parameter_ranges", {}),
            fitness_constraints=data.get("fitness_constraints", {}),
            parallel_processes=data.get("parallel_processes"),
            random_state=data.get("random_state"),
            log_level=data.get("log_level", "INFO"),
            save_intermediate_results=data.get("save_intermediate_results", True),
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
        """高速実行用設定を作成"""
        return cls(population_size=50, generations=30, elite_size=5)

    @classmethod
    def create_thorough(cls) -> "GAConfig":
        """徹底的な探索用設定を作成"""
        return cls(population_size=200, generations=100, elite_size=20)


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

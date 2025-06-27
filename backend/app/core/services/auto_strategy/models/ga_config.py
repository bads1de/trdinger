"""
遺伝的アルゴリズム設定モデル

GA実行時の各種パラメータを管理します。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
import json

from ...indicators.constants import ALL_INDICATORS


@dataclass
class GAConfig:
    """
    遺伝的アルゴリズム設定

    GA実行時の全パラメータを管理します。
    """

    # 基本GA設定（高速化最適化済み）
    population_size: int = 10  # 個体数
    generations: int = 5  # 世代数
    crossover_rate: float = 0.8  # 交叉率
    mutation_rate: float = 0.1  # 突然変異率
    elite_size: int = 2  # エリート保存数

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
    max_indicators: int = 3  # 最大指標数（
    allowed_indicators: List[str] = field(default_factory=lambda: ALL_INDICATORS.copy())

    # パラメータ範囲（JSON形式対応）
    parameter_ranges: Dict[str, List[float]] = field(
        default_factory=lambda: {
            # 統一されたパラメータ名（JSON形式準拠）
            # 基本パラメータ
            "period": [5, 200],  # 一般的な期間パラメータ
            "fast_period": [5, 20],  # 短期期間
            "slow_period": [20, 50],  # 長期期間
            "signal_period": [5, 15],  # シグナル期間
            # 特殊パラメータ
            "std_dev": [1.5, 2.5],  # 標準偏差（BB用）
            "k_period": [10, 20],  # %K期間（STOCH用）
            "d_period": [3, 7],  # %D期間（STOCH用）
            "slowing": [1, 5],  # スローイング（STOCH用）
            # 閾値パラメータ（条件生成用）
            "overbought": [70, 90],  # 買われすぎ閾値
            "oversold": [10, 30],  # 売られすぎ閾値
            # 後方互換性のためのレガシーパラメータ名も保持
            "SMA_period": [5, 200],
            "EMA_period": [5, 200],
            "MACD_fast": [5, 20],
            "MACD_slow": [20, 50],
            "MACD_signal": [5, 15],
            "BB_period": [15, 25],
            "BB_std_dev": [1.5, 2.5],
            "RSI_period": [10, 30],
            "RSI_overbought": [70, 90],
            "RSI_oversold": [10, 30],
            "STOCH_k_period": [10, 20],
            "STOCH_d_period": [3, 7],
            "CCI_period": [10, 25],
            "ADX_period": [10, 25],
            "ATR_period": [10, 25],
            "OBV_period": [1, 1],  # OBVは期間パラメータを持たないが、統一性のため
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
    progress_callback: Optional[Callable[["GAProgress"], None]] = None
    log_level: str = "WARNING"  # パフォーマンス最適化のためWARNINGに変更
    save_intermediate_results: bool = True

    # パフォーマンス設定
    enable_detailed_logging: bool = False  # 詳細ログの有効/無効

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
        # allowed_indicatorsが空の場合は自動設定されるため、エラーにしない

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
        # allowed_indicatorsが空の場合はデフォルトの指標リストを使用
        allowed_indicators = data.get("allowed_indicators", [])
        if not allowed_indicators:
            allowed_indicators = ALL_INDICATORS.copy()

        return cls(
            population_size=data.get("population_size", 100),
            generations=data.get("generations", 50),
            crossover_rate=data.get("crossover_rate", 0.8),
            mutation_rate=data.get("mutation_rate", 0.1),
            elite_size=data.get("elite_size", 10),
            fitness_weights=data.get("fitness_weights", {}),
            primary_metric=data.get("primary_metric", "sharpe_ratio"),
            max_indicators=data.get("max_indicators", 5),
            allowed_indicators=allowed_indicators,
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
        """高速実行用設定を作成（オートストラテジー用デフォルト）"""
        return cls(population_size=10, generations=5, elite_size=2, max_indicators=3)

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
            parallel_processes=None,  # デフォルトはCPU数
        )

    def get_parameter_range(self, param_name: str) -> List[float]:
        """
        パラメータ範囲を取得（JSON形式とレガシー形式の両方をサポート）

        Args:
            param_name: パラメータ名

        Returns:
            パラメータ範囲 [min, max]
        """
        # 直接的な名前での検索
        if param_name in self.parameter_ranges:
            return self.parameter_ranges[param_name]

        # JSON形式からレガシー形式への変換マッピング
        json_to_legacy_mapping = {
            "period": [
                "SMA_period",
                "EMA_period",
                "RSI_period",
                "CCI_period",
                "ADX_period",
                "ATR_period",
            ],
            "fast_period": ["MACD_fast"],
            "slow_period": ["MACD_slow"],
            "signal_period": ["MACD_signal"],
            "std_dev": ["BB_std_dev"],
            "k_period": ["STOCH_k_period"],
            "d_period": ["STOCH_d_period"],
        }

        # JSON形式のパラメータ名からレガシー形式を検索
        for json_param, legacy_params in json_to_legacy_mapping.items():
            if param_name == json_param and legacy_params:
                legacy_param = legacy_params[0]  # 最初のマッピングを使用
                if legacy_param in self.parameter_ranges:
                    return self.parameter_ranges[legacy_param]

        # デフォルト範囲を返す
        return [1, 100]

    def normalize_parameter_name(self, indicator_type: str, param_name: str) -> str:
        """
        パラメータ名を正規化（JSON形式に統一）

        Args:
            indicator_type: インジケーター名
            param_name: パラメータ名

        Returns:
            正規化されたパラメータ名
        """
        # レガシー形式からJSON形式への変換マッピング
        legacy_to_json_mapping = {
            f"{indicator_type}_period": "period",
            f"{indicator_type}_fast": "fast_period",
            f"{indicator_type}_slow": "slow_period",
            f"{indicator_type}_signal": "signal_period",
            f"{indicator_type}_std_dev": "std_dev",
            f"{indicator_type}_k_period": "k_period",
            f"{indicator_type}_d_period": "d_period",
        }

        # 直接マッピングがある場合
        if param_name in legacy_to_json_mapping:
            return legacy_to_json_mapping[param_name]

        # 一般的なマッピング
        if param_name.endswith("_period"):
            return "period"
        elif param_name.endswith("_fast"):
            return "fast_period"
        elif param_name.endswith("_slow"):
            return "slow_period"
        elif param_name.endswith("_signal"):
            return "signal_period"

        # そのまま返す
        return param_name


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

"""
ベースオプティマイザー抽象クラス

全ての最適化手法の共通インターフェースを定義します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """最適化結果"""

    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    optimization_time: float
    convergence_info: Dict[str, Any]


@dataclass
class ParameterSpace:
    """パラメータ空間の定義"""

    type: str  # "real", "integer", "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    categories: Optional[List[Any]] = None

    def __post_init__(self):
        """バリデーション"""
        if self.type in ["real", "integer"] and (self.low is None or self.high is None):
            raise ValueError(f"{self.type}型のパラメータにはlowとhighが必要です")
        if self.type == "categorical" and not self.categories:
            raise ValueError("categorical型のパラメータにはcategoriesが必要です")


class BaseOptimizer(ABC):
    """
    ベースオプティマイザー抽象クラス

    全ての最適化手法（ベイジアン、グリッドサーチ、ランダムサーチ）の
    共通インターフェースを定義します。
    """

    def __init__(self):
        """初期化"""
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_result: Optional[OptimizationResult] = None

    @abstractmethod
    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        parameter_space: Dict[str, ParameterSpace],
        n_calls: int = 50,
        **kwargs: Any
    ) -> OptimizationResult:
        """
        最適化を実行

        Args:
            objective_function: 目的関数（パラメータを受け取りスコアを返す）
            parameter_space: パラメータ空間の定義
            n_calls: 最適化試行回数
            **kwargs: 追加のオプション

        Returns:
            最適化結果

        Raises:
            ValueError: パラメータが無効な場合
            RuntimeError: 最適化に失敗した場合
        """
        pass

    def validate_parameter_space(self, parameter_space: Dict[str, ParameterSpace]) -> None:
        """
        パラメータ空間の妥当性を検証

        Args:
            parameter_space: パラメータ空間の定義

        Raises:
            ValueError: パラメータ空間が無効な場合
        """
        if not parameter_space:
            raise ValueError("パラメータ空間が空です")

        for param_name, param_config in parameter_space.items():
            if not isinstance(param_config, ParameterSpace):
                raise ValueError(f"パラメータ '{param_name}' の設定が無効です")

            # 型固有のバリデーション
            if param_config.type == "real":
                if param_config.low >= param_config.high:
                    raise ValueError(f"パラメータ '{param_name}': lowはhighより小さい必要があります")
            elif param_config.type == "integer":
                if param_config.low >= param_config.high:
                    raise ValueError(f"パラメータ '{param_name}': lowはhighより小さい必要があります")
                if not isinstance(param_config.low, int) or not isinstance(param_config.high, int):
                    raise ValueError(f"パラメータ '{param_name}': integer型のlowとhighは整数である必要があります")
            elif param_config.type == "categorical":
                if not param_config.categories or len(param_config.categories) == 0:
                    raise ValueError(f"パラメータ '{param_name}': categoriesが空です")
            else:
                raise ValueError(f"パラメータ '{param_name}': 未対応の型 '{param_config.type}'")

    def validate_objective_function(self, objective_function: Callable[[Dict[str, Any]], float]) -> None:
        """
        目的関数の妥当性を検証

        Args:
            objective_function: 目的関数

        Raises:
            ValueError: 目的関数が無効な場合
        """
        if not callable(objective_function):
            raise ValueError("目的関数は呼び出し可能である必要があります")

    def _log_optimization_start(self, method_name: str, n_calls: int) -> None:
        """最適化開始のログ出力"""
        logger.info(f"{method_name}最適化を開始: 試行回数={n_calls}")

    def _log_optimization_end(self, method_name: str, best_score: float, total_time: float) -> None:
        """最適化終了のログ出力"""
        logger.info(f"{method_name}最適化完了: ベストスコア={best_score:.4f}, 実行時間={total_time:.2f}秒")

    def _create_optimization_result(
        self,
        best_params: Dict[str, Any],
        best_score: float,
        history: List[Dict[str, Any]],
        optimization_time: float,
        convergence_info: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        最適化結果オブジェクトを作成

        Args:
            best_params: 最適パラメータ
            best_score: 最適スコア
            history: 最適化履歴
            optimization_time: 最適化時間
            convergence_info: 収束情報

        Returns:
            最適化結果
        """
        if convergence_info is None:
            convergence_info = {}

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=history,
            total_evaluations=len(history),
            optimization_time=optimization_time,
            convergence_info=convergence_info
        )

    def get_method_name(self) -> str:
        """最適化手法名を取得"""
        return self.__class__.__name__.replace("Optimizer", "")

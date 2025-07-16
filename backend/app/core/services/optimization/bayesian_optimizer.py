"""
ベイズ最適化エンジン

GAパラメータとMLハイパーパラメータの自動調整を行います。
"""

import logging
import numpy as np

from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args, OptimizeResult

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


class BayesianOptimizer:
    """
    ベイズ最適化エンジン

    GAパラメータやMLハイパーパラメータの効率的な最適化を行います。
    """

    def __init__(self):
        """初期化"""
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_result: Optional[OptimizationResult] = None

        # 最適化設定
        self.config = {
            "n_calls": 50,  # 最適化試行回数
            "n_initial_points": 10,  # 初期ランダム試行回数
            "acq_func": "EI",  # 獲得関数（Expected Improvement）
            "random_state": 42,  # 乱数シード
            "n_jobs": 1,  # 並列実行数
        }

    def optimize_ml_hyperparameters(
        self,
        model_type: str,
        objective_function: Callable[[Dict[str, Any]], float],
        parameter_space: Optional[Dict[str, Any]] = None,
        n_calls: int = 30,
    ) -> OptimizationResult:
        """
        MLハイパーパラメータの最適化

        Args:
            model_type: モデルタイプ
            objective_function: 目的関数
            parameter_space: パラメータ空間
            n_calls: 最適化試行回数

        Returns:
            最適化結果
        """
        try:
            if parameter_space is None:
                parameter_space = self._get_default_ml_parameter_space(model_type)

            logger.info(f"{model_type}のハイパーパラメータ最適化を開始")
            start_time = datetime.now()

            result = self._optimize_with_skopt(
                objective_function, parameter_space, n_calls
            )

            end_time = datetime.now()
            optimization_time = (end_time - start_time).total_seconds()

            optimization_result = OptimizationResult(
                best_params=result["best_params"],
                best_score=result["best_score"],
                optimization_history=result["history"],
                total_evaluations=len(result["history"]),
                optimization_time=optimization_time,
                convergence_info=result.get("convergence_info", {}),
            )

            logger.info(
                f"{model_type}ハイパーパラメータ最適化完了: ベストスコア={result['best_score']:.4f}"
            )

            return optimization_result

        except Exception as e:
            logger.error(f"MLハイパーパラメータ最適化中にエラーが発生しました: {e}")
            raise

    def _optimize_with_skopt(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        parameter_space: Dict[str, Any],
        n_calls: int,
    ) -> Dict[str, Any]:
        """scikit-optimizeを使用した最適化"""
        try:
            # パラメータ空間を定義
            dimensions = []
            param_names = []

            for param_name, param_config in parameter_space.items():
                param_names.append(param_name)

                if param_config["type"] == "real":
                    dimensions.append(
                        Real(param_config["low"], param_config["high"], name=param_name)
                    )
                elif param_config["type"] == "integer":
                    dimensions.append(
                        Integer(
                            param_config["low"], param_config["high"], name=param_name
                        )
                    )
                elif param_config["type"] == "categorical":
                    dimensions.append(
                        Categorical(param_config["categories"], name=param_name)
                    )

            # 目的関数をラップ
            @use_named_args(dimensions)
            def wrapped_objective(**params):
                try:
                    score = objective_function(params)
                    # 最小化問題に変換（スコアが高いほど良い場合は負の値を返す）
                    return -score
                except Exception as e:
                    logger.warning(f"目的関数評価中にエラーが発生しました: {e}")
                    return 1000  # 大きなペナルティ

            # 最適化実行
            result: Optional[OptimizeResult] = gp_minimize(
                func=wrapped_objective,
                dimensions=dimensions,
                n_calls=n_calls,
                n_initial_points=min(10, n_calls // 3),
                acq_func=self.config["acq_func"],
                random_state=self.config["random_state"],
            )

            if result is None:
                logger.error(
                    "gp_minimize が予期せず None を返しました。最適化に失敗しました。"
                )
                raise RuntimeError("最適化失敗: gp_minimize が None を返しました。")

            # 結果を整理
            best_params = dict(zip(param_names, result.x))
            best_score = -result.fun  # 元のスコアに戻す

            history = []
            for i, (params, score) in enumerate(zip(result.x_iters, result.func_vals)):
                param_dict = dict(zip(param_names, params))
                history.append(
                    {
                        "iteration": i + 1,
                        "params": param_dict,
                        "score": -score,  # 元のスコアに戻す
                    }
                )

            return {
                "best_params": best_params,
                "best_score": best_score,
                "history": history,
                "convergence_info": {
                    "converged": len(result.func_vals) >= n_calls,
                    "best_iteration": np.argmin(result.func_vals) + 1,
                },
            }

        except Exception as e:
            logger.error(f"scikit-optimize最適化中にエラーが発生しました: {e}")
            raise

    # フォールバックメソッドは削除（scikit-optimizeが確実に利用可能なため）

    def _get_default_ml_parameter_space(self, model_type: str) -> Dict[str, Any]:
        """デフォルトのMLパラメータ空間を取得"""
        if model_type == "lightgbm":
            return {
                "num_leaves": {"type": "integer", "low": 10, "high": 100},
                "learning_rate": {"type": "real", "low": 0.01, "high": 0.3},
                "feature_fraction": {"type": "real", "low": 0.5, "high": 1.0},
                "bagging_fraction": {"type": "real", "low": 0.5, "high": 1.0},
                "min_data_in_leaf": {"type": "integer", "low": 5, "high": 50},
            }
        else:
            # デフォルト空間
            return {
                "param1": {"type": "real", "low": 0.1, "high": 1.0},
                "param2": {"type": "integer", "low": 1, "high": 10},
            }

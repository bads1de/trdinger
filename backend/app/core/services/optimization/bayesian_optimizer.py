"""
ベイズ最適化エンジン

GAパラメータとMLハイパーパラメータの自動調整を行います。
"""

import logging
import numpy as np

from typing import Dict, Any, Callable, Optional
from datetime import datetime

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args, OptimizeResult

from .base_optimizer import BaseOptimizer, OptimizationResult, ParameterSpace

logger = logging.getLogger(__name__)


class BayesianOptimizer(BaseOptimizer):
    """
    ベイズ最適化エンジン

    GAパラメータやMLハイパーパラメータの効率的な最適化を行います。
    """

    def __init__(self):
        """初期化"""
        super().__init__()

        # ベイジアン最適化固有の設定
        self.config = {
            "n_initial_points": 10,  # 初期ランダム試行回数
            "acq_func": "EI",  # 獲得関数（Expected Improvement）
            "random_state": 42,  # 乱数シード
            "n_jobs": 1,  # 並列実行数
        }

    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        parameter_space: Dict[str, ParameterSpace],
        n_calls: int = 50,
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        ベイジアン最適化を実行

        Args:
            objective_function: 目的関数
            parameter_space: パラメータ空間
            n_calls: 最適化試行回数
            **kwargs: 追加のオプション

        Returns:
            最適化結果
        """
        try:
            # パラメータ空間と目的関数の妥当性を検証
            self.validate_parameter_space(parameter_space)
            self.validate_objective_function(objective_function)

            method_name = self.get_method_name()
            self._log_optimization_start(method_name, n_calls)
            start_time = datetime.now()

            result = self._optimize_with_skopt(
                objective_function, parameter_space, n_calls
            )

            end_time = datetime.now()
            optimization_time = (end_time - start_time).total_seconds()

            optimization_result = self._create_optimization_result(
                best_params=result["best_params"],
                best_score=result["best_score"],
                history=result["history"],
                optimization_time=optimization_time,
                convergence_info=result.get("convergence_info", {}),
            )

            self._log_optimization_end(
                method_name, result["best_score"], optimization_time
            )
            return optimization_result

        except Exception as e:
            logger.error(f"ベイジアン最適化中にエラーが発生しました: {e}")
            raise

    def _optimize_with_skopt(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        parameter_space: Dict[str, ParameterSpace],
        n_calls: int,
    ) -> Dict[str, Any]:
        """scikit-optimizeを使用した最適化"""
        try:
            # パラメータ空間を定義
            dimensions = []
            param_names = []

            for param_name, param_config in parameter_space.items():
                param_names.append(param_name)

                if param_config.type == "real":
                    dimensions.append(
                        Real(param_config.low, param_config.high, name=param_name)
                    )
                elif param_config.type == "integer":
                    dimensions.append(
                        Integer(param_config.low, param_config.high, name=param_name)
                    )
                elif param_config.type == "categorical":
                    dimensions.append(
                        Categorical(param_config.categories, name=param_name)
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

    def get_default_parameter_space(self, model_type: str) -> Dict[str, ParameterSpace]:
        """デフォルトのMLパラメータ空間を取得"""
        if model_type.lower() == "lightgbm":
            return {
                "num_leaves": ParameterSpace(type="integer", low=10, high=100),
                "learning_rate": ParameterSpace(type="real", low=0.01, high=0.3),
                "feature_fraction": ParameterSpace(type="real", low=0.5, high=1.0),
                "bagging_fraction": ParameterSpace(type="real", low=0.5, high=1.0),
                "min_data_in_leaf": ParameterSpace(type="integer", low=5, high=50),
            }
        else:
            # デフォルト空間
            return {
                "n_estimators": ParameterSpace(type="integer", low=50, high=500),
                "learning_rate": ParameterSpace(type="real", low=0.01, high=0.2),
                "max_depth": ParameterSpace(type="integer", low=3, high=15),
            }

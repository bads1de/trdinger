"""
ランダムサーチ最適化エンジン

パラメータ空間をランダムに探索してMLハイパーパラメータの最適化を行います。
"""

import logging
import numpy as np
from typing import Dict, Any, Callable
from datetime import datetime

from .base_optimizer import BaseOptimizer, OptimizationResult, ParameterSpace

logger = logging.getLogger(__name__)


class RandomSearchOptimizer(BaseOptimizer):
    """
    ランダムサーチ最適化エンジン

    パラメータ空間をランダムに探索してMLハイパーパラメータの最適化を行います。
    """

    def __init__(self):
        """初期化"""
        super().__init__()

        # ランダムサーチ固有の設定
        self.config = {
            "random_state": 42,  # 乱数シード
            "early_stopping_patience": 20,  # 改善が見られない場合の早期停止
            "improvement_threshold": 1e-6,  # 改善とみなす最小閾値
        }

    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        parameter_space: Dict[str, ParameterSpace],
        n_calls: int = 50,
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        ランダムサーチ最適化を実行

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

            # ランダムサーチを実行
            result = self._execute_random_search(
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
            logger.error(f"ランダムサーチ最適化中にエラーが発生しました: {e}")
            raise

    def _execute_random_search(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        parameter_space: Dict[str, ParameterSpace],
        n_calls: int,
    ) -> Dict[str, Any]:
        """
        ランダムサーチを実行

        Args:
            objective_function: 目的関数
            parameter_space: パラメータ空間
            n_calls: 試行回数

        Returns:
            最適化結果
        """
        try:
            # 乱数シードを設定
            np.random.seed(self.config["random_state"])

            best_score = float("-inf")
            best_params = None
            history = []
            no_improvement_count = 0

            for i in range(n_calls):
                try:
                    # ランダムパラメータを生成
                    params = self._sample_random_parameters(parameter_space)

                    # 目的関数を評価
                    score = objective_function(params)

                    # 履歴に記録
                    history.append(
                        {"iteration": i + 1, "params": params.copy(), "score": score}
                    )

                    # ベストスコアを更新
                    if score > best_score + self.config["improvement_threshold"]:
                        best_score = score
                        best_params = params.copy()
                        no_improvement_count = 0
                        logger.debug(
                            f"新しいベストスコア: {best_score:.4f} (iteration {i + 1})"
                        )
                    else:
                        no_improvement_count += 1

                    # 早期停止チェック
                    if no_improvement_count >= self.config["early_stopping_patience"]:
                        logger.info(
                            f"早期停止: {self.config['early_stopping_patience']}回連続で改善なし "
                            f"(iteration {i + 1})"
                        )
                        break

                    # 進捗ログ
                    if (i + 1) % max(1, n_calls // 10) == 0:
                        logger.info(
                            f"ランダムサーチ進捗: {i + 1}/{n_calls} "
                            f"(現在のベストスコア: {best_score:.4f})"
                        )

                except Exception as e:
                    logger.warning(f"パラメータ評価エラー (iteration {i + 1}): {e}")
                    # エラーの場合は大きなペナルティスコアを記録
                    history.append(
                        {
                            "iteration": i + 1,
                            "params": params.copy() if "params" in locals() else {},
                            "score": float("-inf"),
                        }
                    )

            if best_params is None:
                raise RuntimeError("有効なパラメータが見つかりませんでした")

            return {
                "best_params": best_params,
                "best_score": best_score,
                "history": history,
                "convergence_info": {
                    "converged": no_improvement_count
                    < self.config["early_stopping_patience"],
                    "total_evaluations": len(history),
                    "best_iteration": max(
                        range(len(history)), key=lambda i: history[i]["score"]
                    )
                    + 1,
                    "early_stopped": no_improvement_count
                    >= self.config["early_stopping_patience"],
                },
            }

        except Exception as e:
            logger.error(f"ランダムサーチ実行中にエラーが発生しました: {e}")
            raise

    def _sample_random_parameters(
        self, parameter_space: Dict[str, ParameterSpace]
    ) -> Dict[str, Any]:
        """
        パラメータ空間からランダムにパラメータをサンプリング

        Args:
            parameter_space: パラメータ空間

        Returns:
            ランダムパラメータ
        """
        try:
            params = {}

            for param_name, param_config in parameter_space.items():
                if param_config.type == "real":
                    # 実数パラメータは一様分布からサンプリング
                    value = np.random.uniform(param_config.low, param_config.high)
                    params[param_name] = float(value)

                elif param_config.type == "integer":
                    # 整数パラメータは一様分布からサンプリング
                    value = np.random.randint(param_config.low, param_config.high + 1)
                    params[param_name] = int(value)

                elif param_config.type == "categorical":
                    # カテゴリカルパラメータはランダム選択
                    value = np.random.choice(param_config.categories)
                    params[param_name] = value
                else:
                    raise ValueError(f"未対応のパラメータ型: {param_config.type}")

            return params

        except Exception as e:
            logger.error(f"ランダムパラメータサンプリング中にエラーが発生しました: {e}")
            raise

    def get_default_parameter_space(self, model_type: str) -> Dict[str, ParameterSpace]:
        """デフォルトのMLパラメータ空間を取得（ランダムサーチ用に調整）"""
        if model_type.lower() == "lightgbm":
            return {
                "num_leaves": ParameterSpace(type="integer", low=10, high=100),
                "learning_rate": ParameterSpace(type="real", low=0.01, high=0.3),
                "feature_fraction": ParameterSpace(type="real", low=0.5, high=1.0),
                "bagging_fraction": ParameterSpace(type="real", low=0.5, high=1.0),
                "min_data_in_leaf": ParameterSpace(type="integer", low=5, high=50),
                "max_depth": ParameterSpace(type="integer", low=3, high=15),
            }
        else:
            # デフォルト空間（ランダムサーチに適した範囲）
            return {
                "n_estimators": ParameterSpace(type="integer", low=50, high=500),
                "learning_rate": ParameterSpace(type="real", low=0.01, high=0.2),
                "max_depth": ParameterSpace(type="integer", low=3, high=15),
                "min_samples_split": ParameterSpace(type="integer", low=2, high=20),
                "min_samples_leaf": ParameterSpace(type="integer", low=1, high=10),
            }

    def set_random_state(self, random_state: int) -> None:
        """ランダムシードを設定"""
        self.config["random_state"] = random_state

    def set_early_stopping_patience(self, patience: int) -> None:
        """早期停止の忍耐回数を設定"""
        if patience <= 0:
            raise ValueError("patience は正の整数である必要があります")
        self.config["early_stopping_patience"] = patience

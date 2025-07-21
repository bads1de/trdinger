"""
グリッドサーチ最適化エンジン

パラメータ空間を網羅的に探索してMLハイパーパラメータの最適化を行います。
"""

import logging
import numpy as np
from typing import Dict, Any, List, Callable
from datetime import datetime
from itertools import product

from .base_optimizer import BaseOptimizer, OptimizationResult, ParameterSpace

logger = logging.getLogger(__name__)


class GridSearchOptimizer(BaseOptimizer):
    """
    グリッドサーチ最適化エンジン

    パラメータ空間を網羅的に探索してMLハイパーパラメータの最適化を行います。
    """

    def __init__(self):
        """初期化"""
        super().__init__()

        # グリッドサーチ固有の設定
        self.config = {
            "max_combinations": 1000,  # 最大組み合わせ数（計算時間制限）
            "random_state": 42,  # 乱数シード（組み合わせが多い場合のサンプリング用）
        }

    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        parameter_space: Dict[str, ParameterSpace],
        n_calls: int = 50,
        **kwargs: Any
    ) -> OptimizationResult:
        """
        グリッドサーチ最適化を実行

        Args:
            objective_function: 目的関数
            parameter_space: パラメータ空間
            n_calls: 最大評価回数（グリッドサーチでは組み合わせ数の上限として使用）
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

            # パラメータグリッドを生成
            param_grid = self._generate_parameter_grid(parameter_space, n_calls)
            
            # グリッドサーチを実行
            result = self._execute_grid_search(objective_function, param_grid)

            end_time = datetime.now()
            optimization_time = (end_time - start_time).total_seconds()

            optimization_result = self._create_optimization_result(
                best_params=result["best_params"],
                best_score=result["best_score"],
                history=result["history"],
                optimization_time=optimization_time,
                convergence_info=result.get("convergence_info", {})
            )

            self._log_optimization_end(method_name, result["best_score"], optimization_time)
            return optimization_result

        except Exception as e:
            logger.error(f"グリッドサーチ最適化中にエラーが発生しました: {e}")
            raise

    def _generate_parameter_grid(
        self, 
        parameter_space: Dict[str, ParameterSpace], 
        max_combinations: int
    ) -> List[Dict[str, Any]]:
        """
        パラメータグリッドを生成

        Args:
            parameter_space: パラメータ空間
            max_combinations: 最大組み合わせ数

        Returns:
            パラメータ組み合わせのリスト
        """
        try:
            # 各パラメータの値リストを生成
            param_values = {}
            
            for param_name, param_config in parameter_space.items():
                if param_config.type == "real":
                    # 実数パラメータは等間隔で分割
                    n_points = min(10, int(max_combinations ** (1/len(parameter_space))))
                    values = np.linspace(param_config.low, param_config.high, n_points)
                    param_values[param_name] = values.tolist()
                    
                elif param_config.type == "integer":
                    # 整数パラメータは範囲内の全ての値または等間隔サンプリング
                    range_size = param_config.high - param_config.low + 1
                    n_points = min(range_size, int(max_combinations ** (1/len(parameter_space))))
                    
                    if range_size <= n_points:
                        values = list(range(param_config.low, param_config.high + 1))
                    else:
                        values = np.linspace(param_config.low, param_config.high, n_points, dtype=int)
                        values = sorted(list(set(values)))  # 重複を除去
                    param_values[param_name] = values
                    
                elif param_config.type == "categorical":
                    # カテゴリカルパラメータは全ての値を使用
                    param_values[param_name] = param_config.categories
                else:
                    raise ValueError(f"未対応のパラメータ型: {param_config.type}")

            # 全組み合わせを生成
            param_names = list(param_values.keys())
            param_combinations = list(product(*[param_values[name] for name in param_names]))
            
            # 組み合わせ数が多すぎる場合はランダムサンプリング
            if len(param_combinations) > max_combinations:
                logger.warning(
                    f"パラメータ組み合わせ数が多すぎます ({len(param_combinations)} > {max_combinations})。"
                    f"ランダムサンプリングで{max_combinations}個に制限します。"
                )
                np.random.seed(self.config["random_state"])
                indices = np.random.choice(
                    len(param_combinations), 
                    size=max_combinations, 
                    replace=False
                )
                param_combinations = [param_combinations[i] for i in indices]

            # 辞書形式に変換
            param_grid = []
            for combination in param_combinations:
                param_dict = dict(zip(param_names, combination))
                param_grid.append(param_dict)

            logger.info(f"グリッドサーチ: {len(param_grid)}個のパラメータ組み合わせを生成")
            return param_grid

        except Exception as e:
            logger.error(f"パラメータグリッド生成中にエラーが発生しました: {e}")
            raise

    def _execute_grid_search(
        self, 
        objective_function: Callable[[Dict[str, Any]], float], 
        param_grid: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        グリッドサーチを実行

        Args:
            objective_function: 目的関数
            param_grid: パラメータグリッド

        Returns:
            最適化結果
        """
        try:
            best_score = float('-inf')
            best_params = None
            history = []

            for i, params in enumerate(param_grid):
                try:
                    # 目的関数を評価
                    score = objective_function(params)
                    
                    # 履歴に記録
                    history.append({
                        "iteration": i + 1,
                        "params": params.copy(),
                        "score": score
                    })
                    
                    # ベストスコアを更新
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        
                    # 進捗ログ
                    if (i + 1) % max(1, len(param_grid) // 10) == 0:
                        logger.info(
                            f"グリッドサーチ進捗: {i + 1}/{len(param_grid)} "
                            f"(現在のベストスコア: {best_score:.4f})"
                        )
                        
                except Exception as e:
                    logger.warning(f"パラメータ評価エラー (iteration {i + 1}): {e}")
                    # エラーの場合は大きなペナルティスコアを記録
                    history.append({
                        "iteration": i + 1,
                        "params": params.copy(),
                        "score": float('-inf')
                    })

            if best_params is None:
                raise RuntimeError("有効なパラメータ組み合わせが見つかりませんでした")

            return {
                "best_params": best_params,
                "best_score": best_score,
                "history": history,
                "convergence_info": {
                    "converged": True,  # グリッドサーチは常に収束
                    "total_combinations": len(param_grid),
                    "best_iteration": max(range(len(history)), key=lambda i: history[i]["score"]) + 1
                }
            }

        except Exception as e:
            logger.error(f"グリッドサーチ実行中にエラーが発生しました: {e}")
            raise

    def get_default_parameter_space(self, model_type: str) -> Dict[str, ParameterSpace]:
        """デフォルトのMLパラメータ空間を取得（グリッドサーチ用に調整）"""
        if model_type.lower() == "lightgbm":
            return {
                "num_leaves": ParameterSpace(type="integer", low=20, high=80),
                "learning_rate": ParameterSpace(type="real", low=0.05, high=0.2),
                "feature_fraction": ParameterSpace(type="real", low=0.7, high=1.0),
                "min_data_in_leaf": ParameterSpace(type="integer", low=10, high=30),
            }
        else:
            # デフォルト空間（グリッドサーチに適した範囲）
            return {
                "n_estimators": ParameterSpace(type="integer", low=100, high=300),
                "learning_rate": ParameterSpace(type="real", low=0.05, high=0.15),
                "max_depth": ParameterSpace(type="integer", low=5, high=10),
            }

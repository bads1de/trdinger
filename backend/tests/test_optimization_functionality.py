"""
最適化機能の動作確認テスト

各最適化手法が正常に動作し、期待される結果を返すことを確認します。
"""

import pytest
import numpy as np
from typing import Dict, Any

from app.core.services.optimization.base_optimizer import ParameterSpace, OptimizationResult
from app.core.services.optimization.bayesian_optimizer import BayesianOptimizer
from app.core.services.optimization.grid_search_optimizer import GridSearchOptimizer
from app.core.services.optimization.random_search_optimizer import RandomSearchOptimizer
from app.core.services.optimization.optimizer_factory import OptimizerFactory


class TestOptimizationFunctionality:
    """最適化機能の動作確認テスト"""

    def test_bayesian_optimizer_functionality(self):
        """ベイジアンオプティマイザーの動作確認"""
        optimizer = BayesianOptimizer()
        
        # 簡単な二次関数を最適化（x=0.5で最大値1.0）
        def objective_function(params: Dict[str, Any]) -> float:
            x = params["x"]
            return 1.0 - 4 * (x - 0.5) ** 2
        
        parameter_space = {
            "x": ParameterSpace(type="real", low=0.0, high=1.0)
        }
        
        result = optimizer.optimize(
            objective_function=objective_function,
            parameter_space=parameter_space,
            n_calls=20
        )
        
        # 結果の検証
        assert isinstance(result, OptimizationResult)
        assert "x" in result.best_params
        assert 0.0 <= result.best_params["x"] <= 1.0
        assert result.best_score > 0.8  # 最適値に近い値が得られることを期待
        assert result.total_evaluations == 20
        assert result.optimization_time > 0
        assert len(result.optimization_history) == 20
        
        # 最適解に近いことを確認（誤差±0.2以内）
        assert abs(result.best_params["x"] - 0.5) < 0.2

    def test_grid_search_optimizer_functionality(self):
        """グリッドサーチオプティマイザーの動作確認"""
        optimizer = GridSearchOptimizer()
        
        # 離散的な最適化問題（x=1, y=1で最大値0）
        def objective_function(params: Dict[str, Any]) -> float:
            x, y = params["x"], params["y"]
            return -(x - 1) ** 2 - (y - 1) ** 2
        
        parameter_space = {
            "x": ParameterSpace(type="integer", low=0, high=2),
            "y": ParameterSpace(type="integer", low=0, high=2)
        }
        
        result = optimizer.optimize(
            objective_function=objective_function,
            parameter_space=parameter_space,
            n_calls=20
        )
        
        # 結果の検証
        assert isinstance(result, OptimizationResult)
        assert result.best_params["x"] == 1
        assert result.best_params["y"] == 1
        assert result.best_score == 0.0  # 最適値
        assert result.total_evaluations <= 9  # 3x3グリッド
        assert result.optimization_time > 0
        assert result.convergence_info["converged"] == True

    def test_random_search_optimizer_functionality(self):
        """ランダムサーチオプティマイザーの動作確認"""
        optimizer = RandomSearchOptimizer()
        
        # 単峰性関数を最適化
        def objective_function(params: Dict[str, Any]) -> float:
            x = params["x"]
            return 1.0 - (x - 0.7) ** 2
        
        parameter_space = {
            "x": ParameterSpace(type="real", low=0.0, high=1.0)
        }
        
        result = optimizer.optimize(
            objective_function=objective_function,
            parameter_space=parameter_space,
            n_calls=30
        )
        
        # 結果の検証
        assert isinstance(result, OptimizationResult)
        assert "x" in result.best_params
        assert 0.0 <= result.best_params["x"] <= 1.0
        assert result.best_score > 0.7  # 合理的なスコア
        assert result.total_evaluations <= 30
        assert result.optimization_time > 0
        
        # 最適解に近いことを確認（誤差±0.3以内）
        assert abs(result.best_params["x"] - 0.7) < 0.3

    def test_categorical_parameter_optimization(self):
        """カテゴリカルパラメータの最適化テスト"""
        optimizer = RandomSearchOptimizer()
        
        # カテゴリカルパラメータを含む最適化
        def objective_function(params: Dict[str, Any]) -> float:
            algorithm = params["algorithm"]
            n_estimators = params["n_estimators"]
            
            # "best"アルゴリズムで高いスコア
            if algorithm == "best":
                return 0.9 + 0.1 * (n_estimators / 100)
            else:
                return 0.5 + 0.1 * (n_estimators / 100)
        
        parameter_space = {
            "algorithm": ParameterSpace(type="categorical", categories=["good", "best", "worst"]),
            "n_estimators": ParameterSpace(type="integer", low=50, high=100)
        }
        
        result = optimizer.optimize(
            objective_function=objective_function,
            parameter_space=parameter_space,
            n_calls=20
        )
        
        # 結果の検証
        assert isinstance(result, OptimizationResult)
        assert result.best_params["algorithm"] in ["good", "best", "worst"]
        assert 50 <= result.best_params["n_estimators"] <= 100
        assert result.best_score > 0.5

    def test_multi_parameter_optimization(self):
        """複数パラメータの最適化テスト"""
        optimizer = BayesianOptimizer()
        
        # 複数パラメータの最適化問題
        def objective_function(params: Dict[str, Any]) -> float:
            x = params["x"]
            y = params["y"]
            z = params["z"]
            
            # 複雑な関数（x=0.3, y=0.7, z=50で最大値）
            return (
                1.0 - (x - 0.3) ** 2 - (y - 0.7) ** 2 - ((z - 50) / 50) ** 2
            )
        
        parameter_space = {
            "x": ParameterSpace(type="real", low=0.0, high=1.0),
            "y": ParameterSpace(type="real", low=0.0, high=1.0),
            "z": ParameterSpace(type="integer", low=10, high=100)
        }
        
        result = optimizer.optimize(
            objective_function=objective_function,
            parameter_space=parameter_space,
            n_calls=25
        )
        
        # 結果の検証
        assert isinstance(result, OptimizationResult)
        assert 0.0 <= result.best_params["x"] <= 1.0
        assert 0.0 <= result.best_params["y"] <= 1.0
        assert 10 <= result.best_params["z"] <= 100
        assert result.best_score > 0.3  # 合理的なスコア
        assert result.total_evaluations == 25

    def test_optimizer_factory_functionality(self):
        """OptimizerFactoryの動作確認"""
        # 各手法のオプティマイザーを作成
        bayesian_opt = OptimizerFactory.create_optimizer("bayesian")
        grid_opt = OptimizerFactory.create_optimizer("grid")
        random_opt = OptimizerFactory.create_optimizer("random")
        
        assert isinstance(bayesian_opt, BayesianOptimizer)
        assert isinstance(grid_opt, GridSearchOptimizer)
        assert isinstance(random_opt, RandomSearchOptimizer)
        
        # 別名での作成
        bayes_opt = OptimizerFactory.create_optimizer("bayes")
        grid_search_opt = OptimizerFactory.create_optimizer("grid_search")
        random_search_opt = OptimizerFactory.create_optimizer("random_search")
        
        assert isinstance(bayes_opt, BayesianOptimizer)
        assert isinstance(grid_search_opt, GridSearchOptimizer)
        assert isinstance(random_search_opt, RandomSearchOptimizer)
        
        # サポートされている手法の確認
        methods = OptimizerFactory.get_supported_methods()
        assert "bayesian" in methods
        assert "grid" in methods
        assert "random" in methods
        
        # 手法サポートの確認
        assert OptimizerFactory.is_supported_method("bayesian")
        assert OptimizerFactory.is_supported_method("bayes")
        assert not OptimizerFactory.is_supported_method("invalid")

    def test_optimization_convergence(self):
        """最適化の収束性テスト"""
        optimizer = BayesianOptimizer()
        
        # 明確な最適解がある関数
        def objective_function(params: Dict[str, Any]) -> float:
            x = params["x"]
            y = params["y"]
            # (0.6, 0.4)で最大値1.0
            return 1.0 - ((x - 0.6) ** 2 + (y - 0.4) ** 2)
        
        parameter_space = {
            "x": ParameterSpace(type="real", low=0.0, high=1.0),
            "y": ParameterSpace(type="real", low=0.0, high=1.0)
        }
        
        result = optimizer.optimize(
            objective_function=objective_function,
            parameter_space=parameter_space,
            n_calls=30
        )
        
        # 収束性の確認
        assert result.best_score > 0.8  # 高いスコアが得られる
        
        # 履歴の確認（スコアが改善されていることを確認）
        scores = [entry["score"] for entry in result.optimization_history]
        max_score_so_far = []
        current_max = float('-inf')
        
        for score in scores:
            if score > current_max:
                current_max = score
            max_score_so_far.append(current_max)
        
        # 最終的なスコアが初期スコアより改善されていることを確認
        assert max_score_so_far[-1] >= max_score_so_far[0]

    def test_optimization_error_handling(self):
        """最適化のエラーハンドリングテスト"""
        optimizer = RandomSearchOptimizer()
        
        # エラーを発生させる目的関数
        def error_objective_function(params: Dict[str, Any]) -> float:
            x = params["x"]
            if x < 0.3:
                raise ValueError("Invalid parameter value")
            return x
        
        parameter_space = {
            "x": ParameterSpace(type="real", low=0.0, high=1.0)
        }
        
        # エラーが発生してもクラッシュしないことを確認
        result = optimizer.optimize(
            objective_function=error_objective_function,
            parameter_space=parameter_space,
            n_calls=10
        )
        
        # 結果が返されることを確認
        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert result.best_score > 0  # 有効なスコアが得られる

    def test_default_parameter_spaces(self):
        """デフォルトパラメータ空間のテスト"""
        bayesian_opt = BayesianOptimizer()
        grid_opt = GridSearchOptimizer()
        random_opt = RandomSearchOptimizer()
        
        # LightGBMのデフォルトパラメータ空間
        bayesian_space = bayesian_opt.get_default_parameter_space("lightgbm")
        grid_space = grid_opt.get_default_parameter_space("lightgbm")
        random_space = random_opt.get_default_parameter_space("lightgbm")
        
        # 共通パラメータの確認
        for space in [bayesian_space, grid_space, random_space]:
            assert "num_leaves" in space
            assert "learning_rate" in space
            assert space["num_leaves"].type == "integer"
            assert space["learning_rate"].type == "real"
        
        # その他のモデルのデフォルトパラメータ空間
        other_space = bayesian_opt.get_default_parameter_space("other")
        assert "n_estimators" in other_space
        assert "learning_rate" in other_space
        assert "max_depth" in other_space

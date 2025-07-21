"""
OptunaOptimizerのテスト
"""

import pytest
import numpy as np
from app.core.services.optimization.optuna_optimizer import (
    OptunaOptimizer,
    ParameterSpace,
)


class TestOptunaOptimizer:
    """OptunaOptimizerのテストクラス"""

    def test_basic_optimization(self):
        """基本的な最適化テスト"""
        optimizer = OptunaOptimizer()

        def objective(params):
            # x=0.5で最大値を取る関数
            return -((params["x"] - 0.5) ** 2)

        parameter_space = {"x": ParameterSpace(type="real", low=0.0, high=1.0)}

        result = optimizer.optimize(objective, parameter_space, n_calls=20)

        # 結果の検証
        assert abs(result.best_params["x"] - 0.5) < 0.3  # 0.5に近い値
        assert result.best_score > -0.2  # 良いスコア
        assert result.total_evaluations <= 20
        assert result.optimization_time > 0
        assert result.study is not None

    def test_integer_parameter_optimization(self):
        """整数パラメータの最適化テスト"""
        optimizer = OptunaOptimizer()

        def objective(params):
            # n=10で最大値を取る関数
            return -((params["n"] - 10) ** 2)

        parameter_space = {"n": ParameterSpace(type="integer", low=1, high=20)}

        result = optimizer.optimize(objective, parameter_space, n_calls=15)

        # 結果の検証
        assert isinstance(result.best_params["n"], int)
        assert abs(result.best_params["n"] - 10) <= 3  # 10に近い値
        assert result.best_score > -20

    def test_categorical_parameter_optimization(self):
        """カテゴリカルパラメータの最適化テスト"""
        optimizer = OptunaOptimizer()

        def objective(params):
            # "best"で最大値を取る関数
            score_map = {"bad": 0.1, "good": 0.5, "best": 1.0}
            return score_map.get(params["choice"], 0.0)

        parameter_space = {
            "choice": ParameterSpace(
                type="categorical", categories=["bad", "good", "best"]
            )
        }

        result = optimizer.optimize(objective, parameter_space, n_calls=10)

        # 結果の検証
        assert result.best_params["choice"] in ["bad", "good", "best"]
        assert result.best_score >= 0.5  # "good"以上のスコア

    def test_multi_parameter_optimization(self):
        """複数パラメータの最適化テスト"""
        optimizer = OptunaOptimizer()

        def objective(params):
            # x=0.3, y=0.7で最大値を取る関数
            return -((params["x"] - 0.3) ** 2) - (params["y"] - 0.7) ** 2

        parameter_space = {
            "x": ParameterSpace(type="real", low=0.0, high=1.0),
            "y": ParameterSpace(type="real", low=0.0, high=1.0),
        }

        result = optimizer.optimize(objective, parameter_space, n_calls=30)

        # 結果の検証
        assert abs(result.best_params["x"] - 0.3) < 0.3
        assert abs(result.best_params["y"] - 0.7) < 0.3
        assert result.best_score > -0.5

    def test_default_parameter_space(self):
        """デフォルトパラメータ空間のテスト"""
        space = OptunaOptimizer.get_default_parameter_space()

        # 期待されるパラメータが存在することを確認
        expected_params = [
            "num_leaves",
            "learning_rate",
            "feature_fraction",
            "bagging_fraction",
            "min_data_in_leaf",
            "max_depth",
        ]

        for param in expected_params:
            assert param in space
            assert isinstance(space[param], ParameterSpace)

        # パラメータの型と範囲を確認
        assert space["num_leaves"].type == "integer"
        assert space["num_leaves"].low == 10
        assert space["num_leaves"].high == 100

        assert space["learning_rate"].type == "real"
        assert space["learning_rate"].low == 0.01
        assert space["learning_rate"].high == 0.3

    def test_optimization_with_exception_handling(self):
        """例外処理のテスト"""
        optimizer = OptunaOptimizer()

        def objective(params):
            # 一部のパラメータで例外を発生させる
            if params["x"] < 0.2:
                raise ValueError("Invalid parameter")
            return params["x"]

        parameter_space = {"x": ParameterSpace(type="real", low=0.0, high=1.0)}

        result = optimizer.optimize(objective, parameter_space, n_calls=20)

        # 例外が発生しても最適化が完了することを確認
        assert result.best_params["x"] >= 0.2  # 例外が発生しない範囲
        assert result.total_evaluations <= 20

    def test_method_name(self):
        """メソッド名の取得テスト"""
        optimizer = OptunaOptimizer()
        assert optimizer.get_method_name() == "Optuna"


class TestParameterSpace:
    """ParameterSpaceのテストクラス"""

    def test_real_parameter_space(self):
        """実数パラメータ空間のテスト"""
        space = ParameterSpace(type="real", low=0.1, high=0.9)

        assert space.type == "real"
        assert space.low == 0.1
        assert space.high == 0.9
        assert space.categories is None

    def test_integer_parameter_space(self):
        """整数パラメータ空間のテスト"""
        space = ParameterSpace(type="integer", low=1, high=100)

        assert space.type == "integer"
        assert space.low == 1
        assert space.high == 100
        assert space.categories is None

    def test_categorical_parameter_space(self):
        """カテゴリカルパラメータ空間のテスト"""
        categories = ["option1", "option2", "option3"]
        space = ParameterSpace(type="categorical", categories=categories)

        assert space.type == "categorical"
        assert space.categories == categories
        assert space.low is None
        assert space.high is None


if __name__ == "__main__":
    pytest.main([__file__])

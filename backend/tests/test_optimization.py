"""
オプティマイザーのユニットテスト

BaseOptimizer、BayesianOptimizer、GridSearchOptimizer、RandomSearchOptimizer、
OptimizerFactoryのテストを行います。
"""

import pytest
import numpy as np
from typing import Dict, Any

from app.core.services.optimization.base_optimizer import (
    BaseOptimizer,
    ParameterSpace,
    OptimizationResult,
)
from app.core.services.optimization.bayesian_optimizer import BayesianOptimizer
from app.core.services.optimization.grid_search_optimizer import GridSearchOptimizer
from app.core.services.optimization.random_search_optimizer import RandomSearchOptimizer
from app.core.services.optimization.optimizer_factory import OptimizerFactory


class TestParameterSpace:
    """ParameterSpaceクラスのテスト"""

    def test_real_parameter_space(self):
        """実数パラメータ空間のテスト"""
        param = ParameterSpace(type="real", low=0.1, high=1.0)
        assert param.type == "real"
        assert param.low == 0.1
        assert param.high == 1.0

    def test_integer_parameter_space(self):
        """整数パラメータ空間のテスト"""
        param = ParameterSpace(type="integer", low=1, high=10)
        assert param.type == "integer"
        assert param.low == 1
        assert param.high == 10

    def test_categorical_parameter_space(self):
        """カテゴリカルパラメータ空間のテスト"""
        categories = ["a", "b", "c"]
        param = ParameterSpace(type="categorical", categories=categories)
        assert param.type == "categorical"
        assert param.categories == categories

    def test_invalid_real_parameter_space(self):
        """無効な実数パラメータ空間のテスト"""
        with pytest.raises(ValueError):
            ParameterSpace(type="real", low=None, high=1.0)

    def test_invalid_categorical_parameter_space(self):
        """無効なカテゴリカルパラメータ空間のテスト"""
        with pytest.raises(ValueError):
            ParameterSpace(type="categorical", categories=None)


class TestBaseOptimizer:
    """BaseOptimizerクラスのテスト"""

    def test_validate_parameter_space_valid(self):
        """有効なパラメータ空間の検証テスト"""
        optimizer = (
            BayesianOptimizer()
        )  # BaseOptimizerは抽象クラスなので具象クラスを使用

        parameter_space = {
            "param1": ParameterSpace(type="real", low=0.1, high=1.0),
            "param2": ParameterSpace(type="integer", low=1, high=10),
            "param3": ParameterSpace(type="categorical", categories=["a", "b", "c"]),
        }

        # 例外が発生しないことを確認
        optimizer.validate_parameter_space(parameter_space)

    def test_validate_parameter_space_empty(self):
        """空のパラメータ空間の検証テスト"""
        optimizer = BayesianOptimizer()

        with pytest.raises(ValueError, match="パラメータ空間が空です"):
            optimizer.validate_parameter_space({})

    def test_validate_parameter_space_invalid_type(self):
        """無効な型のパラメータ空間の検証テスト"""
        optimizer = BayesianOptimizer()

        parameter_space = {
            "param1": ParameterSpace(type="invalid_type", low=0.1, high=1.0)
        }

        with pytest.raises(ValueError, match="未対応の型"):
            optimizer.validate_parameter_space(parameter_space)

    def test_validate_objective_function_valid(self):
        """有効な目的関数の検証テスト"""
        optimizer = BayesianOptimizer()

        def objective_function(params: Dict[str, Any]) -> float:
            return 0.5

        # 例外が発生しないことを確認
        optimizer.validate_objective_function(objective_function)

    def test_validate_objective_function_invalid(self):
        """無効な目的関数の検証テスト"""
        optimizer = BayesianOptimizer()

        with pytest.raises(
            ValueError, match="目的関数は呼び出し可能である必要があります"
        ):
            optimizer.validate_objective_function("not_callable")


class TestBayesianOptimizer:
    """BayesianOptimizerクラスのテスト"""

    def test_initialization(self):
        """初期化のテスト"""
        optimizer = BayesianOptimizer()
        assert isinstance(optimizer, BaseOptimizer)
        assert "n_initial_points" in optimizer.config
        assert "acq_func" in optimizer.config

    def test_optimize_simple(self):
        """シンプルな最適化のテスト"""
        optimizer = BayesianOptimizer()

        def objective_function(params: Dict[str, Any]) -> float:
            # 簡単な二次関数（x=0.5で最大値1.0）
            x = params["x"]
            return 1.0 - 4 * (x - 0.5) ** 2

        parameter_space = {"x": ParameterSpace(type="real", low=0.0, high=1.0)}

        result = optimizer.optimize(objective_function, parameter_space, n_calls=10)

        assert isinstance(result, OptimizationResult)
        assert "x" in result.best_params
        assert 0.0 <= result.best_params["x"] <= 1.0
        assert result.best_score > 0.5  # 最適値に近い値が得られることを期待
        assert len(result.optimization_history) == 10

    def test_get_default_parameter_space(self):
        """デフォルトパラメータ空間の取得テスト"""
        optimizer = BayesianOptimizer()

        # LightGBMの場合
        space = optimizer.get_default_parameter_space("lightgbm")
        assert "num_leaves" in space
        assert "learning_rate" in space

        # その他の場合
        space = optimizer.get_default_parameter_space("other")
        assert "n_estimators" in space
        assert "learning_rate" in space


class TestGridSearchOptimizer:
    """GridSearchOptimizerクラスのテスト"""

    def test_initialization(self):
        """初期化のテスト"""
        optimizer = GridSearchOptimizer()
        assert isinstance(optimizer, BaseOptimizer)
        assert "max_combinations" in optimizer.config

    def test_optimize_simple(self):
        """シンプルな最適化のテスト"""
        optimizer = GridSearchOptimizer()

        def objective_function(params: Dict[str, Any]) -> float:
            # 簡単な関数（x=1, y=1で最大値）
            x, y = params["x"], params["y"]
            return -((x - 1) ** 2) - (y - 1) ** 2

        parameter_space = {
            "x": ParameterSpace(type="integer", low=0, high=2),
            "y": ParameterSpace(type="integer", low=0, high=2),
        }

        result = optimizer.optimize(objective_function, parameter_space, n_calls=20)

        assert isinstance(result, OptimizationResult)
        assert result.best_params["x"] == 1
        assert result.best_params["y"] == 1
        assert result.best_score == 0.0  # 最適値


class TestRandomSearchOptimizer:
    """RandomSearchOptimizerクラスのテスト"""

    def test_initialization(self):
        """初期化のテスト"""
        optimizer = RandomSearchOptimizer()
        assert isinstance(optimizer, BaseOptimizer)
        assert "random_state" in optimizer.config
        assert "early_stopping_patience" in optimizer.config

    def test_optimize_simple(self):
        """シンプルな最適化のテスト"""
        optimizer = RandomSearchOptimizer()

        def objective_function(params: Dict[str, Any]) -> float:
            # 簡単な二次関数
            x = params["x"]
            return 1.0 - (x - 0.5) ** 2

        parameter_space = {"x": ParameterSpace(type="real", low=0.0, high=1.0)}

        result = optimizer.optimize(objective_function, parameter_space, n_calls=20)

        assert isinstance(result, OptimizationResult)
        assert "x" in result.best_params
        assert 0.0 <= result.best_params["x"] <= 1.0
        assert result.best_score > 0.5

    def test_sample_random_parameters(self):
        """ランダムパラメータサンプリングのテスト"""
        optimizer = RandomSearchOptimizer()

        parameter_space = {
            "real_param": ParameterSpace(type="real", low=0.0, high=1.0),
            "int_param": ParameterSpace(type="integer", low=1, high=10),
            "cat_param": ParameterSpace(type="categorical", categories=["a", "b", "c"]),
        }

        params = optimizer._sample_random_parameters(parameter_space)

        assert 0.0 <= params["real_param"] <= 1.0
        assert 1 <= params["int_param"] <= 10
        assert params["cat_param"] in ["a", "b", "c"]


class TestOptimizerFactory:
    """OptimizerFactoryクラスのテスト"""

    def test_create_bayesian_optimizer(self):
        """ベイジアンオプティマイザー作成のテスト"""
        optimizer = OptimizerFactory.create_optimizer("bayesian")
        assert isinstance(optimizer, BayesianOptimizer)

    def test_create_grid_search_optimizer(self):
        """グリッドサーチオプティマイザー作成のテスト"""
        optimizer = OptimizerFactory.create_optimizer("grid")
        assert isinstance(optimizer, GridSearchOptimizer)

    def test_create_random_search_optimizer(self):
        """ランダムサーチオプティマイザー作成のテスト"""
        optimizer = OptimizerFactory.create_optimizer("random")
        assert isinstance(optimizer, RandomSearchOptimizer)

    def test_create_optimizer_with_alias(self):
        """別名を使ったオプティマイザー作成のテスト"""
        optimizer = OptimizerFactory.create_optimizer("bayes")
        assert isinstance(optimizer, BayesianOptimizer)

        optimizer = OptimizerFactory.create_optimizer("grid_search")
        assert isinstance(optimizer, GridSearchOptimizer)

    def test_create_optimizer_invalid_method(self):
        """無効な手法でのオプティマイザー作成のテスト"""
        with pytest.raises(ValueError, match="未対応の最適化手法"):
            OptimizerFactory.create_optimizer("invalid_method")

    def test_get_supported_methods(self):
        """サポートされている手法の取得テスト"""
        methods = OptimizerFactory.get_supported_methods()
        assert "bayesian" in methods
        assert "grid" in methods
        assert "random" in methods

    def test_is_supported_method(self):
        """手法サポート確認のテスト"""
        assert OptimizerFactory.is_supported_method("bayesian")
        assert OptimizerFactory.is_supported_method("bayes")  # 別名
        assert not OptimizerFactory.is_supported_method("invalid")

    def test_get_method_description(self):
        """手法説明の取得テスト"""
        desc = OptimizerFactory.get_method_description("bayesian")
        assert "ベイジアン最適化" in desc


class TestMLTrainingServiceOptimization:
    """MLTrainingServiceの最適化機能のテスト"""

    def test_optimization_settings_creation(self):
        """OptimizationSettingsクラスの作成テスト"""
        from app.core.services.ml.ml_training_service import OptimizationSettings

        # デフォルト設定
        settings = OptimizationSettings()
        assert settings.enabled == False
        assert settings.method == "bayesian"
        assert settings.n_calls == 50
        assert settings.parameter_space == {}

        # カスタム設定
        parameter_space = {
            "n_estimators": {"type": "integer", "low": 50, "high": 500},
            "learning_rate": {"type": "real", "low": 0.01, "high": 0.2},
        }
        settings = OptimizationSettings(
            enabled=True, method="grid", n_calls=30, parameter_space=parameter_space
        )
        assert settings.enabled == True
        assert settings.method == "grid"
        assert settings.n_calls == 30
        assert settings.parameter_space == parameter_space

    def test_prepare_parameter_space(self):
        """パラメータ空間準備のテスト"""
        from app.core.services.ml.ml_training_service import MLTrainingService

        service = MLTrainingService()

        parameter_space_config = {
            "n_estimators": {"type": "integer", "low": 50, "high": 500},
            "learning_rate": {"type": "real", "low": 0.01, "high": 0.2},
            "algorithm": {
                "type": "categorical",
                "categories": ["auto", "ball_tree", "kd_tree"],
            },
        }

        parameter_space = service._prepare_parameter_space(parameter_space_config)

        assert len(parameter_space) == 3
        assert parameter_space["n_estimators"].type == "integer"
        assert parameter_space["n_estimators"].low == 50
        assert parameter_space["n_estimators"].high == 500
        assert parameter_space["learning_rate"].type == "real"
        assert parameter_space["algorithm"].type == "categorical"
        assert parameter_space["algorithm"].categories == [
            "auto",
            "ball_tree",
            "kd_tree",
        ]


class TestMLTrainingAPI:
    """MLTrainingAPIの最適化機能のテスト"""

    def test_parameter_space_config_creation(self):
        """ParameterSpaceConfigクラスの作成テスト"""
        from app.api.ml_training import ParameterSpaceConfig

        # 実数パラメータ
        real_param = ParameterSpaceConfig(type="real", low=0.01, high=0.2)
        assert real_param.type == "real"
        assert real_param.low == 0.01
        assert real_param.high == 0.2
        assert real_param.categories is None

        # 整数パラメータ
        int_param = ParameterSpaceConfig(type="integer", low=50, high=500)
        assert int_param.type == "integer"
        assert int_param.low == 50
        assert int_param.high == 500

        # カテゴリカルパラメータ
        cat_param = ParameterSpaceConfig(
            type="categorical", categories=["auto", "ball_tree", "kd_tree"]
        )
        assert cat_param.type == "categorical"
        assert cat_param.categories == ["auto", "ball_tree", "kd_tree"]
        assert cat_param.low is None
        assert cat_param.high is None

    def test_optimization_settings_config_creation(self):
        """OptimizationSettingsConfigクラスの作成テスト"""
        from app.api.ml_training import OptimizationSettingsConfig, ParameterSpaceConfig

        # デフォルト設定
        settings = OptimizationSettingsConfig()
        assert settings.enabled == False
        assert settings.method == "bayesian"
        assert settings.n_calls == 50
        assert settings.parameter_space == {}

        # カスタム設定
        parameter_space = {
            "n_estimators": ParameterSpaceConfig(type="integer", low=50, high=500),
            "learning_rate": ParameterSpaceConfig(type="real", low=0.01, high=0.2),
        }
        settings = OptimizationSettingsConfig(
            enabled=True, method="grid", n_calls=30, parameter_space=parameter_space
        )
        assert settings.enabled == True
        assert settings.method == "grid"
        assert settings.n_calls == 30
        assert len(settings.parameter_space) == 2
        assert "n_estimators" in settings.parameter_space
        assert "learning_rate" in settings.parameter_space

    def test_ml_training_config_with_optimization(self):
        """最適化設定を含むMLTrainingConfigのテスト"""
        from app.api.ml_training import (
            MLTrainingConfig,
            OptimizationSettingsConfig,
            ParameterSpaceConfig,
        )

        # 最適化設定なし
        config = MLTrainingConfig(
            symbol="BTC/USDT:USDT", start_date="2023-01-01", end_date="2023-12-31"
        )
        assert config.optimization_settings is None

        # 最適化設定あり
        optimization_settings = OptimizationSettingsConfig(
            enabled=True,
            method="bayesian",
            n_calls=25,
            parameter_space={
                "n_estimators": ParameterSpaceConfig(type="integer", low=100, high=300),
                "learning_rate": ParameterSpaceConfig(type="real", low=0.05, high=0.15),
            },
        )

        config = MLTrainingConfig(
            symbol="BTC/USDT:USDT",
            start_date="2023-01-01",
            end_date="2023-12-31",
            optimization_settings=optimization_settings,
        )

        assert config.optimization_settings is not None
        assert config.optimization_settings.enabled == True
        assert config.optimization_settings.method == "bayesian"
        assert config.optimization_settings.n_calls == 25
        assert len(config.optimization_settings.parameter_space) == 2

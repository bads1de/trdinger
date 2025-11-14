"""
OptunaOptimizerのテストモジュール

Optunaベースの最適化エンジンの各機能を包括的にテストします。
"""

import time
from typing import Any, Dict

import numpy as np
import pytest

from app.services.optimization.optuna_optimizer import (
    OptimizationResult,
    OptunaOptimizer,
    ParameterSpace,
)


class TestOptunaOptimizer:
    """OptunaOptimizerクラスのテスト"""

    @pytest.fixture
    def optimizer(self):
        """OptunaOptimizerインスタンス"""
        return OptunaOptimizer()

    @pytest.fixture
    def simple_parameter_space(self):
        """シンプルなパラメータ空間"""
        return {
            "x": ParameterSpace(type="real", low=-10.0, high=10.0),
            "y": ParameterSpace(type="integer", low=0, high=100),
        }

    @pytest.fixture
    def complex_parameter_space(self):
        """複雑なパラメータ空間"""
        return {
            "learning_rate": ParameterSpace(type="real", low=0.001, high=0.1),
            "num_leaves": ParameterSpace(type="integer", low=10, high=100),
            "feature_fraction": ParameterSpace(type="real", low=0.5, high=1.0),
            "regularization": ParameterSpace(
                type="categorical", categories=["l1", "l2", "none"]
            ),
        }

    def test_optimizer_initialization(self, optimizer):
        """正常系: OptunaOptimizerの初期化"""
        assert optimizer is not None
        assert optimizer.study is None

    def test_optimize_simple_objective(self, optimizer, simple_parameter_space):
        """正常系: 単純な目的関数の最適化"""

        def objective(params: Dict[str, Any]) -> float:
            # 最小値は (x=2, y=5) で値は0
            # Optunaは最大化なので、負の値を返す
            x = params["x"]
            y = params["y"]
            return -((x - 2) ** 2 + (y - 5) ** 2)

        result = optimizer.optimize(objective, simple_parameter_space, n_calls=20)

        assert isinstance(result, OptimizationResult)
        assert "x" in result.best_params
        assert "y" in result.best_params
        assert result.total_evaluations == 20
        assert result.optimization_time > 0
        # 最適解に近い値であることを確認（最大化されるので近くなるはず）
        assert abs(result.best_params["x"] - 2.0) < 5.0
        assert abs(result.best_params["y"] - 5) < 20

    def test_optimize_with_multiple_parameters(
        self, optimizer, complex_parameter_space
    ):
        """正常系: 複数パラメータの最適化"""

        def objective(params: Dict[str, Any]) -> float:
            lr = params["learning_rate"]
            nl = params["num_leaves"]
            ff = params["feature_fraction"]
            reg = params["regularization"]

            # 適当なスコア関数
            score = lr * 10 + nl / 100 + ff
            if reg == "l2":
                score += 0.1
            return score

        result = optimizer.optimize(objective, complex_parameter_space, n_calls=15)

        assert "learning_rate" in result.best_params
        assert "num_leaves" in result.best_params
        assert "feature_fraction" in result.best_params
        assert "regularization" in result.best_params
        assert result.total_evaluations == 15

        # パラメータ範囲の検証
        assert 0.001 <= result.best_params["learning_rate"] <= 0.1
        assert 10 <= result.best_params["num_leaves"] <= 100
        assert 0.5 <= result.best_params["feature_fraction"] <= 1.0
        assert result.best_params["regularization"] in ["l1", "l2", "none"]

    def test_optimize_maximization(self, optimizer, simple_parameter_space):
        """正常系: 最大化問題の最適化"""

        def objective(params: Dict[str, Any]) -> float:
            x = params["x"]
            y = params["y"]
            # 最大値は境界付近
            return -(x**2) - (y - 50) ** 2 + 1000

        result = optimizer.optimize(objective, simple_parameter_space, n_calls=20)

        assert result.best_score > 0
        # 最大化されているので高いスコア
        assert result.best_score > 900

    def test_study_created(self, optimizer, simple_parameter_space):
        """正常系: Optunaスタディが作成される"""

        def objective(params: Dict[str, Any]) -> float:
            return params["x"] ** 2

        result = optimizer.optimize(objective, simple_parameter_space, n_calls=5)

        assert optimizer.study is not None
        assert result.study is not None
        assert len(result.study.trials) == 5

    def test_optimization_result_structure(self, optimizer, simple_parameter_space):
        """正常系: 最適化結果の構造確認"""

        def objective(params: Dict[str, Any]) -> float:
            return params["x"] ** 2 + params["y"]

        result = optimizer.optimize(objective, simple_parameter_space, n_calls=10)

        assert hasattr(result, "best_params")
        assert hasattr(result, "best_score")
        assert hasattr(result, "total_evaluations")
        assert hasattr(result, "optimization_time")
        assert hasattr(result, "study")
        assert isinstance(result.best_params, dict)
        assert isinstance(result.best_score, float)
        assert isinstance(result.total_evaluations, int)
        assert isinstance(result.optimization_time, float)


class TestParameterSuggestion:
    """パラメータサジェスト機能のテスト"""

    @pytest.fixture
    def optimizer(self):
        return OptunaOptimizer()

    def test_suggest_real_parameters(self, optimizer):
        """実数パラメータのサジェスト"""
        parameter_space = {
            "learning_rate": ParameterSpace(type="real", low=0.01, high=0.1),
        }

        learning_rates = []

        def objective(params: Dict[str, Any]) -> float:
            learning_rates.append(params["learning_rate"])
            return params["learning_rate"]

        optimizer.optimize(objective, parameter_space, n_calls=10)

        # 全ての学習率が範囲内
        assert all(0.01 <= lr <= 0.1 for lr in learning_rates)
        assert len(learning_rates) == 10

    def test_suggest_integer_parameters(self, optimizer):
        """整数パラメータのサジェスト"""
        parameter_space = {
            "num_leaves": ParameterSpace(type="integer", low=10, high=50),
        }

        num_leaves_list = []

        def objective(params: Dict[str, Any]) -> float:
            num_leaves_list.append(params["num_leaves"])
            return float(params["num_leaves"])

        optimizer.optimize(objective, parameter_space, n_calls=10)

        # 全ての値が整数で範囲内
        assert all(isinstance(nl, int) for nl in num_leaves_list)
        assert all(10 <= nl <= 50 for nl in num_leaves_list)

    def test_suggest_categorical_parameters(self, optimizer):
        """カテゴリカルパラメータのサジェスト"""
        parameter_space = {
            "optimizer": ParameterSpace(
                type="categorical", categories=["adam", "sgd", "rmsprop"]
            ),
        }

        optimizers = []

        def objective(params: Dict[str, Any]) -> float:
            optimizers.append(params["optimizer"])
            return 1.0

        optimizer.optimize(objective, parameter_space, n_calls=10)

        # 全ての値が選択肢に含まれる
        assert all(opt in ["adam", "sgd", "rmsprop"] for opt in optimizers)

    def test_mixed_parameter_types(self, optimizer):
        """混合パラメータタイプのサジェスト"""
        parameter_space = {
            "lr": ParameterSpace(type="real", low=0.001, high=0.1),
            "batch_size": ParameterSpace(type="integer", low=16, high=256),
            "activation": ParameterSpace(
                type="categorical", categories=["relu", "tanh", "sigmoid"]
            ),
        }

        params_list = []

        def objective(params: Dict[str, Any]) -> float:
            params_list.append(params.copy())
            return params["lr"] + params["batch_size"] / 1000

        optimizer.optimize(objective, parameter_space, n_calls=10)

        # 全パラメータが適切な型と範囲
        for params in params_list:
            assert 0.001 <= params["lr"] <= 0.1
            assert isinstance(params["batch_size"], int)
            assert 16 <= params["batch_size"] <= 256
            assert params["activation"] in ["relu", "tanh", "sigmoid"]


class TestOptimizationControl:
    """最適化制御のテスト"""

    @pytest.fixture
    def optimizer(self):
        return OptunaOptimizer()

    def test_n_calls_respected(self, optimizer):
        """試行回数が守られる"""
        parameter_space = {"x": ParameterSpace(type="real", low=0.0, high=1.0)}

        call_count = []

        def objective(params: Dict[str, Any]) -> float:
            call_count.append(1)
            return params["x"]

        result = optimizer.optimize(objective, parameter_space, n_calls=15)

        assert len(call_count) == 15
        assert result.total_evaluations == 15

    def test_optimization_time_recorded(self, optimizer):
        """最適化時間が記録される"""
        parameter_space = {"x": ParameterSpace(type="real", low=0.0, high=1.0)}

        def objective(params: Dict[str, Any]) -> float:
            time.sleep(0.01)  # 少し待機
            return params["x"]

        result = optimizer.optimize(objective, parameter_space, n_calls=5)

        # 最低でも0.05秒はかかるはず
        assert result.optimization_time >= 0.05

    def test_best_params_tracked(self, optimizer):
        """最良パラメータが追跡される"""
        parameter_space = {"x": ParameterSpace(type="real", low=-10.0, high=10.0)}

        def objective(params: Dict[str, Any]) -> float:
            # x=0で最大値
            return -(params["x"] ** 2)

        result = optimizer.optimize(objective, parameter_space, n_calls=20)

        # 最良パラメータはx=0に近いはず
        assert abs(result.best_params["x"]) < 2.0
        assert result.best_score > -4.0  # 0付近の二乗値


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    @pytest.fixture
    def optimizer(self):
        return OptunaOptimizer()

    def test_objective_function_exception(self, optimizer):
        """目的関数で例外が発生した場合"""
        parameter_space = {"x": ParameterSpace(type="real", low=0.0, high=10.0)}

        call_count = [0]

        def failing_objective(params: Dict[str, Any]) -> float:
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise ValueError("Test error")
            return params["x"]

        # Optunaは例外を処理して最適化を継続
        result = optimizer.optimize(
            objective_function=failing_objective,
            parameter_space=parameter_space,
            n_calls=10,
        )

        # 一部の試行は成功しているはず
        assert result.total_evaluations <= 10
        assert result.best_params is not None

    def test_invalid_parameter_space_type(self, optimizer):
        """無効なパラメータタイプ"""
        parameter_space = {
            "x": ParameterSpace(type="invalid_type", low=0.0, high=1.0)  # type: ignore
        }

        def objective(params: Dict[str, Any]) -> float:
            return params.get("x", 0.0)

        # 無効なタイプは処理されないが、エラーも発生しない（パラメータが空になるだけ）
        result = optimizer.optimize(objective, parameter_space, n_calls=5)
        # パラメータが空であることを確認
        assert result.best_params == {}

    def test_missing_bounds_for_real(self, optimizer):
        """実数パラメータで境界が欠落"""
        parameter_space = {
            "x": ParameterSpace(type="real", low=None, high=None)  # type: ignore
        }

        def objective(params: Dict[str, Any]) -> float:
            return params.get("x", 0.0)

        with pytest.raises(AssertionError):
            optimizer.optimize(objective, parameter_space, n_calls=5)

    def test_missing_categories_for_categorical(self, optimizer):
        """カテゴリカルパラメータでカテゴリが欠落"""
        parameter_space = {
            "x": ParameterSpace(type="categorical", categories=None)  # type: ignore
        }

        def objective(params: Dict[str, Any]) -> float:
            return 1.0

        with pytest.raises(AssertionError):
            optimizer.optimize(objective, parameter_space, n_calls=5)


class TestCleanup:
    """クリーンアップ機能のテスト"""

    def test_cleanup_clears_study(self):
        """クリーンアップがスタディをクリアする"""
        optimizer = OptunaOptimizer()
        parameter_space = {"x": ParameterSpace(type="real", low=0.0, high=1.0)}

        def objective(params: Dict[str, Any]) -> float:
            return params["x"]

        optimizer.optimize(objective, parameter_space, n_calls=5)

        assert optimizer.study is not None
        optimizer.cleanup()
        assert optimizer.study is None

    def test_cleanup_safe_when_no_study(self):
        """スタディがない場合のクリーンアップは安全"""
        optimizer = OptunaOptimizer()
        assert optimizer.study is None

        # エラーが発生しないことを確認
        optimizer.cleanup()
        assert optimizer.study is None

    def test_destructor_calls_cleanup(self):
        """デストラクタがクリーンアップを呼ぶ"""
        optimizer = OptunaOptimizer()
        parameter_space = {"x": ParameterSpace(type="real", low=0.0, high=1.0)}

        def objective(params: Dict[str, Any]) -> float:
            return params["x"]

        optimizer.optimize(objective, parameter_space, n_calls=3)
        assert optimizer.study is not None

        # デストラクタが呼ばれる
        del optimizer
        # メモリが解放されることを期待（明示的な検証は困難）


class TestDefaultParameterSpace:
    """デフォルトパラメータ空間のテスト"""

    def test_get_default_parameter_space(self):
        """デフォルトパラメータ空間の取得"""
        space = OptunaOptimizer.get_default_parameter_space()

        assert "num_leaves" in space
        assert "learning_rate" in space
        assert "feature_fraction" in space
        assert "bagging_fraction" in space
        assert "min_data_in_leaf" in space
        assert "max_depth" in space

        # 各パラメータの型を確認
        assert space["num_leaves"].type == "integer"
        assert space["learning_rate"].type == "real"
        assert space["feature_fraction"].type == "real"

    def test_default_space_with_optimizer(self):
        """デフォルトパラメータ空間で最適化"""
        optimizer = OptunaOptimizer()
        space = OptunaOptimizer.get_default_parameter_space()

        def objective(params: Dict[str, Any]) -> float:
            # LightGBMパラメータの簡易評価
            return (
                params["learning_rate"] * 10
                + params["num_leaves"] / 100
                + params["feature_fraction"]
            )

        result = optimizer.optimize(objective, space, n_calls=10)

        # 全てのパラメータが存在
        assert all(key in result.best_params for key in space.keys())


class TestEnsembleParameterSpace:
    """アンサンブルパラメータ空間のテスト"""

    def test_get_ensemble_parameter_space_stacking(self):
        """スタッキング用パラメータ空間"""
        space = OptunaOptimizer.get_ensemble_parameter_space(
            ensemble_method="stacking", enabled_models=["lightgbm", "xgboost"]
        )

        # LightGBMパラメータ
        assert "lgb_num_leaves" in space
        assert "lgb_learning_rate" in space

        # XGBoostパラメータ
        assert "xgb_max_depth" in space
        assert "xgb_learning_rate" in space

        # スタッキング固有パラメータ
        assert "stacking_meta_C" in space
        assert "stacking_cv_folds" in space

    def test_get_ensemble_parameter_space_single_model(self):
        """単一モデルのパラメータ空間"""
        space = OptunaOptimizer.get_ensemble_parameter_space(
            ensemble_method="stacking", enabled_models=["lightgbm"]
        )

        # LightGBMパラメータのみ
        assert "lgb_num_leaves" in space
        assert "lgb_learning_rate" in space

        # XGBoostパラメータは含まれない
        assert "xgb_max_depth" not in space


class TestIntegration:
    """統合テスト"""

    @pytest.fixture
    def sample_training_data(self):
        """サンプル訓練データ"""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        X_val = np.random.randn(20, 5)
        y_val = np.random.randn(20)
        return X_train, y_train, X_val, y_val

    def test_optimize_ml_hyperparameters(self, sample_training_data):
        """MLハイパーパラメータの最適化統合テスト"""
        X_train, y_train, X_val, y_val = sample_training_data

        optimizer = OptunaOptimizer()
        parameter_space = {
            "learning_rate": ParameterSpace(type="real", low=0.01, high=0.1),
            "max_depth": ParameterSpace(type="integer", low=3, high=10),
            "num_estimators": ParameterSpace(type="integer", low=50, high=200),
        }

        def objective(params: Dict[str, Any]) -> float:
            # 簡易的なモデル評価（実際のLightGBMの代わり）
            lr = params["learning_rate"]
            depth = params["max_depth"]
            n_est = params["num_estimators"]

            # ダミースコア計算
            score = lr * 5 + depth / 10 + n_est / 100

            # バリデーションセットで評価（ダミー）
            mse = np.random.rand() * 0.1 + (1.0 - score / 10)
            return -mse  # 最小化なので負の値

        result = optimizer.optimize(objective, parameter_space, n_calls=10)

        assert "learning_rate" in result.best_params
        assert "max_depth" in result.best_params
        assert "num_estimators" in result.best_params
        assert result.total_evaluations == 10
        assert result.best_score is not None

    def test_ga_parameter_optimization(self):
        """GAパラメータ最適化の統合テスト"""
        optimizer = OptunaOptimizer()
        parameter_space = {
            "population_size": ParameterSpace(type="integer", low=50, high=200),
            "crossover_prob": ParameterSpace(type="real", low=0.6, high=0.9),
            "mutation_prob": ParameterSpace(type="real", low=0.01, high=0.2),
        }

        def objective(params: Dict[str, Any]) -> float:
            # GAパラメータの評価（ダミー）
            pop_size = params["population_size"]
            cx_prob = params["crossover_prob"]
            mut_prob = params["mutation_prob"]

            # 適当なフィットネス計算
            fitness = (
                1.0 / pop_size * 10000 + cx_prob * 2 - mut_prob * 3
            )  # 小さい集団サイズと高い交叉率を好む

            return fitness

        result = optimizer.optimize(objective, parameter_space, n_calls=15)

        assert result.total_evaluations == 15
        assert 50 <= result.best_params["population_size"] <= 200
        assert 0.6 <= result.best_params["crossover_prob"] <= 0.9
        assert 0.01 <= result.best_params["mutation_prob"] <= 0.2

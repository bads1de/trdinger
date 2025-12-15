"""
Optuna最適化機能統合テスト

feature_evaluator.pyにOptunaハイパーパラメータ最適化を
統合したテストスイート。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.ml.optimization.optuna_optimizer import OptimizationResult
from scripts.feature_evaluation.feature_evaluator import (
    FeatureEvaluator,
    FeatureEvaluationConfig,
)


@pytest.fixture
def sample_training_data():
    """サンプル学習データを生成（3クラス分類）"""
    np.random.seed(42)
    n_samples = 200

    dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="1h")

    # 特徴量データ
    X = pd.DataFrame(
        {
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
            "feature_4": np.random.randn(n_samples),
            "feature_5": np.random.randn(n_samples),
        },
        index=dates,
    )

    # ターゲット変数（3クラス分類: 0=DOWN, 1=RANGE, 2=UP）
    y = pd.Series(
        np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3]),
        index=dates,
    )

    return X, y


@pytest.fixture
def mock_optuna_optimizer():
    """OptunaOptimizerのモック"""
    with patch("scripts.feature_evaluation.feature_evaluator.OptunaOptimizer") as mock:
        optimizer_instance = MagicMock()
        mock.return_value = optimizer_instance

        # モック最適化結果
        best_params = {
            "lgb_num_leaves": 50,
            "lgb_learning_rate": 0.05,
            "xgb_max_depth": 8,
            "xgb_learning_rate": 0.05,
        }

        mock_result = OptimizationResult(
            best_params=best_params,
            best_score=0.42,
            total_evaluations=50,
            optimization_time=120.5,
            study=MagicMock(),
        )

        optimizer_instance.optimize.return_value = mock_result
        yield mock


class TestFeatureEvaluatorOptimization:
    """FeatureEvaluatorのOptuna最適化テスト"""

    def test_initialization_with_optuna(self):
        """Optuna有効時の初期化テスト"""
        config = FeatureEvaluationConfig(optimize=True, n_trials=10)
        evaluator = FeatureEvaluator(MagicMock(), config)

        assert evaluator.config.optimize is True
        assert evaluator.config.n_trials == 10

    def test_optimize_hyperparameters_lightgbm(
        self, sample_training_data, mock_optuna_optimizer
    ):
        """LightGBMハイパーパラメータ最適化テスト"""
        X, y = sample_training_data
        config = FeatureEvaluationConfig(optimize=True, n_trials=10)
        evaluator = FeatureEvaluator(MagicMock(), config)

        # モック: _evaluate_with_cv
        with patch.object(
            evaluator, "_evaluate_with_cv", return_value={"f1_score": 0.5}
        ):
            best_params = evaluator._optimize_hyperparameters(X, y, "lightgbm")

            # オプティマイザが呼ばれたか
            mock_optuna_optimizer.return_value.optimize.assert_called_once()

            # パラメータ空間がLightGBMのものか
            call_args = mock_optuna_optimizer.return_value.optimize.call_args
            assert "parameter_space" in call_args.kwargs
            param_space = call_args.kwargs["parameter_space"]
            assert any(k.startswith("lgb_") for k in param_space.keys())

            # 結果の検証 (prefixが削除されていること)
            assert "num_leaves" in best_params
            assert "lgb_num_leaves" not in best_params
            assert best_params["num_leaves"] == 50

    def test_optimize_hyperparameters_xgboost(
        self, sample_training_data, mock_optuna_optimizer
    ):
        """XGBoostハイパーパラメータ最適化テスト"""
        X, y = sample_training_data
        config = FeatureEvaluationConfig(optimize=True, n_trials=10)
        evaluator = FeatureEvaluator(MagicMock(), config)

        # モック: _evaluate_with_cv
        with patch.object(
            evaluator, "_evaluate_with_cv", return_value={"f1_score": 0.5}
        ):
            best_params = evaluator._optimize_hyperparameters(X, y, "xgboost")

            # オプティマイザが呼ばれたか
            mock_optuna_optimizer.return_value.optimize.assert_called_once()

            # パラメータ空間がXGBoostのものか
            call_args = mock_optuna_optimizer.return_value.optimize.call_args
            assert "parameter_space" in call_args.kwargs
            param_space = call_args.kwargs["parameter_space"]
            assert any(k.startswith("xgb_") for k in param_space.keys())

            # 結果の検証
            assert "max_depth" in best_params
            assert best_params["max_depth"] == 8

    def test_optimize_disabled(self, sample_training_data, mock_optuna_optimizer):
        """Optuna無効時の挙動テスト"""
        X, y = sample_training_data
        config = FeatureEvaluationConfig(optimize=False)
        evaluator = FeatureEvaluator(MagicMock(), config)

        best_params = evaluator._optimize_hyperparameters(X, y, "lightgbm")

        # オプティマイザが呼ばれていないこと
        mock_optuna_optimizer.return_value.optimize.assert_not_called()
        assert best_params == {}

    def test_analyze_importance_integration(
        self, sample_training_data, mock_optuna_optimizer
    ):
        """analyze_importanceとの統合テスト"""
        X, y = sample_training_data
        config = FeatureEvaluationConfig(
            optimize=True, n_trials=5, use_pipeline_method=True, model="lightgbm"
        )
        evaluator = FeatureEvaluator(MagicMock(), config)

        # 依存メソッドをモック
        evaluator._evaluate_with_cv = MagicMock(return_value={"f1_score": 0.5})
        evaluator._calculate_lightgbm_importance = MagicMock(return_value={"feat": 0.1})

        evaluator.analyze_importance(X, y, "lightgbm")

        # 最適化が実行されたか
        mock_optuna_optimizer.return_value.optimize.assert_called()

        # 計算メソッドにパラメータが渡されたか
        call_args = evaluator._calculate_lightgbm_importance.call_args
        assert "params" in call_args.kwargs
        assert call_args.kwargs["params"]["num_leaves"] == 50  # mockの戻り値


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



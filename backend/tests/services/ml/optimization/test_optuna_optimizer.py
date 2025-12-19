import pytest
import numpy as np
from app.services.ml.optimization.optuna_optimizer import OptunaOptimizer, ParameterSpace, OptimizationResult
from app.services.ml.optimization.ensemble_parameter_space import EnsembleParameterSpace

class TestOptunaOptimizer:
    @pytest.fixture
    def optimizer(self):
        return OptunaOptimizer()

    def test_suggest_parameters(self, optimizer):
        """パラメータ提案ロジックのテスト"""
        space = {
            "p1": ParameterSpace(type="real", low=0.0, high=1.0),
            "p2": ParameterSpace(type="integer", low=1, high=10),
            "p3": ParameterSpace(type="categorical", categories=["a", "b"])
        }
        
        trial = MagicMock()
        trial.suggest_float.return_value = 0.5
        trial.suggest_int.return_value = 5
        trial.suggest_categorical.return_value = "a"
        
        params = optimizer._suggest_parameters(trial, space)
        
        assert params["p1"] == 0.5
        assert params["p2"] == 5
        assert params["p3"] == "a"

    def test_optimize_success(self, optimizer):
        """最適化実行の成功テスト"""
        def dummy_objective(params):
            # p1 + p2 を最大化する単純な目的関数
            return params["p1"] + params["p2"]
            
        space = {
            "p1": ParameterSpace(type="real", low=0.0, high=1.0),
            "p2": ParameterSpace(type="integer", low=1, high=10)
        }
        
        # 試行回数を少なくして実行
        result = optimizer.optimize(dummy_objective, space, n_calls=10)
        
        assert isinstance(result, OptimizationResult)
        assert result.total_evaluations == 10
        assert "p1" in result.best_params
        assert "p2" in result.best_params
        assert result.best_score > 0

    def test_optimize_error_handling(self, optimizer):
        """目的関数内でエラーが発生した場合のハンドリング"""
        def error_objective(params):
            raise ValueError("Test error")
            
        space = {"p1": ParameterSpace(type="real", low=0.0, high=1.0)}
        
        # エラーが発生しても最適化全体は止まらず、TrialPrunedされる（はず）
        # ただし全試行がエラーだと有効な結果が得られない
        with pytest.raises(Exception):
            optimizer.optimize(error_objective, space, n_calls=2)

    def test_cleanup(self, optimizer):
        """リソース解放のテスト"""
        space = {"p1": ParameterSpace(type="real", low=0.0, high=1.0)}
        optimizer.optimize(lambda p: 0.5, space, n_calls=1)
        
        assert optimizer.study is not None
        optimizer.cleanup()
        assert optimizer.study is None

class TestEnsembleParameterSpace:
    def test_get_ensemble_parameter_space(self):
        """統合パラメータ空間の取得"""
        # LightGBM + Stacking
        ps = EnsembleParameterSpace.get_ensemble_parameter_space("stacking", ["lightgbm"])
        
        # LGBMパラメータが含まれているか
        assert "lgb_num_leaves" in ps
        # Stackingパラメータが含まれているか
        assert "stacking_meta_C" in ps
        # XGBoostは含まれていないはず
        assert "xgb_max_depth" not in ps

    def test_get_xgboost_parameter_space(self):
        """XGBoost用空間の取得"""
        ps = EnsembleParameterSpace.get_xgboost_parameter_space()
        assert "xgb_learning_rate" in ps
        assert ps["xgb_learning_rate"].type == "real"

    def test_suggest_methods_smoke(self):
        """各suggestメソッドの動作確認（Optuna Trialとの連携）"""
        trial = MagicMock()
        
        # 各メソッドがエラーなく呼ばれるか
        EnsembleParameterSpace._suggest_lightgbm_params(trial)
        EnsembleParameterSpace._suggest_xgboost_params(trial)
        EnsembleParameterSpace._suggest_stacking_params(trial)
        
        assert trial.suggest_int.called
        assert trial.suggest_float.called
        assert trial.suggest_categorical.called

from unittest.mock import MagicMock

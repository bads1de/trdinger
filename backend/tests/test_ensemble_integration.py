"""
アンサンブルパラメータスペース統合テスト
実際のOptuna最適化との統合をテストします
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import optuna
from app.services.optimization.ensemble_parameter_space import EnsembleParameterSpace


class TestEnsembleIntegration:
    """アンサンブルパラメータスペース統合テストクラス"""

    def test_optuna_integration_all_models(self):
        """全モデルでのOptuna統合テスト"""
        all_models = [
            "lightgbm", "xgboost", "randomforest", "catboost", "tabnet",
            "adaboost", "extratrees", "gradientboosting", "knn", "ridge", "naivebayes"
        ]
        
        # バギングアンサンブルのパラメータ空間を取得
        parameter_space = EnsembleParameterSpace.get_ensemble_parameter_space(
            "bagging", all_models
        )
        
        # Optunaスタディを作成
        study = optuna.create_study(direction="maximize")
        
        def objective(trial):
            # 各パラメータをOptunaで提案
            params = {}
            for param_name, param_space in parameter_space.items():
                if param_space.type == "integer":
                    params[param_name] = trial.suggest_int(
                        param_name, param_space.low, param_space.high
                    )
                elif param_space.type == "real":
                    params[param_name] = trial.suggest_float(
                        param_name, param_space.low, param_space.high
                    )
                elif param_space.type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_space.categories
                    )
            
            # ダミーの目的関数（実際の学習は行わない）
            return 0.8
        
        # 最適化を実行（少数回のみ）
        study.optimize(objective, n_trials=3)
        
        # 結果を確認
        assert len(study.trials) == 3
        assert study.best_value == 0.8
        
        # 最適なパラメータが全て含まれているか確認
        best_params = study.best_params
        
        # 各モデルのパラメータが含まれているか確認
        model_prefixes = ["lgb_", "xgb_", "rf_", "cat_", "tab_", "ada_", "et_", "gb_", "knn_", "ridge_", "nb_"]
        for prefix in model_prefixes:
            has_prefix = any(key.startswith(prefix) for key in best_params.keys())
            assert has_prefix, f"Parameters with prefix '{prefix}' not found in best_params"
        
        # バギング固有のパラメータが含まれているか確認
        assert any(key.startswith("bagging_") for key in best_params.keys())

    def test_optuna_integration_stacking(self):
        """スタッキングアンサンブルでのOptuna統合テスト"""
        selected_models = ["lightgbm", "xgboost", "adaboost", "ridge"]
        
        # スタッキングアンサンブルのパラメータ空間を取得
        parameter_space = EnsembleParameterSpace.get_ensemble_parameter_space(
            "stacking", selected_models
        )
        
        # Optunaスタディを作成
        study = optuna.create_study(direction="maximize")
        
        def objective(trial):
            params = {}
            for param_name, param_space in parameter_space.items():
                if param_space.type == "integer":
                    params[param_name] = trial.suggest_int(
                        param_name, param_space.low, param_space.high
                    )
                elif param_space.type == "real":
                    params[param_name] = trial.suggest_float(
                        param_name, param_space.low, param_space.high
                    )
                elif param_space.type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_space.categories
                    )
            
            return 0.85
        
        # 最適化を実行
        study.optimize(objective, n_trials=2)
        
        # 結果を確認
        assert len(study.trials) == 2
        best_params = study.best_params
        
        # 選択されたモデルのパラメータが含まれているか確認
        assert any(key.startswith("lgb_") for key in best_params.keys())
        assert any(key.startswith("xgb_") for key in best_params.keys())
        assert any(key.startswith("ada_") for key in best_params.keys())
        assert any(key.startswith("ridge_") for key in best_params.keys())
        
        # スタッキング固有のパラメータが含まれているか確認
        assert any(key.startswith("stacking_") for key in best_params.keys())

    def test_parameter_value_ranges(self):
        """パラメータ値の範囲テスト"""
        models = ["adaboost", "extratrees", "knn"]
        parameter_space = EnsembleParameterSpace.get_ensemble_parameter_space(
            "bagging", models
        )
        
        study = optuna.create_study(direction="maximize")
        
        def objective(trial):
            params = {}
            for param_name, param_space in parameter_space.items():
                if param_space.type == "integer":
                    value = trial.suggest_int(
                        param_name, param_space.low, param_space.high
                    )
                    # 値が範囲内にあることを確認
                    assert param_space.low <= value <= param_space.high
                    params[param_name] = value
                elif param_space.type == "real":
                    value = trial.suggest_float(
                        param_name, param_space.low, param_space.high
                    )
                    # 値が範囲内にあることを確認
                    assert param_space.low <= value <= param_space.high
                    params[param_name] = value
                elif param_space.type == "categorical":
                    value = trial.suggest_categorical(
                        param_name, param_space.categories
                    )
                    # 値がカテゴリに含まれることを確認
                    assert value in param_space.categories
                    params[param_name] = value
            
            return 0.9
        
        # 最適化を実行
        study.optimize(objective, n_trials=5)
        
        # 全てのトライアルが成功したことを確認
        assert len(study.trials) == 5
        assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)

    def test_new_algorithms_parameter_validation(self):
        """新しく追加されたアルゴリズムのパラメータ検証"""
        new_algorithms = ["adaboost", "extratrees", "gradientboosting", "knn", "ridge", "naivebayes"]
        
        for algorithm in new_algorithms:
            parameter_space = EnsembleParameterSpace.get_ensemble_parameter_space(
                "bagging", [algorithm]
            )
            
            # アルゴリズム固有のパラメータが存在することを確認
            algorithm_params = [key for key in parameter_space.keys() if not key.startswith("bagging_")]
            assert len(algorithm_params) > 0, f"No parameters found for {algorithm}"
            
            # パラメータの型と範囲が正しく設定されていることを確認
            for param_name, param_space in parameter_space.items():
                if not param_name.startswith("bagging_"):
                    assert param_space.type in ["integer", "real", "categorical"]
                    
                    if param_space.type in ["integer", "real"]:
                        assert hasattr(param_space, "low")
                        assert hasattr(param_space, "high")
                        assert param_space.low < param_space.high
                    elif param_space.type == "categorical":
                        assert hasattr(param_space, "categories")
                        assert len(param_space.categories) > 0

    def test_ensemble_method_validation(self):
        """アンサンブル手法の検証"""
        models = ["lightgbm", "xgboost"]
        
        # バギングアンサンブル
        bagging_params = EnsembleParameterSpace.get_ensemble_parameter_space(
            "bagging", models
        )
        bagging_specific = [key for key in bagging_params.keys() if key.startswith("bagging_")]
        assert len(bagging_specific) > 0
        assert not any(key.startswith("stacking_") for key in bagging_params.keys())
        
        # スタッキングアンサンブル
        stacking_params = EnsembleParameterSpace.get_ensemble_parameter_space(
            "stacking", models
        )
        stacking_specific = [key for key in stacking_params.keys() if key.startswith("stacking_")]
        assert len(stacking_specific) > 0
        assert not any(key.startswith("bagging_") for key in stacking_params.keys())


if __name__ == "__main__":
    pytest.main([__file__])

"""
アンサンブルパラメータスペースのテスト
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.services.optimization.ensemble_parameter_space import EnsembleParameterSpace
from app.services.optimization.optuna_optimizer import ParameterSpace


class TestEnsembleParameterSpace:
    """アンサンブルパラメータスペースのテストクラス"""

    def test_lightgbm_parameter_space(self):
        """LightGBMパラメータ空間のテスト"""
        params = EnsembleParameterSpace.get_lightgbm_parameter_space()

        assert "lgb_num_leaves" in params
        assert "lgb_learning_rate" in params
        assert "lgb_feature_fraction" in params
        assert params["lgb_num_leaves"].type == "integer"
        assert params["lgb_learning_rate"].type == "real"

    def test_xgboost_parameter_space(self):
        """XGBoostパラメータ空間のテスト"""
        params = EnsembleParameterSpace.get_xgboost_parameter_space()

        assert "xgb_max_depth" in params
        assert "xgb_learning_rate" in params
        assert "xgb_subsample" in params
        assert params["xgb_max_depth"].type == "integer"
        assert params["xgb_learning_rate"].type == "real"

    def test_randomforest_parameter_space(self):
        """RandomForestパラメータ空間のテスト"""
        params = EnsembleParameterSpace.get_randomforest_parameter_space()

        assert "rf_n_estimators" in params
        assert "rf_max_depth" in params
        assert "rf_max_features" in params
        assert params["rf_n_estimators"].type == "integer"
        assert params["rf_max_features"].type == "categorical"

    def test_catboost_parameter_space(self):
        """CatBoostパラメータ空間のテスト"""
        params = EnsembleParameterSpace.get_catboost_parameter_space()

        assert "cat_iterations" in params
        assert "cat_learning_rate" in params
        assert "cat_depth" in params
        assert params["cat_iterations"].type == "integer"
        assert params["cat_learning_rate"].type == "real"

    def test_tabnet_parameter_space(self):
        """TabNetパラメータ空間のテスト"""
        params = EnsembleParameterSpace.get_tabnet_parameter_space()

        assert "tab_n_d" in params
        assert "tab_n_a" in params
        assert "tab_n_steps" in params
        assert params["tab_n_d"].type == "integer"
        assert params["tab_gamma"].type == "real"

    def test_adaboost_parameter_space(self):
        """AdaBoostパラメータ空間のテスト"""
        params = EnsembleParameterSpace.get_adaboost_parameter_space()

        assert "ada_n_estimators" in params
        assert "ada_learning_rate" in params
        assert "ada_algorithm" in params
        assert params["ada_n_estimators"].type == "integer"
        assert params["ada_learning_rate"].type == "real"
        assert params["ada_algorithm"].type == "categorical"
        assert "SAMME" in params["ada_algorithm"].categories
        assert "SAMME.R" in params["ada_algorithm"].categories

    def test_extratrees_parameter_space(self):
        """ExtraTreesパラメータ空間のテスト"""
        params = EnsembleParameterSpace.get_extratrees_parameter_space()

        assert "et_n_estimators" in params
        assert "et_max_depth" in params
        assert "et_max_features" in params
        assert params["et_n_estimators"].type == "integer"
        assert params["et_max_features"].type == "categorical"
        assert "sqrt" in params["et_max_features"].categories

    def test_gradientboosting_parameter_space(self):
        """GradientBoostingパラメータ空間のテスト"""
        params = EnsembleParameterSpace.get_gradientboosting_parameter_space()

        assert "gb_n_estimators" in params
        assert "gb_learning_rate" in params
        assert "gb_max_depth" in params
        assert "gb_subsample" in params
        assert params["gb_n_estimators"].type == "integer"
        assert params["gb_learning_rate"].type == "real"

    def test_knn_parameter_space(self):
        """KNNパラメータ空間のテスト"""
        params = EnsembleParameterSpace.get_knn_parameter_space()

        assert "knn_n_neighbors" in params
        assert "knn_weights" in params
        assert "knn_algorithm" in params
        assert params["knn_n_neighbors"].type == "integer"
        assert params["knn_weights"].type == "categorical"
        assert "uniform" in params["knn_weights"].categories
        assert "distance" in params["knn_weights"].categories

    def test_ridge_parameter_space(self):
        """Ridgeパラメータ空間のテスト"""
        params = EnsembleParameterSpace.get_ridge_parameter_space()

        assert "ridge_alpha" in params
        assert "ridge_solver" in params
        assert "ridge_max_iter" in params
        assert params["ridge_alpha"].type == "real"
        assert params["ridge_solver"].type == "categorical"
        assert "auto" in params["ridge_solver"].categories

    def test_naivebayes_parameter_space(self):
        """NaiveBayesパラメータ空間のテスト"""
        params = EnsembleParameterSpace.get_naivebayes_parameter_space()

        assert "nb_alpha" in params
        assert "nb_fit_prior" in params
        assert params["nb_alpha"].type == "real"
        assert params["nb_fit_prior"].type == "categorical"
        assert True in params["nb_fit_prior"].categories
        assert False in params["nb_fit_prior"].categories

    def test_bagging_parameter_space(self):
        """バギングパラメータ空間のテスト"""
        params = EnsembleParameterSpace.get_bagging_parameter_space()

        assert "bagging_n_estimators" in params
        assert "bagging_max_samples" in params
        assert "bagging_max_features" in params
        assert params["bagging_n_estimators"].type == "integer"
        assert params["bagging_max_samples"].type == "real"

    def test_stacking_parameter_space(self):
        """スタッキングパラメータ空間のテスト"""
        params = EnsembleParameterSpace.get_stacking_parameter_space()

        assert "stacking_meta_C" in params
        assert "stacking_meta_penalty" in params
        assert "stacking_cv_folds" in params
        assert params["stacking_meta_C"].type == "real"
        assert params["stacking_meta_penalty"].type == "categorical"

    def test_ensemble_parameter_space_bagging(self):
        """バギングアンサンブルパラメータ空間の統合テスト"""
        enabled_models = ["lightgbm", "xgboost", "adaboost", "extratrees"]
        params = EnsembleParameterSpace.get_ensemble_parameter_space(
            "bagging", enabled_models
        )

        # 各モデルのパラメータが含まれているか確認
        assert "lgb_num_leaves" in params
        assert "xgb_max_depth" in params
        assert "ada_n_estimators" in params
        assert "et_n_estimators" in params

        # バギング固有のパラメータが含まれているか確認
        assert "bagging_n_estimators" in params
        assert "bagging_max_samples" in params

    def test_ensemble_parameter_space_stacking(self):
        """スタッキングアンサンブルパラメータ空間の統合テスト"""
        enabled_models = ["catboost", "randomforest", "knn", "ridge"]
        params = EnsembleParameterSpace.get_ensemble_parameter_space(
            "stacking", enabled_models
        )

        # 各モデルのパラメータが含まれているか確認
        assert "cat_iterations" in params
        assert "rf_n_estimators" in params
        assert "knn_n_neighbors" in params
        assert "ridge_alpha" in params

        # スタッキング固有のパラメータが含まれているか確認
        assert "stacking_meta_C" in params
        assert "stacking_cv_folds" in params

    def test_all_models_ensemble(self):
        """全モデルを含むアンサンブルのテスト"""
        all_models = [
            "lightgbm",
            "xgboost",
            "randomforest",
            "catboost",
            "tabnet",
            "adaboost",
            "extratrees",
            "gradientboosting",
            "knn",
            "ridge",
            "naivebayes",
        ]

        params = EnsembleParameterSpace.get_ensemble_parameter_space(
            "bagging", all_models
        )

        # 全モデルのパラメータが含まれているか確認
        model_prefixes = [
            "lgb_",
            "xgb_",
            "rf_",
            "cat_",
            "tab_",
            "ada_",
            "et_",
            "gb_",
            "knn_",
            "ridge_",
            "nb_",
        ]

        for prefix in model_prefixes:
            has_prefix = any(key.startswith(prefix) for key in params.keys())
            assert has_prefix, f"Parameters with prefix '{prefix}' not found"

    def test_parameter_ranges(self):
        """パラメータ範囲の妥当性テスト"""
        # AdaBoostのテスト
        ada_params = EnsembleParameterSpace.get_adaboost_parameter_space()
        assert ada_params["ada_n_estimators"].low == 50
        assert ada_params["ada_n_estimators"].high == 300
        assert ada_params["ada_learning_rate"].low == 0.01
        assert ada_params["ada_learning_rate"].high == 2.0

        # KNNのテスト
        knn_params = EnsembleParameterSpace.get_knn_parameter_space()
        assert knn_params["knn_n_neighbors"].low == 3
        assert knn_params["knn_n_neighbors"].high == 20

        # Ridgeのテスト
        ridge_params = EnsembleParameterSpace.get_ridge_parameter_space()
        assert ridge_params["ridge_alpha"].low == 0.01
        assert ridge_params["ridge_alpha"].high == 100.0


if __name__ == "__main__":
    pytest.main([__file__])

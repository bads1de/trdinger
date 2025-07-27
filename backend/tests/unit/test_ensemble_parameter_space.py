"""
EnsembleParameterSpaceのユニットテスト
"""

import pytest
from app.core.services.optimization.ensemble_parameter_space import (
    EnsembleParameterSpace,
)
from app.core.services.optimization.optuna_optimizer import ParameterSpace


class TestEnsembleParameterSpace:
    """EnsembleParameterSpaceのテストクラス"""

    def test_get_lightgbm_parameter_space(self):
        """LightGBMパラメータ空間のテスト"""
        param_space = EnsembleParameterSpace.get_lightgbm_parameter_space()

        # 必要なパラメータが含まれていることを確認
        expected_params = [
            "lgb_num_leaves",
            "lgb_learning_rate",
            "lgb_feature_fraction",
            "lgb_bagging_fraction",
            "lgb_min_data_in_leaf",
            "lgb_max_depth",
            "lgb_reg_alpha",
            "lgb_reg_lambda",
        ]

        for param in expected_params:
            assert param in param_space
            assert isinstance(param_space[param], ParameterSpace)

        # パラメータ範囲の妥当性確認
        assert param_space["lgb_num_leaves"].low == 10
        assert param_space["lgb_num_leaves"].high == 100
        assert param_space["lgb_learning_rate"].low == 0.01
        assert param_space["lgb_learning_rate"].high == 0.3

    def test_get_xgboost_parameter_space(self):
        """XGBoostパラメータ空間のテスト"""
        param_space = EnsembleParameterSpace.get_xgboost_parameter_space()

        expected_params = [
            "xgb_max_depth",
            "xgb_learning_rate",
            "xgb_subsample",
            "xgb_colsample_bytree",
            "xgb_min_child_weight",
            "xgb_reg_alpha",
            "xgb_reg_lambda",
            "xgb_gamma",
        ]

        for param in expected_params:
            assert param in param_space
            assert isinstance(param_space[param], ParameterSpace)

        # パラメータ範囲の妥当性確認
        assert param_space["xgb_max_depth"].low == 3
        assert param_space["xgb_max_depth"].high == 15
        assert param_space["xgb_learning_rate"].low == 0.01
        assert param_space["xgb_learning_rate"].high == 0.3

    def test_get_randomforest_parameter_space(self):
        """RandomForestパラメータ空間のテスト"""
        param_space = EnsembleParameterSpace.get_randomforest_parameter_space()

        expected_params = [
            "rf_n_estimators",
            "rf_max_depth",
            "rf_min_samples_split",
            "rf_min_samples_leaf",
            "rf_max_features",
            "rf_bootstrap",
        ]

        for param in expected_params:
            assert param in param_space
            assert isinstance(param_space[param], ParameterSpace)

        # カテゴリカルパラメータの確認
        assert param_space["rf_max_features"].type == "categorical"
        assert "sqrt" in param_space["rf_max_features"].categories
        assert param_space["rf_bootstrap"].type == "categorical"
        assert True in param_space["rf_bootstrap"].categories

    def test_get_catboost_parameter_space(self):
        """CatBoostパラメータ空間のテスト"""
        param_space = EnsembleParameterSpace.get_catboost_parameter_space()

        expected_params = [
            "cat_iterations",
            "cat_learning_rate",
            "cat_depth",
            "cat_l2_leaf_reg",
            "cat_border_count",
            "cat_bagging_temperature",
            "cat_random_strength",
            "cat_subsample",
            "cat_colsample_bylevel",
        ]

        for param in expected_params:
            assert param in param_space
            assert isinstance(param_space[param], ParameterSpace)

        # CatBoost固有のパラメータ範囲確認
        assert param_space["cat_iterations"].low == 100
        assert param_space["cat_iterations"].high == 1000
        assert param_space["cat_depth"].low == 3
        assert param_space["cat_depth"].high == 10

    def test_get_tabnet_parameter_space(self):
        """TabNetパラメータ空間のテスト"""
        param_space = EnsembleParameterSpace.get_tabnet_parameter_space()

        expected_params = [
            "tab_n_d",
            "tab_n_a",
            "tab_n_steps",
            "tab_gamma",
            "tab_lambda_sparse",
            "tab_optimizer_lr",
            "tab_scheduler_step_size",
            "tab_scheduler_gamma",
            "tab_n_independent",
            "tab_n_shared",
            "tab_momentum",
        ]

        for param in expected_params:
            assert param in param_space
            assert isinstance(param_space[param], ParameterSpace)

        # TabNet固有のパラメータ範囲確認
        assert param_space["tab_n_d"].low == 8
        assert param_space["tab_n_d"].high == 64
        assert param_space["tab_gamma"].low == 1.0
        assert param_space["tab_gamma"].high == 2.0

    def test_get_bagging_parameter_space(self):
        """バギングパラメータ空間のテスト"""
        param_space = EnsembleParameterSpace.get_bagging_parameter_space()

        expected_params = [
            "bagging_n_estimators",
            "bagging_max_samples",
            "bagging_max_features",
        ]

        for param in expected_params:
            assert param in param_space
            assert isinstance(param_space[param], ParameterSpace)

        # バギング固有のパラメータ範囲確認
        assert param_space["bagging_n_estimators"].low == 3
        assert param_space["bagging_n_estimators"].high == 10

    def test_get_stacking_parameter_space(self):
        """スタッキングパラメータ空間のテスト"""
        param_space = EnsembleParameterSpace.get_stacking_parameter_space()

        expected_params = [
            "stacking_meta_C",
            "stacking_meta_penalty",
            "stacking_meta_solver",
            "stacking_cv_folds",
        ]

        for param in expected_params:
            assert param in param_space
            assert isinstance(param_space[param], ParameterSpace)

        # メタモデルのカテゴリカルパラメータ確認
        assert param_space["stacking_meta_penalty"].type == "categorical"
        assert "l1" in param_space["stacking_meta_penalty"].categories
        assert "l2" in param_space["stacking_meta_penalty"].categories

    def test_get_ensemble_parameter_space_bagging(self):
        """バギングアンサンブルのパラメータ空間統合テスト"""
        enabled_models = ["lightgbm", "xgboost", "randomforest"]
        param_space = EnsembleParameterSpace.get_ensemble_parameter_space(
            "bagging", enabled_models
        )

        # 各モデルのパラメータが含まれていることを確認
        lgb_params = [k for k in param_space.keys() if k.startswith("lgb_")]
        xgb_params = [k for k in param_space.keys() if k.startswith("xgb_")]
        rf_params = [k for k in param_space.keys() if k.startswith("rf_")]
        bagging_params = [k for k in param_space.keys() if k.startswith("bagging_")]

        assert len(lgb_params) > 0, "LightGBMパラメータが含まれていません"
        assert len(xgb_params) > 0, "XGBoostパラメータが含まれていません"
        assert len(rf_params) > 0, "RandomForestパラメータが含まれていません"
        assert len(bagging_params) > 0, "バギングパラメータが含まれていません"

    def test_get_ensemble_parameter_space_stacking(self):
        """スタッキングアンサンブルのパラメータ空間統合テスト"""
        enabled_models = ["lightgbm", "catboost", "tabnet"]
        param_space = EnsembleParameterSpace.get_ensemble_parameter_space(
            "stacking", enabled_models
        )

        # 各モデルのパラメータが含まれていることを確認
        lgb_params = [k for k in param_space.keys() if k.startswith("lgb_")]
        cat_params = [k for k in param_space.keys() if k.startswith("cat_")]
        tab_params = [k for k in param_space.keys() if k.startswith("tab_")]
        stacking_params = [k for k in param_space.keys() if k.startswith("stacking_")]

        assert len(lgb_params) > 0, "LightGBMパラメータが含まれていません"
        assert len(cat_params) > 0, "CatBoostパラメータが含まれていません"
        assert len(tab_params) > 0, "TabNetパラメータが含まれていません"
        assert len(stacking_params) > 0, "スタッキングパラメータが含まれていません"

    def test_get_ensemble_parameter_space_all_models(self):
        """全モデルを含むアンサンブルのパラメータ空間テスト"""
        enabled_models = ["lightgbm", "xgboost", "randomforest", "catboost", "tabnet"]
        param_space = EnsembleParameterSpace.get_ensemble_parameter_space(
            "bagging", enabled_models
        )

        # 全モデルのパラメータが含まれていることを確認
        model_prefixes = ["lgb_", "xgb_", "rf_", "cat_", "tab_"]

        for prefix in model_prefixes:
            model_params = [k for k in param_space.keys() if k.startswith(prefix)]
            assert len(model_params) > 0, f"{prefix}パラメータが含まれていません"

        # バギングパラメータも含まれていることを確認
        bagging_params = [k for k in param_space.keys() if k.startswith("bagging_")]
        assert len(bagging_params) > 0, "バギングパラメータが含まれていません"

    def test_parameter_space_types(self):
        """パラメータ空間の型が正しいことを確認"""
        param_space = EnsembleParameterSpace.get_ensemble_parameter_space(
            "bagging", ["lightgbm", "xgboost"]
        )

        for param_name, param_spec in param_space.items():
            assert isinstance(param_spec, ParameterSpace)
            assert param_spec.type in ["integer", "real", "categorical"]

            if param_spec.type in ["integer", "real"]:
                assert hasattr(param_spec, "low")
                assert hasattr(param_spec, "high")
                assert param_spec.low < param_spec.high
            elif param_spec.type == "categorical":
                assert hasattr(param_spec, "categories")
                assert len(param_spec.categories) > 0


if __name__ == "__main__":
    pytest.main([__file__])

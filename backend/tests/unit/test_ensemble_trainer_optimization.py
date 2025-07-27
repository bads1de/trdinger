"""
EnsembleTrainerのハイパーパラメータ最適化機能のユニットテスト
"""

import pytest
import pandas as pd
import numpy as np
from app.core.services.ml.ensemble.ensemble_trainer import EnsembleTrainer


class TestEnsembleTrainerOptimization:
    """EnsembleTrainerの最適化機能テストクラス"""

    def create_test_data(self):
        """テスト用データを作成"""
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        y = pd.Series(np.random.randint(0, 3, n_samples), name="target")

        return X, y

    def test_extract_optimized_parameters_lightgbm(self):
        """LightGBMパラメータ抽出のテスト"""
        ensemble_config = {"method": "bagging"}
        trainer = EnsembleTrainer(ensemble_config=ensemble_config)

        training_params = {
            "lgb_num_leaves": 50,
            "lgb_learning_rate": 0.1,
            "lgb_max_depth": 8,
            "other_param": "value",
        }

        optimized_params = trainer._extract_optimized_parameters(training_params)

        # LightGBMパラメータが正しく抽出されることを確認
        lgb_params = optimized_params["base_models"]["lightgbm"]
        assert lgb_params["num_leaves"] == 50
        assert lgb_params["learning_rate"] == 0.1
        assert lgb_params["max_depth"] == 8

        # プレフィックスが除去されていることを確認
        assert "lgb_num_leaves" not in lgb_params

    def test_extract_optimized_parameters_xgboost(self):
        """XGBoostパラメータ抽出のテスト"""
        ensemble_config = {"method": "bagging"}
        trainer = EnsembleTrainer(ensemble_config=ensemble_config)

        training_params = {
            "xgb_max_depth": 6,
            "xgb_learning_rate": 0.05,
            "xgb_subsample": 0.8,
            "xgb_colsample_bytree": 0.9,
        }

        optimized_params = trainer._extract_optimized_parameters(training_params)

        # XGBoostパラメータが正しく抽出されることを確認
        xgb_params = optimized_params["base_models"]["xgboost"]
        assert xgb_params["max_depth"] == 6
        assert xgb_params["learning_rate"] == 0.05
        assert xgb_params["subsample"] == 0.8
        assert xgb_params["colsample_bytree"] == 0.9

    def test_extract_optimized_parameters_randomforest(self):
        """RandomForestパラメータ抽出のテスト"""
        ensemble_config = {"method": "bagging"}
        trainer = EnsembleTrainer(ensemble_config=ensemble_config)

        training_params = {
            "rf_n_estimators": 100,
            "rf_max_depth": 10,
            "rf_min_samples_split": 5,
            "rf_bootstrap": True,
        }

        optimized_params = trainer._extract_optimized_parameters(training_params)

        # RandomForestパラメータが正しく抽出されることを確認
        rf_params = optimized_params["base_models"]["randomforest"]
        assert rf_params["n_estimators"] == 100
        assert rf_params["max_depth"] == 10
        assert rf_params["min_samples_split"] == 5
        assert rf_params["bootstrap"] is True

    def test_extract_optimized_parameters_catboost(self):
        """CatBoostパラメータ抽出のテスト"""
        ensemble_config = {"method": "bagging"}
        trainer = EnsembleTrainer(ensemble_config=ensemble_config)

        training_params = {
            "cat_iterations": 500,
            "cat_learning_rate": 0.1,
            "cat_depth": 6,
            "cat_l2_leaf_reg": 3.0,
        }

        optimized_params = trainer._extract_optimized_parameters(training_params)

        # CatBoostパラメータが正しく抽出されることを確認
        cat_params = optimized_params["base_models"]["catboost"]
        assert cat_params["iterations"] == 500
        assert cat_params["learning_rate"] == 0.1
        assert cat_params["depth"] == 6
        assert cat_params["l2_leaf_reg"] == 3.0

    def test_extract_optimized_parameters_tabnet(self):
        """TabNetパラメータ抽出のテスト"""
        ensemble_config = {"method": "bagging"}
        trainer = EnsembleTrainer(ensemble_config=ensemble_config)

        training_params = {
            "tab_n_d": 32,
            "tab_n_a": 32,
            "tab_n_steps": 5,
            "tab_gamma": 1.5,
            "tab_lambda_sparse": 1e-4,
        }

        optimized_params = trainer._extract_optimized_parameters(training_params)

        # TabNetパラメータが正しく抽出されることを確認
        tab_params = optimized_params["base_models"]["tabnet"]
        assert tab_params["n_d"] == 32
        assert tab_params["n_a"] == 32
        assert tab_params["n_steps"] == 5
        assert tab_params["gamma"] == 1.5
        assert tab_params["lambda_sparse"] == 1e-4

    def test_extract_optimized_parameters_bagging(self):
        """バギングパラメータ抽出のテスト"""
        ensemble_config = {"method": "bagging"}
        trainer = EnsembleTrainer(ensemble_config=ensemble_config)

        training_params = {
            "bagging_n_estimators": 5,
            "bagging_max_samples": 0.8,
            "bagging_max_features": 0.9,
        }

        optimized_params = trainer._extract_optimized_parameters(training_params)

        # バギングパラメータが正しく抽出されることを確認
        bagging_params = optimized_params["bagging"]
        assert bagging_params["n_estimators"] == 5
        assert bagging_params["max_samples"] == 0.8
        assert bagging_params["max_features"] == 0.9

    def test_extract_optimized_parameters_stacking(self):
        """スタッキングパラメータ抽出のテスト"""
        ensemble_config = {"method": "stacking"}
        trainer = EnsembleTrainer(ensemble_config=ensemble_config)

        training_params = {
            "stacking_meta_C": 1.0,
            "stacking_meta_penalty": "l2",
            "stacking_meta_solver": "liblinear",
            "stacking_cv_folds": 5,
        }

        optimized_params = trainer._extract_optimized_parameters(training_params)

        # スタッキングパラメータが正しく抽出されることを確認
        stacking_params = optimized_params["stacking"]
        assert stacking_params["cv_folds"] == 5

        # メタモデルパラメータが正しく抽出されることを確認
        meta_params = stacking_params["meta_model_params"]
        assert meta_params["C"] == 1.0
        assert meta_params["penalty"] == "l2"
        assert meta_params["solver"] == "liblinear"

    def test_extract_optimized_parameters_mixed(self):
        """複数モデルの混合パラメータ抽出のテスト"""
        ensemble_config = {"method": "bagging"}
        trainer = EnsembleTrainer(ensemble_config=ensemble_config)

        training_params = {
            "lgb_num_leaves": 50,
            "lgb_learning_rate": 0.1,
            "xgb_max_depth": 6,
            "xgb_learning_rate": 0.05,
            "rf_n_estimators": 100,
            "cat_iterations": 500,
            "tab_n_d": 32,
            "bagging_n_estimators": 5,
            "random_state": 42,
        }

        optimized_params = trainer._extract_optimized_parameters(training_params)

        # 各モデルのパラメータが正しく分離されることを確認
        assert optimized_params["base_models"]["lightgbm"]["num_leaves"] == 50
        assert optimized_params["base_models"]["xgboost"]["max_depth"] == 6
        assert optimized_params["base_models"]["randomforest"]["n_estimators"] == 100
        assert optimized_params["base_models"]["catboost"]["iterations"] == 500
        assert optimized_params["base_models"]["tabnet"]["n_d"] == 32
        assert optimized_params["bagging"]["n_estimators"] == 5

    def test_extract_optimized_parameters_empty(self):
        """空のパラメータでの抽出テスト"""
        ensemble_config = {"method": "bagging"}
        trainer = EnsembleTrainer(ensemble_config=ensemble_config)

        training_params = {}

        optimized_params = trainer._extract_optimized_parameters(training_params)

        # 空の辞書が返されることを確認
        assert optimized_params["base_models"]["lightgbm"] == {}
        assert optimized_params["base_models"]["xgboost"] == {}
        assert optimized_params["base_models"]["randomforest"] == {}
        assert optimized_params["base_models"]["catboost"] == {}
        assert optimized_params["base_models"]["tabnet"] == {}
        assert optimized_params["bagging"] == {}
        assert optimized_params["stacking"] == {}

    def test_extract_optimized_parameters_unknown_prefix(self):
        """未知のプレフィックスでのパラメータ抽出テスト"""
        ensemble_config = {"method": "bagging"}
        trainer = EnsembleTrainer(ensemble_config=ensemble_config)

        training_params = {
            "unknown_param": "value",
            "another_unknown": 123,
            "lgb_num_leaves": 50,  # 既知のパラメータも含める
        }

        optimized_params = trainer._extract_optimized_parameters(training_params)

        # 既知のパラメータのみが抽出されることを確認
        assert optimized_params["base_models"]["lightgbm"]["num_leaves"] == 50

        # 未知のパラメータは無視されることを確認
        for model_params in optimized_params["base_models"].values():
            assert "unknown_param" not in model_params
            assert "another_unknown" not in model_params

    def test_parameter_extraction_structure(self):
        """パラメータ抽出結果の構造テスト"""
        ensemble_config = {"method": "bagging"}
        trainer = EnsembleTrainer(ensemble_config=ensemble_config)

        training_params = {"lgb_num_leaves": 50}

        optimized_params = trainer._extract_optimized_parameters(training_params)

        # 期待される構造を持つことを確認
        assert "base_models" in optimized_params
        assert "bagging" in optimized_params
        assert "stacking" in optimized_params

        # base_modelsに期待されるモデルが含まれることを確認
        expected_models = ["lightgbm", "xgboost", "randomforest", "catboost", "tabnet"]
        for model in expected_models:
            assert model in optimized_params["base_models"]
            assert isinstance(optimized_params["base_models"][model], dict)


if __name__ == "__main__":
    pytest.main([__file__])

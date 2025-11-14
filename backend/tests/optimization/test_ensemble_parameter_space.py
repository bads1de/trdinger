"""
EnsembleParameterSpaceのテストモジュール

アンサンブル学習用ハイパーパラメータ空間定義の各機能をテストします。
"""

from typing import Any, Dict

import optuna
import pytest

from app.services.optimization.ensemble_parameter_space import EnsembleParameterSpace
from app.services.optimization.optuna_optimizer import ParameterSpace


class TestEnsembleParameterSpace:
    """EnsembleParameterSpaceクラスのテスト"""

    def test_get_lightgbm_parameter_space(self):
        """LightGBMパラメータ空間の取得"""
        space = EnsembleParameterSpace.get_lightgbm_parameter_space()

        # 必須パラメータが含まれる
        assert "lgb_num_leaves" in space
        assert "lgb_learning_rate" in space
        assert "lgb_feature_fraction" in space
        assert "lgb_bagging_fraction" in space
        assert "lgb_min_data_in_leaf" in space
        assert "lgb_max_depth" in space
        assert "lgb_reg_alpha" in space
        assert "lgb_reg_lambda" in space

        # パラメータの型確認
        assert space["lgb_num_leaves"].type == "integer"
        assert space["lgb_learning_rate"].type == "real"
        assert space["lgb_feature_fraction"].type == "real"

        # パラメータ範囲の確認
        assert space["lgb_num_leaves"].low == 10
        assert space["lgb_num_leaves"].high == 100
        assert space["lgb_learning_rate"].low == 0.01
        assert space["lgb_learning_rate"].high == 0.3

    def test_get_xgboost_parameter_space(self):
        """XGBoostパラメータ空間の取得"""
        space = EnsembleParameterSpace.get_xgboost_parameter_space()

        # 必須パラメータが含まれる
        assert "xgb_max_depth" in space
        assert "xgb_learning_rate" in space
        assert "xgb_subsample" in space
        assert "xgb_colsample_bytree" in space
        assert "xgb_min_child_weight" in space
        assert "xgb_reg_alpha" in space
        assert "xgb_reg_lambda" in space
        assert "xgb_gamma" in space

        # パラメータの型確認
        assert space["xgb_max_depth"].type == "integer"
        assert space["xgb_learning_rate"].type == "real"
        assert space["xgb_subsample"].type == "real"

        # パラメータ範囲の確認
        assert space["xgb_max_depth"].low == 3
        assert space["xgb_max_depth"].high == 15
        assert space["xgb_learning_rate"].low == 0.01
        assert space["xgb_learning_rate"].high == 0.3

    def test_get_stacking_parameter_space(self):
        """スタッキングパラメータ空間の取得"""
        space = EnsembleParameterSpace.get_stacking_parameter_space()

        # 必須パラメータが含まれる
        assert "stacking_meta_C" in space
        assert "stacking_meta_penalty" in space
        assert "stacking_meta_solver" in space
        assert "stacking_cv_folds" in space

        # パラメータの型確認
        assert space["stacking_meta_C"].type == "real"
        assert space["stacking_meta_penalty"].type == "categorical"
        assert space["stacking_meta_solver"].type == "categorical"
        assert space["stacking_cv_folds"].type == "integer"

        # カテゴリカルパラメータの選択肢確認
        assert space["stacking_meta_penalty"].categories == ["l1", "l2", "elasticnet"]
        assert space["stacking_meta_solver"].categories == ["liblinear", "saga"]

        # パラメータ範囲の確認
        assert space["stacking_meta_C"].low == 0.01
        assert space["stacking_meta_C"].high == 10.0
        assert space["stacking_cv_folds"].low == 3
        assert space["stacking_cv_folds"].high == 10


class TestGetEnsembleParameterSpace:
    """get_ensemble_parameter_spaceメソッドのテスト"""

    def test_stacking_with_lightgbm_only(self):
        """LightGBMのみでスタッキング"""
        space = EnsembleParameterSpace.get_ensemble_parameter_space(
            ensemble_method="stacking", enabled_models=["lightgbm"]
        )

        # LightGBMパラメータが含まれる
        assert "lgb_num_leaves" in space
        assert "lgb_learning_rate" in space

        # XGBoostパラメータは含まれない
        assert "xgb_max_depth" not in space
        assert "xgb_learning_rate" not in space

        # スタッキングパラメータが含まれる
        assert "stacking_meta_C" in space
        assert "stacking_cv_folds" in space

    def test_stacking_with_xgboost_only(self):
        """XGBoostのみでスタッキング"""
        space = EnsembleParameterSpace.get_ensemble_parameter_space(
            ensemble_method="stacking", enabled_models=["xgboost"]
        )

        # XGBoostパラメータが含まれる
        assert "xgb_max_depth" in space
        assert "xgb_learning_rate" in space

        # LightGBMパラメータは含まれない
        assert "lgb_num_leaves" not in space
        assert "lgb_learning_rate" not in space

        # スタッキングパラメータが含まれる
        assert "stacking_meta_C" in space
        assert "stacking_cv_folds" in space

    def test_stacking_with_both_models(self):
        """LightGBMとXGBoost両方でスタッキング"""
        space = EnsembleParameterSpace.get_ensemble_parameter_space(
            ensemble_method="stacking", enabled_models=["lightgbm", "xgboost"]
        )

        # 両方のモデルのパラメータが含まれる
        assert "lgb_num_leaves" in space
        assert "lgb_learning_rate" in space
        assert "xgb_max_depth" in space
        assert "xgb_learning_rate" in space

        # スタッキングパラメータが含まれる
        assert "stacking_meta_C" in space
        assert "stacking_cv_folds" in space

        # 全パラメータ数の確認
        lightgbm_params = 8  # LightGBMパラメータ数
        xgboost_params = 8  # XGBoostパラメータ数
        stacking_params = 4  # スタッキングパラメータ数
        expected_total = lightgbm_params + xgboost_params + stacking_params
        assert len(space) == expected_total

    def test_stacking_with_empty_models(self):
        """モデルが指定されていない場合"""
        space = EnsembleParameterSpace.get_ensemble_parameter_space(
            ensemble_method="stacking", enabled_models=[]
        )

        # スタッキングパラメータのみ含まれる
        assert "stacking_meta_C" in space
        assert "stacking_cv_folds" in space

        # モデル固有パラメータは含まれない
        assert "lgb_num_leaves" not in space
        assert "xgb_max_depth" not in space


class TestSuggestLightGBMParams:
    """_suggest_lightgbm_paramsメソッドのテスト"""

    @pytest.fixture
    def mock_trial(self):
        """モックOptunaトライアル"""
        # 実際のOptunaトライアルを作成
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        return trial

    def test_suggest_lightgbm_params(self, mock_trial):
        """LightGBMパラメータのサジェスト"""
        params = EnsembleParameterSpace._suggest_lightgbm_params(mock_trial)

        # 全てのパラメータが返される
        assert "lgb_num_leaves" in params
        assert "lgb_learning_rate" in params
        assert "lgb_feature_fraction" in params
        assert "lgb_bagging_fraction" in params
        assert "lgb_min_data_in_leaf" in params
        assert "lgb_max_depth" in params
        assert "lgb_reg_alpha" in params
        assert "lgb_reg_lambda" in params

        # パラメータの型確認
        assert isinstance(params["lgb_num_leaves"], int)
        assert isinstance(params["lgb_learning_rate"], float)
        assert isinstance(params["lgb_max_depth"], int)

        # パラメータ範囲の確認
        assert 10 <= params["lgb_num_leaves"] <= 100
        assert 0.01 <= params["lgb_learning_rate"] <= 0.3
        assert 0.5 <= params["lgb_feature_fraction"] <= 1.0
        assert 3 <= params["lgb_max_depth"] <= 15

    def test_suggest_lightgbm_params_multiple_trials(self):
        """複数回のサジェスト"""
        study = optuna.create_study(direction="maximize")

        params_list = []
        for _ in range(5):
            trial = study.ask()
            params = EnsembleParameterSpace._suggest_lightgbm_params(trial)
            params_list.append(params)

        # 各パラメータが異なる値を持つ（確率的に）
        learning_rates = [p["lgb_learning_rate"] for p in params_list]
        # 5回のうち少なくとも2つは異なる値のはず
        assert len(set(learning_rates)) >= 2


class TestSuggestXGBoostParams:
    """_suggest_xgboost_paramsメソッドのテスト"""

    @pytest.fixture
    def mock_trial(self):
        """モックOptunaトライアル"""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        return trial

    def test_suggest_xgboost_params(self, mock_trial):
        """XGBoostパラメータのサジェスト"""
        params = EnsembleParameterSpace._suggest_xgboost_params(mock_trial)

        # 全てのパラメータが返される
        assert "xgb_max_depth" in params
        assert "xgb_learning_rate" in params
        assert "xgb_subsample" in params
        assert "xgb_colsample_bytree" in params
        assert "xgb_min_child_weight" in params
        assert "xgb_reg_alpha" in params
        assert "xgb_reg_lambda" in params
        assert "xgb_gamma" in params

        # パラメータの型確認
        assert isinstance(params["xgb_max_depth"], int)
        assert isinstance(params["xgb_learning_rate"], float)
        assert isinstance(params["xgb_min_child_weight"], int)

        # パラメータ範囲の確認
        assert 3 <= params["xgb_max_depth"] <= 15
        assert 0.01 <= params["xgb_learning_rate"] <= 0.3
        assert 0.5 <= params["xgb_subsample"] <= 1.0
        assert 0.0 <= params["xgb_gamma"] <= 0.5


class TestSuggestStackingParams:
    """_suggest_stacking_paramsメソッドのテスト"""

    @pytest.fixture
    def mock_trial(self):
        """モックOptunaトライアル"""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        return trial

    def test_suggest_stacking_params(self, mock_trial):
        """スタッキングパラメータのサジェスト"""
        params = EnsembleParameterSpace._suggest_stacking_params(mock_trial)

        # 全てのパラメータが返される
        assert "stacking_meta_C" in params
        assert "stacking_meta_penalty" in params
        assert "stacking_meta_solver" in params
        assert "stacking_cv_folds" in params

        # パラメータの型確認
        assert isinstance(params["stacking_meta_C"], float)
        assert isinstance(params["stacking_meta_penalty"], str)
        assert isinstance(params["stacking_meta_solver"], str)
        assert isinstance(params["stacking_cv_folds"], int)

        # パラメータ範囲の確認
        assert 0.01 <= params["stacking_meta_C"] <= 10.0
        assert params["stacking_meta_penalty"] in ["l1", "l2", "elasticnet"]
        assert params["stacking_meta_solver"] in ["liblinear", "saga"]
        assert 3 <= params["stacking_cv_folds"] <= 10


class TestParameterRanges:
    """パラメータ範囲の検証テスト"""

    def test_learning_rate_range_lightgbm(self):
        """LightGBM学習率の範囲検証"""
        space = EnsembleParameterSpace.get_lightgbm_parameter_space()
        lr_space = space["lgb_learning_rate"]

        assert lr_space.low == 0.01
        assert lr_space.high == 0.3
        assert lr_space.type == "real"

    def test_learning_rate_range_xgboost(self):
        """XGBoost学習率の範囲検証"""
        space = EnsembleParameterSpace.get_xgboost_parameter_space()
        lr_space = space["xgb_learning_rate"]

        assert lr_space.low == 0.01
        assert lr_space.high == 0.3
        assert lr_space.type == "real"

    def test_tree_depth_range_lightgbm(self):
        """LightGBM木の深さの範囲検証"""
        space = EnsembleParameterSpace.get_lightgbm_parameter_space()
        depth_space = space["lgb_max_depth"]

        assert depth_space.low == 3
        assert depth_space.high == 15
        assert depth_space.type == "integer"

    def test_tree_depth_range_xgboost(self):
        """XGBoost木の深さの範囲検証"""
        space = EnsembleParameterSpace.get_xgboost_parameter_space()
        depth_space = space["xgb_max_depth"]

        assert depth_space.low == 3
        assert depth_space.high == 15
        assert depth_space.type == "integer"

    def test_regularization_range_lightgbm(self):
        """LightGBM正則化パラメータの範囲検証"""
        space = EnsembleParameterSpace.get_lightgbm_parameter_space()

        assert space["lgb_reg_alpha"].low == 0.0
        assert space["lgb_reg_alpha"].high == 1.0
        assert space["lgb_reg_lambda"].low == 0.0
        assert space["lgb_reg_lambda"].high == 1.0

    def test_regularization_range_xgboost(self):
        """XGBoost正則化パラメータの範囲検証"""
        space = EnsembleParameterSpace.get_xgboost_parameter_space()

        assert space["xgb_reg_alpha"].low == 0.0
        assert space["xgb_reg_alpha"].high == 1.0
        assert space["xgb_reg_lambda"].low == 0.0
        assert space["xgb_reg_lambda"].high == 1.0


class TestIntegrationWithOptuna:
    """Optunaとの統合テスト"""

    def test_lightgbm_optimization_integration(self):
        """LightGBMパラメータの最適化統合"""
        from app.services.optimization.optuna_optimizer import OptunaOptimizer

        optimizer = OptunaOptimizer()
        space = EnsembleParameterSpace.get_lightgbm_parameter_space()

        def objective(params: Dict[str, Any]) -> float:
            # LightGBMパラメータの簡易評価
            score = (
                params["lgb_learning_rate"] * 5
                + params["lgb_num_leaves"] / 100
                + params["lgb_feature_fraction"]
            )
            return score

        result = optimizer.optimize(objective, space, n_calls=10)

        # 全てのLightGBMパラメータが最適化される
        assert all(
            key in result.best_params
            for key in [
                "lgb_num_leaves",
                "lgb_learning_rate",
                "lgb_feature_fraction",
                "lgb_bagging_fraction",
                "lgb_min_data_in_leaf",
                "lgb_max_depth",
                "lgb_reg_alpha",
                "lgb_reg_lambda",
            ]
        )

    def test_xgboost_optimization_integration(self):
        """XGBoostパラメータの最適化統合"""
        from app.services.optimization.optuna_optimizer import OptunaOptimizer

        optimizer = OptunaOptimizer()
        space = EnsembleParameterSpace.get_xgboost_parameter_space()

        def objective(params: Dict[str, Any]) -> float:
            # XGBoostパラメータの簡易評価
            score = (
                params["xgb_learning_rate"] * 5
                + params["xgb_max_depth"] / 10
                + params["xgb_subsample"]
            )
            return score

        result = optimizer.optimize(objective, space, n_calls=10)

        # 全てのXGBoostパラメータが最適化される
        assert all(
            key in result.best_params
            for key in [
                "xgb_max_depth",
                "xgb_learning_rate",
                "xgb_subsample",
                "xgb_colsample_bytree",
                "xgb_min_child_weight",
                "xgb_reg_alpha",
                "xgb_reg_lambda",
                "xgb_gamma",
            ]
        )

    def test_ensemble_optimization_integration(self):
        """アンサンブル全体の最適化統合"""
        from app.services.optimization.optuna_optimizer import OptunaOptimizer

        optimizer = OptunaOptimizer()
        space = EnsembleParameterSpace.get_ensemble_parameter_space(
            ensemble_method="stacking", enabled_models=["lightgbm", "xgboost"]
        )

        def objective(params: Dict[str, Any]) -> float:
            # 全パラメータを考慮した評価
            lgb_score = params["lgb_learning_rate"] * 3 + params["lgb_num_leaves"] / 50
            xgb_score = params["xgb_learning_rate"] * 3 + params["xgb_max_depth"] / 10
            stacking_score = params["stacking_meta_C"] / 10
            return lgb_score + xgb_score + stacking_score

        result = optimizer.optimize(objective, space, n_calls=15)

        # LightGBM、XGBoost、スタッキングの全パラメータが含まれる
        assert "lgb_num_leaves" in result.best_params
        assert "xgb_max_depth" in result.best_params
        assert "stacking_meta_C" in result.best_params

    def test_suggest_methods_with_real_trial(self):
        """実際のOptunaトライアルでサジェストメソッドをテスト"""
        study = optuna.create_study(direction="maximize")

        def objective(trial: optuna.Trial) -> float:
            # 各サジェストメソッドを使用
            lgb_params = EnsembleParameterSpace._suggest_lightgbm_params(trial)
            xgb_params = EnsembleParameterSpace._suggest_xgboost_params(trial)
            stacking_params = EnsembleParameterSpace._suggest_stacking_params(trial)

            # 全パラメータが適切に生成される
            assert len(lgb_params) == 8
            assert len(xgb_params) == 8
            assert len(stacking_params) == 4

            # ダミースコア
            return 1.0

        study.optimize(objective, n_trials=5)

        # 最適化が成功
        assert len(study.trials) == 5
        assert study.best_value is not None


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_unknown_ensemble_method(self):
        """未知のアンサンブル手法"""
        # 未知の手法でもエラーにならない（該当パラメータがないだけ）
        space = EnsembleParameterSpace.get_ensemble_parameter_space(
            ensemble_method="unknown_method", enabled_models=["lightgbm"]
        )

        # LightGBMパラメータは含まれる
        assert "lgb_num_leaves" in space

        # スタッキングパラメータは含まれない（unknown_methodなので）
        assert "stacking_meta_C" not in space

    def test_duplicate_model_in_enabled_list(self):
        """重複したモデルがリストにある場合"""
        space = EnsembleParameterSpace.get_ensemble_parameter_space(
            ensemble_method="stacking",
            enabled_models=["lightgbm", "lightgbm", "xgboost"],
        )

        # 重複していても正常に処理される
        assert "lgb_num_leaves" in space
        assert "xgb_max_depth" in space

        # パラメータ数は正常（重複カウントされない）
        lgb_params = [k for k in space.keys() if k.startswith("lgb_")]
        assert len(lgb_params) == 8  # 重複分は追加されない

    def test_case_sensitive_model_names(self):
        """大文字小文字を区別するモデル名"""
        # 小文字で指定
        space_lower = EnsembleParameterSpace.get_ensemble_parameter_space(
            ensemble_method="stacking", enabled_models=["lightgbm"]
        )

        # 大文字で指定（マッチしない）
        space_upper = EnsembleParameterSpace.get_ensemble_parameter_space(
            ensemble_method="stacking", enabled_models=["LIGHTGBM"]
        )

        # 小文字のみマッチ
        assert "lgb_num_leaves" in space_lower
        assert "lgb_num_leaves" not in space_upper

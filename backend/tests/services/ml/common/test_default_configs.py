"""
デフォルト設定のテスト

default_configs.py で定義されているデフォルト設定のテストです。
"""

from app.services.ml.common.default_configs import (
    DefaultTrainingConfigs,
    get_default_ensemble_config,
    get_default_single_model_config,
)


class TestDefaultTrainingConfigs:
    """DefaultTrainingConfigsクラスのテスト"""

    def test_get_default_ensemble_config(self):
        """デフォルトのアンサンブル設定が正しく取得できる"""
        config = DefaultTrainingConfigs.get_default_ensemble_config()

        assert config is not None
        assert isinstance(config, dict)
        assert config["enabled"] is True
        assert config["method"] == "stacking"
        assert "stacking_params" in config

        stacking_params = config["stacking_params"]
        assert "base_models" in stacking_params
        assert "lightgbm" in stacking_params["base_models"]
        assert "xgboost" in stacking_params["base_models"]
        assert stacking_params["meta_model"] == "lightgbm"
        assert stacking_params["cv_folds"] == 5
        assert stacking_params["use_probas"] is True
        assert stacking_params["random_state"] == 42

    def test_get_default_single_model_config(self):
        """デフォルトの単一モデル設定が正しく取得できる"""
        config = DefaultTrainingConfigs.get_default_single_model_config()

        assert config is not None
        assert isinstance(config, dict)
        assert config["model_type"] == "lightgbm"

    def test_configs_are_independent(self):
        """設定オブジェクトが独立している（変更が他に影響しない）"""
        config1 = DefaultTrainingConfigs.get_default_ensemble_config()
        config2 = DefaultTrainingConfigs.get_default_ensemble_config()

        # 設定を変更
        config1["enabled"] = False

        # 別のインスタンスには影響しない
        assert config2["enabled"] is True


class TestModuleLevelFunctions:
    """モジュールレベル関数のテスト"""

    def test_get_default_ensemble_config_function(self):
        """get_default_ensemble_config関数が正しく動作する"""
        config = get_default_ensemble_config()

        assert config is not None
        assert isinstance(config, dict)
        assert config["enabled"] is True
        assert config["method"] == "stacking"

    def test_get_default_single_model_config_function(self):
        """get_default_single_model_config関数が正しく動作する"""
        config = get_default_single_model_config()

        assert config is not None
        assert isinstance(config, dict)
        assert config["model_type"] == "lightgbm"

    def test_function_matches_class_method(self):
        """モジュールレベル関数がクラスメソッドと同じ結果を返す"""
        class_config = DefaultTrainingConfigs.get_default_ensemble_config()
        function_config = get_default_ensemble_config()

        assert class_config == function_config

        class_single = DefaultTrainingConfigs.get_default_single_model_config()
        function_single = get_default_single_model_config()

        assert class_single == function_single





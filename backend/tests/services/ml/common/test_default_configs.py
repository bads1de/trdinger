"""
デフォルト設定のテスト
"""

from app.services.ml.common.default_configs import (
    get_default_ensemble_config,
    get_default_single_model_config,
)


class TestDefaultConfigs:
    """デフォルト設定のテスト"""

    def test_get_default_ensemble_config(self):
        """デフォルトのアンサンブル設定が正しく取得できる"""
        config = get_default_ensemble_config()

        assert isinstance(config, dict)
        assert config["enabled"] is True
        assert config["method"] == "stacking"
        
        p = config["stacking_params"]
        assert "lightgbm" in p["base_models"]
        assert "xgboost" in p["base_models"]
        assert p["meta_model"] == "lightgbm"
        assert p["cv_folds"] == 5
        assert p["use_probas"] is True
        assert p["random_state"] == 42

    def test_get_default_single_model_config(self):
        """デフォルトの単一モデル設定が正しく取得できる"""
        config = get_default_single_model_config()
        assert isinstance(config, dict)
        assert config["model_type"] == "lightgbm"

    def test_configs_are_independent(self):
        """設定オブジェクトが独立している（変更が他に影響しない）"""
        c1 = get_default_ensemble_config()
        c2 = get_default_ensemble_config()
        c1["enabled"] = False
        assert c2["enabled"] is True





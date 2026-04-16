"""
config.py のテスト

app/services/ml/common/config.py のテストモジュール
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.ml.common.config import (
    MLConfigManager,
    get_default_ensemble_config,
    get_default_single_model_config,
    ml_config_manager,
)


class TestGetDefaultEnsembleConfig:
    """get_default_ensemble_config 関数のテスト"""

    def test_get_default_ensemble_config(self):
        """デフォルトアンサンブル設定の取得"""
        config = get_default_ensemble_config()

        assert isinstance(config, dict)
        assert config["enabled"] is True
        assert config["method"] == "stacking"
        assert "stacking_params" in config
        assert config["stacking_params"]["base_models"] == ["lightgbm", "xgboost"]
        assert config["stacking_params"]["meta_model"] == "lightgbm"
        assert config["stacking_params"]["cv_folds"] == 5


class TestGetDefaultSingleModelConfig:
    """get_default_single_model_config 関数のテスト"""

    def test_get_default_single_model_config(self):
        """デフォルト単一モデル設定の取得"""
        config = get_default_single_model_config()

        assert isinstance(config, dict)
        assert config["model_type"] == "lightgbm"


class TestMLConfigManager:
    """MLConfigManager クラスのテスト"""

    def test_initialization_default_path(self):
        """デフォルトパスで初期化"""
        manager = MLConfigManager()
        assert manager.config is not None
        assert manager.config_file_path == Path("config/ml_config.json")

    def test_initialization_custom_path(self):
        """カスタムパスで初期化"""
        manager = MLConfigManager("custom/config.json")
        assert manager.config_file_path == Path("custom/config.json")

    def test_config_property(self):
        """configプロパティの取得"""
        manager = MLConfigManager()
        config = manager.config
        assert config is not None

    def test_config_setter(self):
        """configプロパティの設定"""
        manager = MLConfigManager()
        from app.services.ml.common.ml_config import MLConfig

        new_config = MLConfig()
        manager.config = new_config
        assert manager.config == new_config

    def test_get_config_dict(self):
        """設定を辞書形式で取得"""
        manager = MLConfigManager()
        config_dict = manager.get_config_dict()

        assert isinstance(config_dict, dict)

    def test_save_config_success(self):
        """設定の保存（成功）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "ml_config.json"
            manager = MLConfigManager(str(config_path))

            result = manager.save_config()

            assert result is True
            assert config_path.exists()

            # 保存された内容を確認
            with open(config_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                assert "_metadata" in saved_data

    def test_save_config_creates_directory(self):
        """設定の保存（ディレクトリ作成）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "subdir" / "ml_config.json"
            manager = MLConfigManager(str(config_path))

            result = manager.save_config()

            assert result is True
            assert config_path.exists()

    def test_load_config_success(self):
        """設定の読み込み（成功）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "ml_config.json"
            manager = MLConfigManager(str(config_path))

            # まず設定を保存
            manager.save_config()

            # 新しいマネージャーで読み込み
            new_manager = MLConfigManager(str(config_path))
            result = new_manager.load_config()

            assert result is True

    def test_load_config_file_not_exists(self):
        """設定の読み込み（ファイルが存在しない）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nonexistent.json"
            manager = MLConfigManager(str(config_path))

            result = manager.load_config()

            assert result is False

    def test_load_config_invalid_json(self):
        """設定の読み込み（無効なJSON）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid.json"
            config_path.write_text("invalid json content", encoding="utf-8")

            manager = MLConfigManager(str(config_path))
            result = manager.load_config()

            assert result is False

    def test_update_config_success(self):
        """設定の更新（成功）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "ml_config.json"
            manager = MLConfigManager(str(config_path))

            updates = {"model_type": "xgboost"}
            result = manager.update_config(updates)

            assert result is True
            assert config_path.exists()

    def test_update_config_nested(self):
        """設定の更新（ネストされた辞書）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "ml_config.json"
            manager = MLConfigManager(str(config_path))

            updates = {"ensemble_params": {"method": "blending"}}
            result = manager.update_config(updates)

            assert result is True

    def test_update_config_save_failure(self):
        """設定の更新（保存失敗）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "readonly" / "ml_config.json"
            manager = MLConfigManager(str(config_path))

            # 読み取り専用ディレクトリを作成して保存失敗をシミュレート
            # Windowsでは読み取り専利の作成が難しいため、モックを使用
            with patch.object(manager, "save_config", return_value=False):
                updates = {"model_type": "xgboost"}
                result = manager.update_config(updates)

                assert result is False

    def test_reset_config_success(self):
        """設定のリセット（成功）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "ml_config.json"
            manager = MLConfigManager(str(config_path))

            # 設定を変更
            from app.services.ml.common.ml_config import MLConfig

            manager.config = MLConfig(model_type="xgboost")

            # リセット
            result = manager.reset_config()

            assert result is True

    def test_reset_config_save_failure(self):
        """設定のリセット（保存失敗）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "ml_config.json"
            manager = MLConfigManager(str(config_path))

            with patch.object(manager, "save_config", return_value=False):
                result = manager.reset_config()

                assert result is False

    def test_merge_config_updates_simple(self):
        """設定更新のマージ（単純な値）"""
        manager = MLConfigManager()
        current = {"model_type": "lightgbm", "n_estimators": 100}
        updates = {"n_estimators": 200}

        result = manager._merge_config_updates(current, updates)

        assert result["model_type"] == "lightgbm"
        assert result["n_estimators"] == 200

    def test_merge_config_updates_nested(self):
        """設定更新のマージ（ネストされた辞書）"""
        manager = MLConfigManager()
        current = {"ensemble_params": {"method": "stacking", "cv_folds": 5}}
        updates = {"ensemble_params": {"cv_folds": 10}}

        result = manager._merge_config_updates(current, updates)

        assert result["ensemble_params"]["method"] == "stacking"
        assert result["ensemble_params"]["cv_folds"] == 10

    def test_merge_config_updates_new_key(self):
        """設定更新のマージ（新しいキー）"""
        manager = MLConfigManager()
        current = {"model_type": "lightgbm"}
        updates = {"new_key": "new_value"}

        result = manager._merge_config_updates(current, updates)

        assert result["new_key"] == "new_value"


class TestGlobalMLConfigManager:
    """グローバルml_config_managerのテスト"""

    def test_global_ml_config_manager_exists(self):
        """グローバルインスタンスの存在確認"""
        assert ml_config_manager is not None
        assert isinstance(ml_config_manager, MLConfigManager)

    def test_global_ml_config_manager_config(self):
        """グローバルインスタンスの設定取得"""
        config = ml_config_manager.config
        assert config is not None

    def test_global_ml_config_manager_get_config_dict(self):
        """グローバルインスタンスの辞書取得"""
        config_dict = ml_config_manager.get_config_dict()
        assert isinstance(config_dict, dict)

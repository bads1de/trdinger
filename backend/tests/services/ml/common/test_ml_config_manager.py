import pytest
from unittest.mock import patch
from app.services.ml.common.config import MLConfigManager
from app.config.unified_config import unified_config


class TestMLConfigManager:
    @pytest.fixture
    def config_file(self, tmp_path):
        return tmp_path / "ml_config.json"

    @pytest.fixture
    def manager(self, config_file):
        # グローバルの unified_config.ml をバックアップ
        original_ml = unified_config.ml
        m = MLConfigManager(config_file_path=str(config_file))
        yield m
        # テスト後に元に戻す
        unified_config.ml = original_ml

    def test_save_and_load_config(self, manager, config_file):
        """設定の保存と読み込み"""
        # 1. 保存
        success = manager.save_config()
        assert success is True
        assert config_file.exists()

        # 2. 読み込み
        success_load = manager.load_config()
        assert success_load is True

    def test_update_config(self, manager):
        """設定の更新"""
        # 学習設定の推定器数を変えてみる
        update = {"training": {"lgb_n_estimators": 999}}
        success = manager.update_config(update)

        assert success is True
        assert unified_config.ml.training.lgb_n_estimators == 999

    def test_reset_config(self, manager):
        """設定のリセット"""
        # 一旦変更
        manager.update_config({"training": {"lgb_n_estimators": 999}})
        assert unified_config.ml.training.lgb_n_estimators == 999

        # リセット
        success = manager.reset_config()
        assert success is True
        # デフォルト値に戻っていることを確認
        assert unified_config.ml.training.lgb_n_estimators != 999

    def test_load_config_not_found(self, tmp_path):
        """設定ファイルがない場合の読み込み"""
        non_existent = tmp_path / "ghost.json"
        manager = MLConfigManager(config_file_path=str(non_existent))
        assert manager.load_config() is False

    def test_save_config_error_handling(self, manager):
        """保存時のエラーハンドリング"""
        # 書き込み禁止のディレクトリなどをシミュレート
        with patch("builtins.open", side_effect=PermissionError("Forbidden")):
            success = manager.save_config()
            assert success is False

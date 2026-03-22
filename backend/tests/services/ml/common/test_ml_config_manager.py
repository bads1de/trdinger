import json
from unittest.mock import mock_open, patch

import pytest

from app.config.unified_config import unified_config
from app.services.ml.common.config import MLConfigManager


class TestMLConfigManager:
    """MLConfigManager のテスト (ファイル操作を全てモックしてPermissionErrorを防止)"""

    @pytest.fixture
    def manager(self):
        # Path.exists をモックして初期化時の load_config を防ぐ
        with patch("app.services.ml.common.config.Path.exists", return_value=False):
            m = MLConfigManager(config_file_path="/mock/ml_config.json")
            return m

    def test_save_config_success(self, manager):
        """設定保存の正常系テスト"""
        original_ml = unified_config.ml
        try:
            # 書き込み先親ディレクトリの存在チェックもモック
            with patch("app.services.ml.common.config.open", mock_open()) as m_open:
                with patch(
                    "app.services.ml.common.config.Path.exists", return_value=True
                ):
                    success = manager.save_config()
                    assert success is True
                    # ファイルオープンが呼ばれたことを確認
                    m_open.assert_called_once_with(
                        manager.config_file_path, "w", encoding="utf-8"
                    )
        finally:
            unified_config.ml = original_ml

    def test_load_config_success(self, manager):
        """設定読み込みの正常系テスト"""
        original_ml = unified_config.ml
        try:
            # 特定の値を設定したJSONデータをシミュレート
            mock_data = {"training": {"lgb_n_estimators": 1234}}
            m_open = mock_open(read_data=json.dumps(mock_data))

            with patch("app.services.ml.common.config.open", m_open):
                with patch(
                    "app.services.ml.common.config.Path.exists", return_value=True
                ):
                    success = manager.load_config()
                    assert success is True
                    # unified_config.ml に反映されているか確認
                    assert unified_config.ml.training.lgb_n_estimators == 1234
        finally:
            unified_config.ml = original_ml

    def test_update_config_success(self, manager):
        """設定更新のテスト"""
        original_ml = unified_config.ml
        try:
            # save_config が成功するように内部メソッドをモック
            with patch.object(manager, "save_config", return_value=True):
                update = {"training": {"lgb_n_estimators": 9999}}
                success = manager.update_config(update)
                assert success is True
                assert unified_config.ml.training.lgb_n_estimators == 9999
        finally:
            unified_config.ml = original_ml

    def test_reset_config(self, manager):
        """設定リセットのテスト"""
        original_ml = unified_config.ml
        try:
            # 一時的に値を変更
            unified_config.ml.training.lgb_n_estimators = 7777

            with patch.object(manager, "save_config", return_value=True):
                success = manager.reset_config()
                assert success is True
                # デフォルト値（一般的に 7777 ではない）に戻っていることを確認
                assert unified_config.ml.training.lgb_n_estimators != 7777
        finally:
            unified_config.ml = original_ml

    def test_load_config_not_found(self, manager):
        """ファイルが存在しない場合の読み込みテスト"""
        with patch("app.services.ml.common.config.Path.exists", return_value=False):
            success = manager.load_config()
            assert success is False

    def test_save_config_error_handling(self, manager):
        """保存時の各種エラー（PermissionError等）のハンドリング"""
        # open時にPermissionErrorを発生させる
        with patch(
            "app.services.ml.common.config.open",
            side_effect=PermissionError("Forbidden"),
        ):
            with patch("app.services.ml.common.config.Path.exists", return_value=True):
                success = manager.save_config()
                assert success is False

    def test_mkdir_fail_handling(self, manager):
        """ディレクトリ作成失敗時のハンドリング"""
        with patch(
            "app.services.ml.common.config.Path.exists", side_effect=[False, False]
        ):
            with patch(
                "app.services.ml.common.config.Path.mkdir",
                side_effect=OSError("Disk Full"),
            ):
                # ディレクトリ作成に失敗しても例外で落ちずに False を返すことを確認
                # 実際には open で落ちるが save_config 全体で try-except されている
                success = manager.save_config()
                assert success is False

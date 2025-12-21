import os
import glob
import pytest
import shutil
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from app.services.ml.models.model_manager import ModelManager


class TestModelManagerCleanup:
    @pytest.fixture
    def manager(self, tmp_path):
        manager = ModelManager()
        manager.config = MagicMock()
        manager.config.model_save_path = str(tmp_path)
        manager.config.model_file_extension = ".joblib"
        manager.config.max_model_versions = 2  # 最新2つを残す
        manager.config.model_retention_days = 7
        return manager

    def test_cleanup_with_sidecar(self, manager, tmp_path):
        """サイドカーファイル削除のテスト"""
        # _get_model_search_paths をモックして tmp_path のみを返すようにする
        with patch.object(
            manager, "_get_model_search_paths", return_value=[str(tmp_path)]
        ):
            # ファイルを作成
            for i in range(5):
                name = f"test_model_{i}.joblib"
                path = os.path.join(tmp_path, name)
                sidecar_path = path.replace(".joblib", ".meta.json")

                # ファイル作成
                with open(path, "w") as f:
                    f.write("model")
                with open(sidecar_path, "w") as f:
                    f.write("{}")

                # タイムスタンプ操作（10日前）
                # Windowsでも安全なように、現在時刻から引く
                old_time = (datetime.now() - timedelta(days=10)).timestamp()
                os.utime(path, (old_time, old_time))
                os.utime(sidecar_path, (old_time, old_time))

            # クリーンアップ実行
            manager.cleanup_expired_models()

            # デバッグ: 残っているファイルを表示
            left_models = glob.glob(os.path.join(tmp_path, "*.joblib"))
            if len(left_models) > 0:
                print(f"DEBUG: Left models: {left_models}")
                for p in left_models:
                    print(f"  {p}: mtime={os.path.getmtime(p)}")

            # 全て削除されていることを確認
            assert len(glob.glob(os.path.join(tmp_path, "*.joblib"))) == 0
            assert len(glob.glob(os.path.join(tmp_path, "*.meta.json"))) == 0

    def test_old_version_cleanup_with_sidecar(self, manager, tmp_path):
        """バージョン管理による古いモデルとサイドカーの削除テスト"""
        model_name = "versioned_model"

        # タイムスタンプの基準を現在にする
        base_time = datetime.now().timestamp()

        # unified_config のモックは不要だが、念のため
        # from app.config.unified_config import unified_config

        with patch.object(
            manager, "_get_model_search_paths", return_value=[str(tmp_path)]
        ):
            files = []
            for i in range(5):
                # i が大きいほど新しいとする (100秒間隔)
                # ファイル生成順序とmtime順序を一致させるため、明示的にutimeする
                timestamp = base_time - (5 - i) * 100

                name = f"{model_name}_{i:02d}.joblib"
                path = os.path.join(tmp_path, name)
                sidecar_path = path.replace(".joblib", ".meta.json")

                with open(path, "w") as f:
                    f.write("model")
                with open(sidecar_path, "w") as f:
                    f.write("{}")

                os.utime(path, (timestamp, timestamp))
                os.utime(sidecar_path, (timestamp, timestamp))

            # max_model_versions = 2 なので、古い3つ (00, 01, 02) が削除されるはず
            manager._cleanup_old_models(model_name)

            remaining_models = glob.glob(
                os.path.join(tmp_path, f"{model_name}_*.joblib")
            )
            remaining_sidecars = glob.glob(
                os.path.join(tmp_path, f"{model_name}_*.meta.json")
            )

            remaining_models.sort()

            assert len(remaining_models) == 2
            assert len(remaining_sidecars) == 2

            # 残っているのは新しいもの (03, 04)
            rem_names = [os.path.basename(p) for p in remaining_models]
            assert f"{model_name}_03.joblib" in rem_names
            assert f"{model_name}_04.joblib" in rem_names

"""
ModelManager のユニットテスト

サイドカーJSONによるメタデータ管理のテスト
"""

import json
import os
import pytest
import tempfile

from app.services.ml.model_manager import ModelManager


@pytest.fixture
def temp_model_dir():
    """一時的なモデルディレクトリを作成"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def model_manager_with_temp_dir(temp_model_dir):
    """一時ディレクトリを使用するModelManagerを作成"""
    manager = ModelManager()
    # モデル保存パスを一時ディレクトリに変更
    manager.config.model_save_path = temp_model_dir
    return manager


class TestSidecarMetadata:
    """サイドカーJSONメタデータのテスト"""

    def test_save_model_creates_sidecar_json(
        self, model_manager_with_temp_dir, temp_model_dir
    ):
        """save_modelがサイドカーJSONファイルを作成することを確認"""
        manager = model_manager_with_temp_dir

        # ダミーモデルを保存
        dummy_model = {"model_type": "test"}
        metadata = {"accuracy": 0.95, "precision": 0.90}

        model_path = manager.save_model(
            model=dummy_model,
            model_name="test_model",
            metadata=metadata,
        )

        assert model_path is not None

        # サイドカーJSONファイルが存在することを確認
        sidecar_path = manager._get_sidecar_path(model_path)
        assert os.path.exists(
            sidecar_path
        ), f"サイドカーJSONファイルが作成されていません: {sidecar_path}"

        # サイドカーJSONの内容を確認
        with open(sidecar_path, "r", encoding="utf-8") as f:
            sidecar_data = json.load(f)

        assert "accuracy" in sidecar_data or "accuracy" in sidecar_data.get(
            "metadata", {}
        )

    def test_load_metadata_only_does_not_load_model(
        self, model_manager_with_temp_dir, temp_model_dir
    ):
        """load_metadata_onlyがモデル本体をロードしないことを確認"""
        manager = model_manager_with_temp_dir

        # ダミーモデルを保存
        dummy_model = {"model_type": "test", "weights": [1, 2, 3]}
        metadata = {"accuracy": 0.95}

        model_path = manager.save_model(
            model=dummy_model,
            model_name="test_model",
            metadata=metadata,
        )

        # メタデータのみを読み込み
        loaded_metadata = manager.load_metadata_only(model_path)

        assert loaded_metadata is not None
        assert "model" not in loaded_metadata or loaded_metadata.get("model") is None

    def test_load_metadata_only_fallback_to_joblib(
        self, model_manager_with_temp_dir, temp_model_dir
    ):
        """サイドカーJSONがない場合、fallbackとしてjoblibからメタデータを読み込むことを確認"""
        manager = model_manager_with_temp_dir

        # モデルを保存
        dummy_model = {"model_type": "test"}
        metadata = {"accuracy": 0.95}

        model_path = manager.save_model(
            model=dummy_model,
            model_name="test_model",
            metadata=metadata,
        )

        # サイドカーJSONを削除
        sidecar_path = manager._get_sidecar_path(model_path)
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)

        # メタデータ読み込み（フォールバック）
        loaded_metadata = manager.load_metadata_only(model_path)

        # フォールバック時でもメタデータが取得できることを確認
        assert loaded_metadata is not None


class TestModelManagerSave:
    """モデル保存のテスト"""

    def test_save_and_load_model(self, model_manager_with_temp_dir):
        """モデルの保存と読み込みが正常に動作することを確認"""
        manager = model_manager_with_temp_dir

        dummy_model = {"type": "test_model"}
        metadata = {"version": "1.0"}

        model_path = manager.save_model(
            model=dummy_model,
            model_name="test",
            metadata=metadata,
        )

        assert model_path is not None
        assert os.path.exists(model_path)

        # 読み込み
        loaded = manager.load_model(model_path)
        assert loaded is not None
        assert loaded["model"] == dummy_model

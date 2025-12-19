import pytest
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from app.services.ml.models.model_manager import ModelManager, model_manager
from app.services.ml.common.exceptions import MLModelError

class TestModelManager:
    @pytest.fixture
    def manager(self, tmp_path):
        """テスト用のModelManagerインスタンス（一時ディレクトリ使用）"""
        m = ModelManager()
        # 設定を一時的に書き換え
        m.config = MagicMock()
        m.config.model_save_path = str(tmp_path / "models")
        m.config.model_file_extension = ".joblib"
        m.config.max_model_versions = 3
        m.config.model_retention_days = 30
        os.makedirs(m.config.model_save_path, exist_ok=True)
        return m

    def test_save_model_success(self, manager):
        """モデル保存の成功テスト"""
        # シリアライズ可能なダミーモデルを使用
        mock_model = {"coef": [1, 2, 3]}
        model_name = "test_model"
        metadata = {"accuracy": 0.9}
        
        # AlgorithmRegistryのパッチ
        with patch('app.services.ml.models.model_manager.ModelManager._extract_algorithm_name', return_value="mock_algo"):
            path = manager.save_model(mock_model, model_name, metadata=metadata)
            
            assert path is not None
            assert os.path.exists(path)
            assert "mock_algo" in path
            # サイドカーも存在するか
            sidecar = path.replace(".joblib", ".meta.json")
            assert os.path.exists(sidecar)

    def test_save_model_none_fail(self, manager):
        """Noneモデル保存時のエラー（safe_ml_operationによりNoneが返る）"""
        path = manager.save_model(None, "test")
        assert path is None

    def test_load_model_success(self, manager, tmp_path):
        """モデル読み込みの成功テスト"""
        model_file = tmp_path / "models" / "test.joblib"
        model_data = {
            "model": {"dummy": True},
            "scaler": None,
            "feature_columns": ["f1"],
            "metadata": {"type": "test"}
        }
        joblib.dump(model_data, str(model_file))
        
        result = manager.load_model(str(model_file))
        assert result["metadata"]["type"] == "test"
        assert result["feature_columns"] == ["f1"]

    def test_load_model_old_format(self, manager, tmp_path):
        """旧形式（直接オブジェクト）の読み込み"""
        model_file = tmp_path / "models" / "old.joblib"
        mock_model = [1, 2, 3] # シリアライズ可能なリスト
        joblib.dump(mock_model, str(model_file))
        
        result = manager.load_model(str(model_file))
        assert result["model"] == [1, 2, 3]
        assert result["metadata"] == {}

    def test_load_model_not_found(self, manager):
        """ファイル不在時の読み込みエラー（safe_ml_operationによりNoneが返る）"""
        result = manager.load_model("/non/existent/path")
        assert result is None

    def test_list_models(self, manager, tmp_path):
        """モデル一覧取得のテスト"""
        model_dir = tmp_path / "models"
        m1 = model_dir / "m1.joblib"
        m2 = model_dir / "m2.joblib"
        m1.touch()
        m2.touch()
        
        # search_pathsをパッチ。unified_config.ml全体をモックにする
        with patch('app.services.ml.models.model_manager.unified_config.ml') as mock_ml:
            mock_ml.get_model_search_paths.return_value = [str(model_dir)]
            models = manager.list_models()
            assert len(models) == 2
            assert any(m["name"] == "m1.joblib" for m in models)

    def test_cleanup_old_models(self, manager, tmp_path):
        """バージョン制限によるクリーンアップのテスト"""
        model_dir = tmp_path / "models"
        # max_model_versions = 3 に対して 5つ作成
        for i in range(5):
            f = model_dir / f"test_{i:02d}.joblib"
            f.touch()
            # 更新時刻をずらす
            mtime = os.path.getmtime(str(f)) - (10 - i) * 60
            os.utime(str(f), (mtime, mtime))
            
        manager._cleanup_old_models("test")
        
        remaining = os.listdir(str(model_dir))
        # .meta.jsonなどは除外してjoblibのみ数える
        joblib_files = [f for f in remaining if f.endswith(".joblib")]
        assert len(joblib_files) == 3

    def test_extract_model_performance_metrics(self, manager):
        """メトリクス抽出のテスト"""
        metadata = {
            "accuracy": 0.85,
            "classification_report": {
                "macro avg": {"precision": 0.8, "recall": 0.7, "f1-score": 0.75}
            }
        }
        # precisionが0の場合のフォールバック確認
        metrics = manager.extract_model_performance_metrics("/path", metadata=metadata)
        assert metrics["accuracy"] == 0.85
        assert metrics["precision"] == 0.8
        assert metrics["f1_score"] == 0.75

    def test_sidecar_metadata(self, manager, tmp_path):
        """サイドカーJSONの保存と読み込み"""
        model_path = str(tmp_path / "models" / "test.joblib")
        model_data = {
            "metadata": {"acc": 0.9},
            "feature_columns": ["f1"],
            "model_name": "test"
        }
        
        # 保存
        manager._save_metadata_sidecar(model_path, model_data)
        sidecar_path = model_path.replace(".joblib", ".meta.json")
        assert os.path.exists(sidecar_path)
        
        # 読み込み
        loaded_meta = manager.load_metadata_only(model_path)
        assert loaded_meta["metadata"]["acc"] == 0.9
        assert loaded_meta["feature_columns"] == ["f1"]

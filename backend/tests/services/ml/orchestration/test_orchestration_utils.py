import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from app.services.ml.orchestration.orchestration_utils import (
    load_model_metadata_safely,
    get_latest_model_with_info,
    get_model_info_with_defaults
)

class TestOrchestrationUtils:
    def test_load_model_metadata_safely_success(self):
        """メタデータの安全な読み込み成功"""
        with patch('app.services.ml.orchestration.orchestration_utils.model_manager') as mock_mgr:
            mock_mgr.load_metadata_only.return_value = {"metadata": {"model_type": "lgb"}}
            res = load_model_metadata_safely("path/to/model.pkl")
            assert res["metadata"]["model_type"] == "lgb"

    def test_load_model_metadata_safely_fail(self):
        """メタデータの読み込み失敗時"""
        with patch('app.services.ml.orchestration.orchestration_utils.model_manager') as mock_mgr:
            mock_mgr.load_metadata_only.return_value = None
            assert load_model_metadata_safely("invalid") is None

    def test_get_latest_model_with_info_success(self):
        """最新モデル情報の取得成功"""
        with patch('app.services.ml.orchestration.orchestration_utils.model_manager') as mock_mgr:
            with patch('os.path.exists', return_value=True):
                with patch('os.path.getsize', return_value=1024*1024):
                    with patch('os.path.getmtime', return_value=datetime.now().timestamp()):
                        mock_mgr.get_latest_model.return_value = "latest.pkl"
                        mock_mgr.load_metadata_only.return_value = {"metadata": {"model_type": "lgb"}}
                        mock_mgr.extract_model_performance_metrics.return_value = {"accuracy": 0.9}
                        
                        info = get_latest_model_with_info()
                        assert info["path"] == "latest.pkl"
                        assert info["metrics"]["accuracy"] == 0.9
                        assert "file_info" in info

    def test_get_model_info_with_defaults_empty(self):
        """モデル情報がない場合のデフォルト値"""
        res = get_model_info_with_defaults(None)
        assert res["model_type"] == "No Model"
        assert res["training_samples"] == 0
        assert res["last_updated"] == "未学習"

    def test_get_model_info_with_defaults_filled(self):
        """モデル情報がある場合のパース"""
        info = {
            "metadata": {"model_type": "xgboost", "training_samples": 100},
            "metrics": {"accuracy": 0.85},
            "file_info": {"size_mb": 1.5, "modified_at": datetime(2023, 1, 1)}
        }
        res = get_model_info_with_defaults(info)
        assert res["model_type"] == "xgboost"
        assert res["accuracy"] == 0.85
        assert res["file_size_mb"] == 1.5

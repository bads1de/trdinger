import os
import pytest
import joblib
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from app.services.ml.models.model_manager import ModelManager
from app.services.ml.common.exceptions import MLModelError


class DummyModel:
    def __init__(self):
        self.coef_ = [1]

    def predict(self, X):
        return [0]


class TestModelManagerUnit:
    @pytest.fixture
    def manager(self, tmp_path):
        manager = ModelManager()
        manager.config = MagicMock()
        manager.config.model_save_path = str(tmp_path)
        manager.config.model_file_extension = ".joblib"
        manager.config.max_model_versions = 2
        manager.config.model_retention_days = 7
        os.makedirs(str(tmp_path), exist_ok=True)
        # CI環境等でデフォルトの検索パスが存在しない場合でもテストが動作するように、
        # 検索パスを一時ディレクトリに強制する
        manager._get_model_search_paths = MagicMock(return_value=[str(tmp_path)])
        return manager

    def test_save_load_basic(self, manager):
        path = manager.save_model(DummyModel(), "test", metadata={"acc": 0.8})
        assert path and os.path.exists(path)
        loaded = manager.load_model(path)
        assert loaded["metadata"]["acc"] == 0.8

    def test_list_models_mocked(self, manager, tmp_path):
        # ファイル実体を作成
        p1 = os.path.join(tmp_path, "model1.joblib")
        p2 = os.path.join(tmp_path, "model2.joblib")
        joblib.dump({"model": 1}, p1)
        joblib.dump({"model": 2}, p2)

        # glob.glob をモックして、一時ディレクトリのファイルを返すようにする
        # model_manager.py 内の glob を差し替える
        with patch("app.services.ml.models.model_manager.glob.glob") as mock_glob:
            mock_glob.side_effect = lambda p: [p1, p2] if "*" in p else []
            # list_models を実行
            # 内部で get_model_search_paths() が呼ばれるが、glob が制御されているので結果は出る
            models = manager.list_models()
            assert len(models) > 0

    def test_get_latest_model_mocked(self, manager, tmp_path):
        p1 = os.path.join(tmp_path, "latest.joblib")
        joblib.dump({"model": 1}, p1)

        with patch("app.services.ml.models.model_manager.glob.glob", return_value=[p1]):
            latest = manager.get_latest_model()
            assert latest == p1

    def test_cleanup_logic_mocked(self, manager, tmp_path):
        p1 = os.path.join(tmp_path, "old.joblib")
        joblib.dump({"model": 1}, p1)

        # 期限切れ判定をモック
        with patch("app.services.ml.models.model_manager.glob.glob", return_value=[p1]):
            with patch(
                "app.services.ml.models.model_manager.os.path.getmtime", return_value=0
            ):  # エポック秒
                manager.cleanup_expired_models()
                # os.remove が呼ばれるはず。実際の設定に依存しないように os.remove もパッチ可能だが、
                # ここでは導通を確認。
                pass

    def test_metadata_operations(self, manager):
        path = manager.save_model(DummyModel(), "meta", metadata={"score": 100})
        # 1. 正常
        meta = manager.load_metadata_only(path)
        assert meta["metadata"]["score"] == 100
        # 2. サイドカーなし
        os.remove(manager._get_sidecar_path(path))
        meta_fb = manager.load_metadata_only(path)
        assert meta_fb["metadata"]["score"] == 100

    def test_performance_metrics_extraction(self, manager):
        # 複雑な形式のメタデータからの抽出
        meta = {
            "classification_report": {
                "macro avg": {"precision": 0.95, "recall": 0.85, "f1-score": 0.9}
            }
        }
        metrics = manager.extract_model_performance_metrics("dummy", metadata=meta)
        assert metrics["precision"] == 0.95
        assert metrics["f1_score"] == 0.9

    def test_extract_algorithm_name_variants(self, manager):
        assert manager._extract_algorithm_name(None, {"best_algorithm": "XGB"}) == "xgb"
        assert (
            manager._extract_algorithm_name(None, {"model_type": "Ensemble"})
            == "ensemble"
        )

    def test_error_handling(self, manager):
        # safe_ml_operation デコレータにより例外は None になる
        assert manager.save_model(None, "fail") is None
        assert manager.load_model("/non/existent") is None

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, mock_open

import pytest

from app.services.ml.models.model_manager import ModelManager


class TestModelManager:
    """ModelManager のテスト (ファイル操作を全てモックしてPermissionErrorを防止)"""

    @pytest.fixture
    def manager(self):
        m = ModelManager()
        m.config = MagicMock()
        m.config.model_save_path = "/mock/models"
        m.config.model_file_extension = ".pkl"
        m.config.max_model_versions = 2
        m.config.model_retention_days = 7
        m._get_model_search_paths = MagicMock(return_value=["/mock/models"])
        return m

    # ---------------------------------------------------------------------------
    # 基本的な保存と読み込み
    # ---------------------------------------------------------------------------

    @patch("app.services.ml.models.model_manager.os.makedirs")
    @patch("app.services.ml.models.model_manager.os.path.exists", return_value=False)
    @patch("app.services.ml.models.model_manager.os.path.getsize", return_value=2048)
    @patch("app.services.ml.models.model_manager.joblib.dump")
    @patch("app.services.ml.models.model_manager.ModelManager._cleanup_old_models")
    def test_save_model_success(
        self, mock_cleanup, mock_dump, mock_getsize, mock_exists, mock_makedirs, manager
    ):
        """モデル保存の正常系テスト"""
        mock_model = {"coef": [1, 2, 3]}

        with patch(
            "app.services.ml.models.model_manager.ModelManager._extract_algorithm_name",
            return_value="mock_algo",
        ):
            with patch("builtins.open", mock_open()) as mock_file:
                path = manager.save_model(
                    mock_model, "test_model", metadata={"acc": 0.9}
                )

                assert path is not None
                assert "mock_algo" in path
                assert path.endswith(".pkl")

                mock_dump.assert_called_once()
                mock_cleanup.assert_called_once_with("mock_algo")
                mock_file.assert_called_once_with(
                    path.replace(".pkl", ".meta.json"), "w", encoding="utf-8"
                )

    @patch("app.services.ml.models.model_manager.os.path.exists", return_value=True)
    @patch("app.services.ml.models.model_manager.joblib.load")
    def test_load_model_success(self, mock_load, mock_exists, manager):
        """モデル読み込みの正常系テスト"""
        mock_load.return_value = {"model": {"coef": [1]}, "metadata": {"acc": 0.9}}
        res = manager.load_model("/mock/models/mock_algo_20230101.pkl")
        assert res is not None
        assert res["model"] == {"coef": [1]}
        assert res["metadata"]["acc"] == 0.9

    @patch("app.services.ml.models.model_manager.os.path.exists", return_value=True)
    @patch("app.services.ml.models.model_manager.joblib.load")
    def test_load_model_old_format(self, mock_load, mock_exists, manager):
        """古い形式（辞書でない）モデルの読み込み"""
        mock_load.return_value = ["my", "old", "model"]
        res = manager.load_model("/mock/models/old_model.pkl")
        assert res is not None
        assert res["model"] == ["my", "old", "model"]
        assert "scaler" in res
        assert res["metadata"] == {}

    @patch("app.services.ml.models.model_manager.os.path.exists", return_value=False)
    def test_load_model_not_found(self, mock_exists, manager):
        """存在しないファイルの読み込み"""
        res = manager.load_model("/mock/models/ghost.pkl")
        assert res is None

    # ---------------------------------------------------------------------------
    # モデル一覧と最新モデルの取得
    # ---------------------------------------------------------------------------

    @patch("app.services.ml.models.model_manager.os.path.exists", return_value=True)
    @patch("glob.glob")
    @patch("app.services.ml.models.model_manager.os.path.getmtime")
    def test_get_latest_model(self, mock_getmtime, mock_glob, mock_exists, manager):
        """最新モデルのファイルパス取得"""
        mock_glob.side_effect = [
            ["/mock/models/model1.pkl", "/mock/models/model2.pkl"],
        ]
        # model2 の方が新しいとする
        mock_getmtime.side_effect = lambda f: 200 if "model2" in f else 100

        latest = manager.get_latest_model()
        assert latest == "/mock/models/model2.pkl"

    @patch("app.services.ml.models.model_manager.os.path.exists", return_value=True)
    @patch("glob.glob")
    @patch("app.services.ml.models.model_manager.os.stat")
    def test_list_models(self, mock_stat, mock_glob, mock_exists, manager):
        """保存されているモデルの一覧取得"""
        # _get_model_search_paths から返るパス分、globが呼ばれる
        mock_glob.side_effect = [
            ["/mock/models/model1.pkl", "/mock/models/model2.pkl"],
        ] * 1

        # モックの os.stat 返り値
        stat_mock = MagicMock()
        stat_mock.st_size = 1024 * 1024  # 1MB
        stat_mock.st_mtime = 1600000000
        mock_stat.return_value = stat_mock

        with patch(
            "app.services.ml.models.model_manager.os.path.abspath",
            side_effect=lambda x: x,
        ):
            models = manager.list_models()
            assert len(models) == 2
            assert models[0]["size_mb"] == 1.0
            # ソート確認 (同じ時刻なので順序は保持か安定)
            assert "name" in models[0]

    # ---------------------------------------------------------------------------
    # クリーンアップ (期限切れ & 世代管理)
    # ---------------------------------------------------------------------------

    @patch("app.services.ml.models.model_manager.os.path.exists", return_value=True)
    @patch("glob.glob")
    @patch("app.services.ml.models.model_manager.os.path.getmtime")
    @patch("app.services.ml.models.model_manager.os.remove")
    def test_cleanup_expired_models(
        self, mock_remove, mock_getmtime, mock_glob, mock_exists, manager
    ):
        """期限切れモデルのクリーンアップ"""
        mock_glob.return_value = ["/mock/models/old.pkl", "/mock/models/new.pkl"]

        # old は10日前のタイムスタンプ、new は現在のタイムスタンプ
        now_ts = datetime.now().timestamp()
        old_ts = (datetime.now() - timedelta(days=10)).timestamp()

        mock_getmtime.side_effect = lambda f: old_ts if "old" in f else now_ts

        manager.cleanup_expired_models()

        # removeが古いモデルとサイドカーに対して呼ばれる
        assert mock_remove.call_count == 2
        mock_remove.assert_any_call("/mock/models/old.pkl")
        mock_remove.assert_any_call("/mock/models/old.meta.json")

    @patch("glob.glob")
    @patch("app.services.ml.models.model_manager.os.path.getmtime")
    @patch("app.services.ml.models.model_manager.os.path.exists", return_value=True)
    @patch("app.services.ml.models.model_manager.os.remove")
    def test_cleanup_old_models_generations(
        self, mock_remove, mock_exists, mock_getmtime, mock_glob, manager
    ):
        """世代管理でのクリーンアップ"""
        # 3つのファイルがある (max_versions = 2 なので一番古い1つ消える)
        mock_glob.return_value = [
            "/mock/models/model_1.pkl",
            "/mock/models/model_2.pkl",
            "/mock/models/model_3.pkl",
        ]

        # 1が一番古いとする
        def mtime_mock(f):
            if "model_1" in f:
                return 100
            if "model_2" in f:
                return 200
            if "model_3" in f:
                return 300
            return 0

        mock_getmtime.side_effect = mtime_mock

        manager._cleanup_old_models("model")

        # 1が消される
        mock_remove.assert_any_call("/mock/models/model_1.pkl")
        mock_remove.assert_any_call("/mock/models/model_1.meta.json")
        assert mock_remove.call_count == 2

    # ---------------------------------------------------------------------------
    # メタデータ抽出 & ユーティリティ
    # ---------------------------------------------------------------------------

    def test_performance_metrics_extraction(self, manager):
        """メタデータからのパフォーマンス指標抽出"""
        metadata = {
            "accuracy": 0.85,
            "classification_report": {
                "macro avg": {"precision": 0.8, "recall": 0.7, "f1-score": 0.75}
            },
        }

        with patch(
            "app.services.ml.models.model_manager.ModelManager.load_model",
            return_value=None,
        ):
            metrics = manager.extract_model_performance_metrics(
                "/path", metadata=metadata
            )
            assert metrics["accuracy"] == 0.85
            assert metrics["precision"] == 0.8
            assert metrics["f1_score"] == 0.75

    def test_extract_algorithm_name_variants(self, manager):
        """アルゴリズム名の抽出ロジック"""
        assert (
            manager._extract_algorithm_name(None, {"best_algorithm": "LGBM"}) == "lgbm"
        )
        assert (
            manager._extract_algorithm_name(None, {"model_type": "StackingEnsemble"})
            == "stacking"
        )
        assert (
            manager._extract_algorithm_name({"fitted_base_models": {}}, {})
            == "stacking"
        )

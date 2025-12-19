import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import HTTPException
from app.services.ml.orchestration.ml_management_orchestration_service import MLManagementOrchestrationService

class TestMLManagementOrchestrationService:
    @pytest.fixture
    def service(self):
        return MLManagementOrchestrationService()

    @pytest.mark.asyncio
    async def test_get_formatted_models(self, service):
        """モデル一覧取得と整形"""
        mock_models = [
            {"name": "m1", "path": "p1", "size_mb": 1.0, "modified_at": MagicMock(), "directory": "d1"}
        ]
        with patch('app.services.ml.models.model_manager.model_manager.list_models', return_value=mock_models):
            with patch('app.services.ml.orchestration.ml_management_orchestration_service.load_model_metadata_safely', return_value={"metadata": {}}):
                res = await service.get_formatted_models()
                assert len(res["models"]) == 1
                assert res["models"][0]["id"] == "m1"

    @pytest.mark.asyncio
    async def test_delete_model_success(self, service):
        """モデル個別削除成功"""
        mock_models = [{"name": "m1", "path": "p1"}]
        with patch('app.services.ml.models.model_manager.model_manager.list_models', return_value=mock_models):
            with patch('os.remove') as mock_remove:
                res = await service.delete_model("m1")
                assert res["success"] is True
                mock_remove.assert_called_once_with("p1")

    @pytest.mark.asyncio
    async def test_delete_model_not_found(self, service):
        """存在しないモデルの削除"""
        with patch('app.services.ml.models.model_manager.model_manager.list_models', return_value=[]):
            with pytest.raises(HTTPException) as exc:
                await service.delete_model("ghost")
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_all_models(self, service):
        """全モデル削除"""
        mock_models = [
            {"name": "m1", "path": "p1"},
            {"name": "m2", "path": "p2"}
        ]
        with patch('app.services.ml.models.model_manager.model_manager.list_models', return_value=mock_models):
            with patch('os.remove') as mock_remove:
                res = await service.delete_all_models()
                assert res["deleted_count"] == 2
                assert mock_remove.call_count == 2

    @pytest.mark.asyncio
    async def test_get_ml_status_no_model(self, service):
        """モデルがない時のステータス"""
        with patch('app.services.ml.orchestration.ml_management_orchestration_service.get_latest_model_with_info', return_value=None):
            res = await service.get_ml_status()
            assert res["status"] == "no_model"
            assert res["is_model_loaded"] is False

    @pytest.mark.asyncio
    async def test_get_feature_importance(self, service):
        """特徴量重要度の取得"""
        mock_info = {
            "metadata": {"feature_importance": {"f1": 0.5, "f2": 0.3}}
        }
        with patch('app.services.ml.orchestration.ml_management_orchestration_service.get_latest_model_with_info', return_value=mock_info):
            res = await service.get_feature_importance(top_n=1)
            assert "f1" in res["feature_importance"]
            assert "f2" not in res["feature_importance"] # top_n=1

    @pytest.mark.asyncio
    async def test_load_model_flow(self, service):
        """モデル読み込みフロー"""
        mock_models = [{"name": "m1", "path": "p1"}]
        with patch('app.services.ml.models.model_manager.model_manager.list_models', return_value=mock_models):
            with patch('app.services.ml.orchestration.ml_management_orchestration_service.ml_training_service.load_model', return_value=True):
                with patch.object(service, 'get_current_model_info', return_value={"loaded": True}):
                    res = await service.load_model("m1")
                    assert res["success"] is True
                    assert "m1" in res["message"]

        @pytest.mark.asyncio

        async def test_delete_all_models_partial_fail(self, service):

            """一部のモデル削除に失敗した場合"""

            mock_models = [

                {"name": "m1", "path": "p1"},

                {"name": "m2", "path": "p2"}

            ]

            with patch('app.services.ml.models.model_manager.model_manager.list_models', return_value=mock_models):

                # p1 は成功、p2 は PermissionError

                with patch('os.remove', side_effect=[None, PermissionError("Denied")]):

                    res = await service.delete_all_models()

                    assert res["deleted_count"] == 1

                    assert res["failed_count"] == 1

                    assert "m2" in res["failed_models"]

    

        @pytest.mark.asyncio

        async def test_get_current_model_info_not_found(self, service):

            """現在のモデルファイルがディスク上にない場合"""

            with patch('app.services.ml.orchestration.ml_training_service.ml_training_service.get_current_model_path', return_value="ghost.pkl"):

                with patch('app.services.ml.orchestration.ml_training_service.ml_training_service.get_current_model_info', return_value={"model_type": "lgb"}):

                    with patch('os.path.exists', return_value=False):

                        res = await service.get_current_model_info()

                        assert res["loaded"] is True

                        assert res["last_updated"] is not None # 現在時刻が入る

    

        @pytest.mark.asyncio

        async def test_is_active_model_safety(self, service):

            """アクティブモデル判定の安全性テスト"""

            # pathキーがない不正な辞書

            assert service._is_active_model({}) is False

            

            # モデルロードなし時

            with patch('app.services.ml.orchestration.ml_training_service.ml_training_service.get_current_model_path', return_value=None):

                assert service._is_active_model({"path": "any"}) is False

    

        @pytest.mark.asyncio

        async def test_update_ml_config_fail(self, service):

            """設定更新失敗時のハンドリング"""

            with patch('app.services.ml.orchestration.ml_management_orchestration_service.ml_config_manager.update_config', return_value=False):

                res = await service.update_ml_config({})

                assert res["success"] is False

    